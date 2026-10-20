"""Real scene changes, weak audio, fit rules, cache invalidation and configuration."""
from pathlib import Path
import json
import subprocess
import cv2
import numpy as np
import pytest

from autoedit.audio import analyze_audio, choose_boundaries
from autoedit.cache import key_for
from autoedit.config import Config, load_config
from autoedit.probe import fingerprint, fit_filter, probe
from autoedit.video import analyze_clip, composition_descriptor


def test_configuration_validation_and_cli_precedence(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("fps: 24\nsize: 160x240\nweights:\n  motion: 0.4\n")
    cfg = load_config(path, {"fps": 30, "fit": "cover"})
    assert cfg.fps == 30 and cfg.fit == "cover" and cfg.weights["motion"] == .4
    assert cfg.weights["structure"] == .07
    with pytest.raises(ValueError, match="even"):
        Config(size="161x240").validate()
    with pytest.raises(ValueError, match="edge_center_bias"):
        Config(edge_center_bias=1.1).validate()
    path.write_text("unknown_key: 3")
    with pytest.raises(ValueError, match="unknown"):
        load_config(path)


def test_content_and_analysis_settings_invalidate_cache(tmp_path):
    path = tmp_path / "source"
    path.write_bytes(b"first")
    identity = fingerprint(path)
    key = key_for(identity, {"fit": "contain"}, "video")
    assert key != key_for(identity, {"fit": "cover"}, "video")
    path.write_bytes(b"other")
    assert key != key_for(fingerprint(path), {"fit": "contain"}, "video")


def test_real_internal_cut_and_quality_ignore_contain_padding(tmp_path):
    path = tmp_path / "two scenes.mp4"
    frames = []
    for i in range(48):
        frame = np.full((80, 160, 3), (30, 80, 180) if i < 24 else (190, 140, 25), np.uint8)
        cv2.circle(frame, (45 if i < 24 else 115, 40), 22, (225, 230, 235), -1)
        cv2.putText(frame, str(i), (4, 75), cv2.FONT_HERSHEY_SIMPLEX, .5, (20, 20, 20), 1)
        frames.append(frame)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "160x80", "-r", "24",
                    "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
                   input=b"".join(f.tobytes() for f in frames), capture_output=True, check=True)
    cfg = Config(size="160x240", fps=24, cache_dir=str(tmp_path / "cache"))
    clip = analyze_clip(path, cfg)
    assert any(a <= 1. <= b for a, b in clip["scene_boundaries"])
    assert clip["descriptors"][0]["quality"]["black"] < .1
    cached = analyze_clip(path, cfg)
    assert np.array_equal(cached["frames"], clip["frames"])
    assert cached["scene_boundaries"] == clip["scene_boundaries"]
    cover = analyze_clip(path, Config(size="160x240", fps=24, fit="cover", cache_dir=cfg.cache_dir))
    assert not np.array_equal(cover["frames"], clip["frames"])


def test_energy_changes_preferred_grouping_without_moving_beats():
    cfg = Config()
    base = {"fps": 30, "total_frames": 180, "fallback": "none", "beat_times": list(np.arange(.25, 6., .25)),
            "energy_times": [0, 6], "warnings": []}
    quiet, _ = choose_boundaries({**base, "energy": [0, 0]}, cfg)
    loud, _ = choose_boundaries({**base, "energy": [1, 1]}, cfg)
    assert len(loud) > len(quiet)
    assert set(loud[1:-1]) <= {round(t*30) for t in base["beat_times"]}


def test_weak_rhythm_produces_an_explicit_method(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from generate_test_media import generate_audio
    path = tmp_path / "weak.wav"
    generate_audio(path, duration=6, bpm=93, offset=.41, kind="weak")
    result = analyze_audio(path, Config(duration=6, cache_dir=str(tmp_path/"cache")))
    assert result["fallback"] in {"none", "onsets", "regular"}
    assert "not a calibrated probability" in result["reliability"]["heuristic"]
    cuts, warnings = choose_boundaries(result, Config())
    assert cuts[0] == 0 and cuts[-1] == 180
    if result["fallback"] != "none":
        assert warnings
