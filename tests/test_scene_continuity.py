import json
import subprocess

import cv2
import numpy as np
import pytest

from autoedit.candidates import make_candidate
from autoedit.config import Config
from autoedit.features import DESCRIPTOR_VERSION, describe, legacy_visual_distance
from autoedit.video import SCENE_DETECTOR_VERSION, analyze_clip, detect_scene_boundaries


FPS = 24
WIDTH, HEIGHT = 160, 240
DURATION = 3
CUT_FRAME = 2 * FPS


def _same_palette_frames(cut=True):
    y, x = np.indices((HEIGHT, WIDTH))
    vertical = np.where((x // 10) % 2, 220, 30).astype(np.uint8)
    horizontal = np.where((y // 10) % 2, 220, 30).astype(np.uint8)
    a = cv2.cvtColor(vertical, cv2.COLOR_GRAY2BGR)
    b = cv2.cvtColor(horizontal if cut else vertical, cv2.COLOR_GRAY2BGR)
    return [a if i < CUT_FRAME else b for i in range(FPS * DURATION)]


def _write_video(path, frames):
    raw = b"".join(frame.tobytes() for frame in frames)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pixel_format", "bgr24",
         "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS), "-i", "pipe:0",
         "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12", "-pix_fmt", "yuv420p", str(path)],
        input=raw, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    return path


@pytest.fixture(scope="module")
def generated_videos(tmp_path_factory):
    root = tmp_path_factory.mktemp("scene-continuity")
    hardcut = _write_video(root / "same-palette-cut.mp4", _same_palette_frames(cut=True))
    continuous = _write_video(root / "sharp-continuous.mp4", _same_palette_frames(cut=False))
    return hardcut, continuous


def _config(cache_dir):
    return Config(size="160x240", fps=FPS, pacing="medium", min_shot=1.0, max_shot=2.0,
                  samples_per_second=6.0, analysis_long_side=120, cache_dir=str(cache_dir),
                  reuse="allow").validate()


def test_same_palette_composition_cut_detected_and_crossing_candidate_rejected(generated_videos, tmp_path):
    path, _ = generated_videos
    cfg = _config(tmp_path / "cache")
    clip = analyze_clip(path, cfg)

    # The old histogram + low-resolution structure score intentionally misses
    # this cut: each quadrant has the same color proportions, but pixels move.
    before, after = clip["frames"][10], clip["frames"][14]
    old = legacy_visual_distance(describe(before), describe(after))
    old_score = 0.45 * old["color"] + 0.45 * old["structure"] + 0.10 * old["edge"]
    pixel_mae = float(np.abs(before.astype(np.float32) - after.astype(np.float32)).mean() / 255)
    assert old_score < cfg.scene_threshold
    assert pixel_mae > 0.1

    boundaries = detect_scene_boundaries(clip["frames"], clip["times"], clip["probe"], cfg)
    assert boundaries
    assert any(start < 2.0 < end or abs((start + end) / 2 - 2.0) < 0.1 for start, end in boundaries)
    assert clip["scene_detector_version"] == SCENE_DETECTOR_VERSION == 2

    # Source interval [1.5, 2.5) crosses the detected internal hard cut.
    assert make_candidate(0, clip, 1.5, 1.0, 0, FPS, cfg, allow_cuts=False) is None


def test_sharp_but_continuous_footage_remains_one_usable_region(generated_videos, tmp_path):
    _, path = generated_videos
    cfg = _config(tmp_path / "cache")
    clip = analyze_clip(path, cfg)
    assert clip["scene_boundaries"] == []
    # High edge detail alone should not split or disqualify a continuous shot.
    assert max(d["quality"]["sharpness"] for d in clip["descriptors"]) > 0.001
    assert make_candidate(0, clip, 0.5, 1.0, 0, 2 * FPS, cfg, allow_cuts=False) is not None


def test_old_scene_cache_migrates_from_cached_frames_without_video_decode(generated_videos, tmp_path, monkeypatch):
    path, _ = generated_videos
    cache = tmp_path / "cache"
    cfg = _config(cache)
    original = analyze_clip(path, cfg)
    metadata_path = next(p for p in cache.glob("*.json") if json.loads(p.read_text()).get("identity", {}).get("path") == str(path.resolve()))
    metadata = json.loads(metadata_path.read_text())
    metadata["scene_detector_version"] = 1
    metadata["scene_boundaries"] = []
    metadata_path.write_text(json.dumps(metadata))

    def forbidden_decode(*args, **kwargs):
        raise AssertionError("old scene cache migration must reuse stored analysis frames")

    monkeypatch.setattr("autoedit.video.decode_samples", forbidden_decode)
    migrated = analyze_clip(path, cfg)
    assert migrated["scene_detector_version"] == 2
    assert migrated["scene_boundaries"]
    cut_start, cut_end = migrated["scene_boundaries"][0]
    near_cut = [motion for t, motion in zip(migrated["times"], migrated["motions"])
                if cut_start - 0.2 <= t <= cut_end + 0.2]
    assert near_cut
    assert all(motion["reliability"] == 0 for motion in near_cut)
    saved = json.loads(metadata_path.read_text())
    assert saved["scene_detector_version"] == 2
    assert saved["scene_boundaries"]


def test_old_descriptors_migrate_without_redecoding_or_changing_scene_detection(generated_videos, tmp_path, monkeypatch):
    path, _ = generated_videos
    cache = tmp_path / "cache"
    cfg = _config(cache)
    original = analyze_clip(path, cfg)
    metadata_path = next(cache.glob("*.json"))
    metadata = json.loads(metadata_path.read_text())
    metadata.pop("descriptor_version")
    for descriptor in metadata["descriptors"]:
        descriptor.pop("oriented_edges")
        descriptor.pop("descriptor_version")
    metadata_path.write_text(json.dumps(metadata))

    def forbidden_decode(*args, **kwargs):
        raise AssertionError("descriptor migration must reuse stored analysis frames")

    monkeypatch.setattr("autoedit.video.decode_samples", forbidden_decode)
    migrated = analyze_clip(path, cfg)
    assert migrated["descriptor_version"] == DESCRIPTOR_VERSION
    assert all("oriented_edges" in descriptor for descriptor in migrated["descriptors"])
    assert migrated["scene_boundaries"] == original["scene_boundaries"]
    assert migrated["motions"] == original["motions"]
    assert migrated["pts"] == original["pts"]
    assert json.loads(metadata_path.read_text())["descriptor_version"] == DESCRIPTOR_VERSION
