"""Subprocess and media-boundary integration checks for the complete editor."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from autoedit.audio import analyze_audio
from autoedit.candidates import make_candidate
from autoedit.config import Config
from autoedit.features import describe
from autoedit.pipeline import build_plan, make_plan
from autoedit.probe import frame_timestamps, probe, run
from autoedit.validate import audio_alignment, decode_frames, decoded_frame_count
from autoedit.video import analyze_clip
from generate_test_media import generate_media


@pytest.fixture(scope="session")
def media(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic media with spaces")
    return root, generate_media(root, duration=5.0)


def _env():
    env = os.environ.copy()
    source = str(ROOT / "src")
    env["PYTHONPATH"] = source + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _cli(*args, timeout=90):
    return subprocess.run([sys.executable, "-m", "autoedit.cli", *map(str, args)], cwd=ROOT,
                          env=_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, timeout=timeout, check=False)


def _assert_cli_ok(result):
    assert result.returncode == 0, f"CLI failed ({result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def _create_args(audio, clips, output, cache, *extra):
    return ("create", "--audio", audio, "--clips", clips, "--output", output,
            "--size", "160x240", "--fps", "24", "--duration", "4",
            "--audio-start", "0.375", "--seed", "17", "--cache-dir", cache, *extra)


def _decoded_source_match(plan, video):
    """Compare every output frame to native source frames mapped from ffprobe PTS.

    The expected mapping is recomputed from timestamps and the segment trim and
    speed; saved descriptor/sample indices are not used.
    """
    cfg = Config(**plan["configuration"])
    dimensions = cfg.dimensions
    output = decode_frames(video, dimensions)
    assert len(output) == plan["output"]["total_frames"]
    actual_by_clip = {}
    times_by_clip = {}
    for clip_id, source in enumerate(plan["sources"]):
        source_path = source["identity"]["path"]
        pts, _, _ = frame_timestamps(source_path)
        native = decode_frames(source_path, dimensions, fit=cfg.fit)
        assert len(pts) == len(native)
        times_by_clip[clip_id] = pts
        actual_by_clip[clip_id] = native

    worst = 0.0
    for segment in plan["segments"]:
        clip_id = segment["clip"]
        pts = times_by_clip[clip_id]
        native = actual_by_clip[clip_id]
        first = bisect_left(pts, segment["source_start"] - 1e-9)
        for local in range(segment["output_frame_count"]):
            target = segment["source_start"] + local / cfg.fps * segment["speed"]
            expected = max(first, bisect_right(pts, target + 1e-9) - 1)
            expected = min(expected, len(pts) - 1)
            nearby = range(max(first, expected - 2), min(len(pts), expected + 3))
            out = output[segment["output_start_frame"] + local].astype(np.float32)
            errors = [(float(np.mean(np.abs(out - native[index].astype(np.float32))) / 255), index)
                      for index in nearby]
            error, best_index = min(errors)
            worst = max(worst, error)
            assert error < .075, (f"output frame {segment['output_start_frame']+local} differs from all nearby native frames; "
                                  f"best MAE={error:.4f}, expected native index={expected}")
            assert abs(best_index - expected) <= 1, (f"rendered source mapping differs by more than one native frame: "
                                                      f"expected {expected}, best {best_index}")
    return worst


def _assert_render_is_sound(plan, video, dimensions=(160, 240), fps=24):
    info = probe(video)
    stream, audio = info["video"], info["audio"]
    assert stream["codec_name"] == "h264"
    assert stream["pix_fmt"] == "yuv420p"
    assert (stream["width"], stream["height"]) == dimensions
    assert abs(float(Fraction(stream["avg_frame_rate"])) - fps) < 1e-6
    assert audio and audio["codec_name"] == "aac"
    assert decoded_frame_count(video) == plan["output"]["total_frames"]
    run(["ffmpeg", "-v", "error", "-xerror", "-i", video, "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"])
    pts, duration, _ = frame_timestamps(video)
    assert len(pts) == plan["output"]["total_frames"]
    assert all(abs(t - i / fps) < 1 / 90_000 + 1e-6 for i, t in enumerate(pts))
    assert abs(duration - plan["output"]["total_frames"] / fps) <= 1 / fps
    alignment = audio_alignment(plan, video)
    assert alignment["valid"], alignment
    return alignment


def test_cli_create_validate_and_replay_are_frame_and_audio_stable(media, tmp_path):
    root, manifest = media
    audio = manifest["audio"]["rhythmic"]
    source = Path(next(v["path"] for v in manifest["videos"] if v["composition"] == "similar-a"))
    clips = root / "single input clip with spaces"
    clips.mkdir()
    copied_source = clips / "source with spaces.mp4"
    shutil.copy2(source, copied_source)
    output_dir = root / "rendered outputs with spaces"
    output_dir.mkdir()
    output = output_dir / "finished edit with spaces.mp4"
    cache = root / "cache with spaces"

    created = _cli(*_create_args(audio, clips, output, cache))
    _assert_cli_ok(created)
    plan_path = output.with_suffix(".plan.json")
    plan = json.loads(plan_path.read_text())
    assert plan["sources"][0]["identity"]["path"] == str(copied_source.resolve())
    assert plan["audio"]["identity"]["path"] == str(Path(audio).resolve())
    assert plan["configuration"]["seed"] == 17
    assert plan["audio"]["audio_start"] == .375
    assert plan["output"]["total_frames"] == 96
    alignment = _assert_render_is_sound(plan, output)
    assert alignment["correlation"] >= .70
    assert abs(alignment["lag_seconds"]) <= .05
    _decoded_source_match(plan, output)

    # The saved plan replays only its recorded choices and yields the same
    # decoded output frames as create.
    replay = output_dir / "replayed edit.mp4"
    replayed = _cli("render", "--plan", plan_path, "--output", replay)
    _assert_cli_ok(replayed)
    first = decode_frames(output, (160, 240))
    second = decode_frames(replay, (160, 240))
    assert first.shape == second.shape
    assert np.array_equal(first, second)

    validation = _cli("validate", "--plan", plan_path, "--video", output)
    _assert_cli_ok(validation)
    report = json.loads(output.with_suffix(".validation.json").read_text())
    assert report["valid"]

    # Existing destinations are protected before the expensive analysis starts.
    overwrite = _cli(*_create_args(audio, clips, output, cache))
    assert overwrite.returncode == 2
    assert "destination already exists" in overwrite.stderr

    missing_plan = json.loads(plan_path.read_text())
    missing_plan["sources"][0]["identity"]["path"] = str(root / "missing source with spaces.mp4")
    missing_path = root / "missing source plan.json"
    missing_path.write_text(json.dumps(missing_plan))
    missing_result = _cli("render", "--plan", missing_path, "--output", root / "missing source output.mp4")
    assert missing_result.returncode == 2
    assert "missing source file" in missing_result.stderr

    changed_source = root / "changed source with spaces.mp4"
    shutil.copy2(copied_source, changed_source)
    changed_source.write_bytes(changed_source.read_bytes() + b"changed")
    changed_plan = json.loads(plan_path.read_text())
    changed_plan["sources"][0]["identity"]["path"] = str(changed_source)
    changed_path = root / "changed source plan.json"
    changed_path.write_text(json.dumps(changed_plan))
    changed_result = _cli("render", "--plan", changed_path, "--output", root / "changed source output.mp4")
    assert changed_result.returncode == 2
    assert "source changed since analysis" in changed_result.stderr

    too_long = _cli("analyze", "--audio", audio, "--clips", clips, "--output", root / "too long.json",
                    "--size", "160x240", "--fps", "24", "--duration", "6", "--cache-dir", cache)
    assert too_long.returncode == 2
    assert "exceeds available audio" in too_long.stderr


def test_cli_render_forced_vfr_rotation_and_mixed_source_fps(media, tmp_path):
    root, manifest = media
    video_specs = [
        next(v for v in manifest["videos"] if v["composition"] == "similar-a"),
        next(v for v in manifest["videos"] if v["composition"] == "similar-b"),
        next(v for v in manifest["videos"] if v["vfr"]),
        next(v for v in manifest["videos"] if v["rotation"] == 90),
    ]
    cache = root / "manual plan cache"
    cfg = Config(duration=4.0, audio_start=.25, fps=24, size="160x240", reuse="allow",
                 cache_dir=str(cache), candidates_per_slot=8, beam_width=8, refine_top=1, refine_steps=1)
    audio = analyze_audio(manifest["audio"]["rhythmic"], cfg)
    clips = [analyze_clip(spec["path"], cfg) for spec in video_specs]
    assert [round(float(Fraction(c["probe"]["video"]["avg_frame_rate"]))) for c in clips[:2]] == [24, 30]
    assert clips[2]["probe"]["video"]["avg_frame_rate"]
    assert any(b - a != pytest.approx(1 / 24) for a, b in zip(clips[2]["pts"], clips[2]["pts"][1:]))
    assert clips[3]["probe"]["rotation"] == 90
    assert clips[0]["probe"]["audio"] is not None
    assert all(c["probe"]["audio"] is None for c in clips[1:])

    boundaries = [0, 24, 48, 72, 96]
    path = []
    speeds = [.85, 1.15, .95, 1.10]
    for slot, (clip, speed) in enumerate(zip(clips, speeds)):
        candidate = make_candidate(slot, clip, .25, speed, boundaries[slot], 24, cfg)
        assert candidate is not None
        path.append(candidate)
    plan = make_plan(audio, clips, cfg, boundaries, path)
    plan_path = root / "manual heterogeneous plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    output = tmp_path / "manual heterogeneous output.mp4"
    rendered = _cli("render", "--plan", plan_path, "--output", output)
    _assert_cli_ok(rendered)
    alignment = _assert_render_is_sound(plan, output)
    assert alignment["valid"]
    assert _decoded_source_match(plan, output) < .075
    rendered_frames = decode_frames(output, cfg.dimensions)
    for cut in boundaries[1:-1]:
        for index in (cut - 1, cut):
            gray = cv2.cvtColor(rendered_frames[index], cv2.COLOR_BGR2GRAY)
            assert float(np.mean(gray < 12)) < .98, f"unexpected inserted black frame at join index {index}"


def test_short_single_clip_fills_duration_only_with_explicit_reuse(media, tmp_path):
    root, manifest = media
    short = next(v["path"] for v in manifest["videos"] if v["composition"] == "short")
    audio = manifest["audio"]["rhythmic"]
    clips = root / "one short usable clip"
    clips.mkdir()
    shutil.copy2(short, clips / "only short source.mp4")
    output = tmp_path / "short source repeated.mp4"
    cache = root / "short clip cache"
    result = _cli(*_create_args(audio, clips, output, cache, "--reuse", "allow"))
    _assert_cli_ok(result)
    plan = json.loads(output.with_suffix(".plan.json").read_text())
    cap = .7 / plan["configuration"]["speed_min"]
    slot_lengths = [(b - a) / plan["output"]["fps"]
                    for a, b in zip(plan["cut_frames"], plan["cut_frames"][1:])]
    assert plan["configuration"]["reuse"] == "allow"
    assert len(slot_lengths) >= 2
    assert max(slot_lengths) <= cap + 1 / plan["output"]["fps"]
    assert decoded_frame_count(output) == plan["output"]["total_frames"]
    run(["ffmpeg", "-v", "error", "-xerror", "-i", output, "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"])


def test_empty_input_and_forbidden_short_footage_fail(media, tmp_path):
    root, manifest = media
    audio = manifest["audio"]["rhythmic"]
    cache = root / "failure cache"
    empty = tmp_path / "empty clips directory"
    empty.mkdir()
    empty_result = _cli(*_create_args(audio, empty, tmp_path / "empty result.mp4", cache))
    assert empty_result.returncode == 2
    assert "no usable footage" in empty_result.stderr

    short = next(v["path"] for v in manifest["videos"] if v["composition"] == "short")
    clips = tmp_path / "forbidden short footage"
    clips.mkdir()
    shutil.copy2(short, clips / "short.mp4")
    # No reuse option means the new default: every selected shot uses a
    # distinct source clip, so one short source cannot cover this soundtrack.
    forbidden = _cli(*_create_args(audio, clips, tmp_path / "forbidden result.mp4", cache))
    assert forbidden.returncode == 2
    assert "insufficient total footage" in forbidden.stderr


def test_default_reuse_forbid_records_unique_source_usage(media, tmp_path):
    root, manifest = media
    audio = manifest["audio"]["rhythmic"]
    clips = tmp_path / "two distinct sources"
    clips.mkdir()
    for composition in ("similar-a", "similar-b"):
        spec = next(v for v in manifest["videos"] if v["composition"] == composition)
        shutil.copy2(spec["path"], clips / (composition + ".mp4"))
    output = tmp_path / "no reused sources.mp4"
    result = _cli(*_create_args(audio, clips, output, tmp_path / "no reuse cache", "--pacing", "fast"))
    _assert_cli_ok(result)
    plan = json.loads(output.with_suffix(".plan.json").read_text())
    selected = [segment["clip"] for segment in plan["segments"]]
    assert plan["configuration"]["reuse"] == "forbid"
    assert len(selected) == len(set(selected))


def test_default_dimensions_and_reuse_follow_native_sources(media, tmp_path):
    root, manifest = media
    audio = manifest["audio"]["rhythmic"]
    clips = tmp_path / "native landscape footage"
    clips.mkdir()
    for composition in ("similar-a", "similar-b"):
        spec = next(v for v in manifest["videos"] if v["composition"] == composition)
        shutil.copy2(spec["path"], clips / (composition + ".mp4"))
    output = tmp_path / "native aspect edit.mp4"
    result = _cli("create", "--audio", audio, "--clips", clips, "--output", output,
                  "--duration", "2", "--fps", "24", "--cache-dir", tmp_path / "native aspect cache")
    _assert_cli_ok(result)
    plan = json.loads(output.with_suffix(".plan.json").read_text())
    assert plan["configuration"]["size"] == "240x160"
    assert (plan["output"]["width"], plan["output"]["height"]) == (240, 160)
    assert plan["configuration"]["reuse"] == "forbid"
    chosen = [segment["clip"] for segment in plan["segments"]]
    assert len(chosen) == len(set(chosen))
    _assert_render_is_sound(plan, output, dimensions=(240, 160), fps=24)


def test_resolution_stretches_full_frame_and_overrides_config(media, tmp_path):
    from autoedit.features import visual_distance
    from autoedit.video import analysis_dimensions

    root, manifest = media
    audio = manifest["audio"]["rhythmic"]
    source_spec = next(v for v in manifest["videos"] if v["composition"] == "similar-a")
    clips = tmp_path / "counter aspect source"
    clips.mkdir()
    source = clips / "landscape source.mp4"
    shutil.copy2(source_spec["path"], source)
    config = tmp_path / "legacy canvas config.yaml"
    config.write_text("size: 160x240\nfit: cover\n")
    output = tmp_path / "stretched portrait output.mp4"
    result = _cli("create", "--audio", audio, "--clips", clips, "--output", output,
                  "--config", config, "--resolution", "160x240", "--duration", "2", "--fps", "24",
                  "--cache-dir", tmp_path / "stretch cache")
    _assert_cli_ok(result)
    plan = json.loads(output.with_suffix(".plan.json").read_text())
    assert plan["configuration"]["size"] == "160x240"
    assert plan["configuration"]["fit"] == "stretch"
    assert (plan["output"]["width"], plan["output"]["height"]) == (160, 240)
    assert plan["configuration"]["reuse"] == "forbid"
    _assert_render_is_sound(plan, output, dimensions=(160, 240), fps=24)
    _decoded_source_match(plan, output)

    # Compare the selected boundary descriptors with frames decoded from the
    # rendered, stretched canvas. This catches descriptor code that still uses
    # contain padding or a centered crop when the render uses full-frame stretch.
    cfg = Config(**plan["configuration"])
    low_res = analysis_dimensions(cfg)
    output_frames = decode_frames(output, low_res)
    for segment in plan["segments"]:
        for key, local_index in (("entry", 0), ("exit", segment["output_frame_count"]-1)):
            absolute_index = segment["output_start_frame"] + local_index
            observed = describe(output_frames[absolute_index])
            intended = segment[key]
            assert max(visual_distance(intended, observed).values()) < .15


def test_cli_analyze_scans_all_fixture_variants_and_records_corrupt_input(media, tmp_path):
    root, manifest = media
    output = tmp_path / "complete fixture analysis.json"
    result = _cli("analyze", "--audio", manifest["audio"]["rhythmic"], "--clips", root / "video fixtures",
                  "--output", output, "--size", "160x240", "--fps", "24", "--duration", "4",
                  "--audio-start", ".375", "--seed", "17", "--cache-dir", root / "complete analysis cache")
    _assert_cli_ok(result)
    analysis = json.loads(output.read_text())
    valid_names = {Path(source["identity"]["path"]).name for source in analysis["sources"]}
    assert len(analysis["sources"]) == 9
    assert "corrupt.mp4" not in valid_names
    assert len(analysis["skipped_sources"]) == 1
    assert "corrupt.mp4" in Path(analysis["skipped_sources"][0]["path"]).name
    assert any(source["probe"]["audio"] is not None for source in analysis["sources"])
    assert any("rotated metadata" in name for name in valid_names)
    assert any("variable frame rate" in name for name in valid_names)


def test_fixed_seed_repeats_actual_analysis_and_refined_selection(media):
    root, manifest = media
    cfg = Config(duration=2.0, audio_start=.375, fps=24, size="160x240", pacing="fast",
                 cache_dir=str(root / "determinism cache"), seed=29,
                 candidates_per_slot=16, beam_width=8, refine_top=2)
    paths = [v["path"] for v in manifest["videos"] if v["composition"] in {"similar-a", "similar-b"}]
    clips = [analyze_clip(path, cfg) for path in paths]
    audio = analyze_audio(manifest["audio"]["rhythmic"], cfg)
    first = build_plan(audio, clips, cfg)
    second = build_plan(audio, clips, cfg)
    assert first["cut_frames"] == second["cut_frames"]
    assert first["segments"] == second["segments"]
    assert first["scores"] == second["scores"]
