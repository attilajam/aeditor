import json
import os
from copy import deepcopy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from autoedit.audio import analyze_audio
from autoedit.config import Config
from autoedit.pipeline import build_plan
from autoedit.probe import MediaError
from autoedit.render import render_plan
from autoedit.review_planner import build_review_plan
from autoedit.validate import validate_user_timing, validate_video
from autoedit.video import analyze_clip
from generate_test_media import generate_media
from autoedit.review import initial_markers


@pytest.fixture(scope="module")
def review_inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("review-media")
    manifest = generate_media(root, duration=5.0)
    clips_dir = root / "clips"
    clips_dir.mkdir()
    sources = [Path(v["path"]) for v in manifest["videos"] if v["composition"] in {"similar-a", "different"}]
    cfg = Config(duration=4.0, audio_start=0.0, fps=24, size="160x240", reuse="forbid",
                 min_shot=3.0, max_shot=4.0, cache_dir=str(root / "cache"),
                 candidates_per_slot=12, beam_width=12, refine_top=2, refine_steps=1).validate()
    audio = analyze_audio(manifest["audio"]["rhythmic"], cfg)
    clips = [analyze_clip(path, cfg) for path in sources[:2]]
    base = build_plan(audio, clips, cfg)
    return root, cfg, base


def _adjustment_rows(base, frames):
    fps = base["configuration"]["fps"]
    originals = base["cut_frames"][1:-1]
    rows = []
    for i, frame in enumerate(frames[1:-1]):
        original = originals[i] if i < len(originals) else round(
            min(base["audio"]["beat_times"], key=lambda beat: abs(beat * fps - frame), default=frame / fps) * fps)
        rows.append({"id": f"cut-{i}", "kind": "manual", "detected_time": None,
                     "original_frame": original, "frame": frame, "enabled": True})
    return rows


def test_fixed_manual_nudge_renders_and_validates(review_inputs):
    root, _, base = review_inputs
    frames = [0, 49, 96]
    plan = build_review_plan(base, frames, adjustments=_adjustment_rows(base, frames))
    assert plan["cut_frames"] == frames
    assert plan["audio"]["beat_times"] == base["audio"]["beat_times"]
    assert len({segment["clip"] for segment in plan["segments"]}) == len(plan["segments"])
    assert validate_user_timing(plan)["valid"]
    output = root / "review.mp4"
    render_plan(plan, output)
    report = validate_video(plan, output)
    assert report["valid"], report["errors"]
    assert report["manual_adjustments"]


def test_review_upgrades_legacy_defaults_but_preserves_custom_weights_and_cut_frames(review_inputs):
    _, _, base = review_inputs
    legacy = deepcopy(base)
    legacy.pop("matching")
    legacy["configuration"].pop("edge_center_bias")
    old_weights = {"color": .25, "structure": .45, "edge": .15, "motion": .15}
    legacy["configuration"]["weights"].update(old_weights)
    frames = [0, 49, 96]
    plan = build_review_plan(legacy, frames, adjustments=_adjustment_rows(legacy, frames))
    assert plan["cut_frames"] == frames
    assert plan["matching"]["method"] == "spatial_unsigned_edge_pyramid"
    assert plan["configuration"]["weights"] == Config().weights
    assert any("upgraded" in warning for warning in plan["warnings"])

    legacy["configuration"]["weights"]["edge"] = .42
    custom = build_review_plan(legacy, frames, adjustments=_adjustment_rows(legacy, frames))
    assert custom["configuration"]["weights"]["edge"] == .42
    assert custom["configuration"]["weights"]["color"] == .25
    assert not any("upgraded" in warning for warning in custom["warnings"])


def test_malformed_timing_provenance_is_rejected(review_inputs):
    _, _, base = review_inputs
    frames = [0, 49, 96]
    plan = build_review_plan(base, frames, adjustments=_adjustment_rows(base, frames))
    plan["user_timing"]["adjustments"][0]["frame"] = 48
    report = validate_user_timing(plan)
    assert not report["valid"]
    assert "reconstruct" in report["errors"][0]


def test_initial_markers_and_forged_beat_provenance(review_inputs):
    _, _, base = review_inputs
    plan = deepcopy(base)
    markers = initial_markers(base)
    plan["user_timing"] = {"version": 1, "base_cut_frames": list(base["cut_frames"]),
        "adjustments": markers, "no_cut_ranges": [], "cut_frames": list(base["cut_frames"])}
    assert validate_user_timing(plan)["valid"]
    beat_marker = next(marker for marker in markers if marker["kind"] == "beat")
    beat_marker["detected_time"] += .1
    assert not validate_user_timing(plan)["valid"]


def test_unshifted_manual_marker_and_strict_no_cut_range(review_inputs):
    _, _, base = review_inputs
    manual = {"id": "manual-1", "kind": "manual", "detected_time": None,
              "original_frame": 49, "frame": 49, "enabled": True}
    plan = build_review_plan(base, [0, 49, 96], adjustments=[manual])
    report = validate_user_timing(plan)
    assert report["valid"] and report["manual_cut_frames"] == [49]
    with pytest.raises(MediaError, match="no_cut_range"):
        build_review_plan(base, [0, 49, 96], adjustments=[manual], no_cut_ranges=[[48, 50]])


def test_infeasible_fixed_schedule_is_not_changed(review_inputs):
    _, _, base = review_inputs
    base = deepcopy(base)
    base["sources"] = base["sources"][:1]
    base["configuration"]["reuse"] = "forbid"
    # One source cannot supply two separate slots when clip reuse is forbidden;
    # the planner must keep the requested split rather than regroup it.
    with pytest.raises(MediaError, match="fixed review schedule"):
        build_review_plan(base, [0, 48, 96])


def test_kick_markers_validate_and_survive_review_round_trip(review_inputs):
    _, _, base = review_inputs
    base = deepcopy(base)
    beats = base["audio"]["beat_times"]
    base["audio"]["beat_labels"] = [i % 2 == 0 for i in range(len(beats))]
    markers = initial_markers(base)
    kick = [marker for marker in markers if marker["kind"] == "kick"]
    assert kick, "expected kick-labeled beats to surface as kick markers"
    assert all(marker["id"].startswith("kick-") for marker in kick)
    assert all(marker["detected_time"] in beats for marker in kick)
    plan = deepcopy(base)
    plan["user_timing"] = {"version": 1, "base_cut_frames": list(base["cut_frames"]),
        "adjustments": markers, "no_cut_ranges": [], "cut_frames": list(base["cut_frames"])}
    assert validate_user_timing(plan)["valid"]
    forged = deepcopy(markers)
    next(marker for marker in forged if marker["kind"] == "kick")["detected_time"] += .1
    plan["user_timing"]["adjustments"] = forged
    assert not validate_user_timing(plan)["valid"]
    reviewed = build_review_plan(base, [0, 54, 96])
    assert "kick" in {item["kind"] for item in reviewed["user_timing"]["adjustments"]}
    assert validate_user_timing(reviewed)["valid"]
