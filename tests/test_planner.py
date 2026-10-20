"""Focused checks for feasible candidates and bounded path planning."""

from copy import deepcopy

import cv2
import numpy as np

from autoedit.config import Config
from autoedit.features import boundary_quality, describe
from autoedit.models import Candidate
from autoedit import planner
from autoedit.candidates import make_candidate, prune_diverse
from autoedit.planner import beam_search, usage_cost
from autoedit.video import visible_source_index


def _clip(image=None, *, duration=4.0, pts=None, scene_boundaries=None, samples=None):
    if image is None:
        image = np.full((64, 96, 3), (50, 100, 180), dtype=np.uint8)
        cv2.rectangle(image, (12, 12), (82, 52), (240, 220, 80), -1)
    desc = describe(image)
    pts = pts or list(np.arange(0, duration, 1 / 30))
    samples = samples or list(pts)
    return {
        "duration": duration,
        "pts": list(pts),
        "times": list(samples),
        "descriptors": [deepcopy(desc) for _ in samples],
        "motions": [{"vx": 0., "vy": 0., "magnitude": 0., "reliability": 0.} for _ in samples],
        "scene_boundaries": scene_boundaries or [],
        "reference_sharpness": desc["quality"]["sharpness"],
    }


def _candidate(clip_id, start, cost=0., *, count=10, fps=10, speed=1., tag=None, end=None):
    end = end if end is not None else start + count / fps * speed
    motion = {"vx": 0., "vy": 0., "magnitude": 0., "reliability": 0.}
    return Candidate(clip_id, start, end, speed, 0, count,
                     {"tag": tag}, {"tag": tag}, motion, motion,
                     {"entry_penalty": 0., "exit_penalty": 0., "interior_penalty": 0.},
                     start, end, segment_cost=cost)


def test_candidate_output_duration_and_visible_source_end_are_valid():
    cfg = Config(fps=10, speed_min=0.8, speed_max=1.2).validate()
    clip = _clip(duration=3.)
    candidate = make_candidate(0, clip, 0.2, 1.1, 0, 11, cfg)
    assert candidate is not None
    assert candidate.output_frame_count / cfg.fps == 1.1
    assert candidate.source_end - candidate.source_start == candidate.output_frame_count / cfg.fps * candidate.speed
    assert candidate.entry_pts < candidate.source_end
    assert candidate.exit_pts < candidate.source_end
    assert candidate.entry_pts == clip["pts"][visible_source_index(clip["pts"], .2, 0, 1.1, cfg.fps)]
    assert candidate.exit_pts == clip["pts"][visible_source_index(clip["pts"], .2, 10, 1.1, cfg.fps)]


def test_vfr_mapping_uses_actual_presentation_timestamps():
    # At output frame 1 (100 ms), the last available VFR frame is at 40 ms.
    # A nominal 30 FPS index calculation would incorrectly choose a later frame.
    pts = [0.0, 0.04, 0.13, 0.17, 0.31]
    assert visible_source_index(pts, 0.0, 1, 1.0, 10) == 1
    assert visible_source_index(pts, 0.0, 2, 1.0, 10) == 3


def test_internal_source_cut_is_avoided_unless_explicitly_relaxed():
    cfg = Config(fps=10, speed_min=1., speed_max=1.).validate()
    clip = _clip(duration=3., scene_boundaries=[[.8, .9]])
    assert make_candidate(0, clip, .2, 1., 0, 10, cfg) is None
    relaxed = make_candidate(0, clip, .2, 1., 0, 10, cfg, allow_cuts=True)
    assert relaxed is not None
    assert relaxed.warnings == ["segment crosses an estimated internal scene cut"]


def test_reuse_forbid_applies_across_the_complete_path():
    cfg = Config(reuse="forbid").validate()
    first, second, third = _candidate(0, 0), _candidate(1, 0), _candidate(2, 0)
    assert usage_cost((first,), _candidate(0, 2), cfg) is None
    result = beam_search([[first, second, third], [_candidate(0, 2), _candidate(1, 2), _candidate(2, 2)],
                          [_candidate(0, 4), _candidate(1, 4), _candidate(2, 4)]], cfg, keep=10)
    assert result
    assert all(len({candidate.clip for candidate in state.path}) == 3 for state in result)


def test_reuse_and_source_overlap_penalties_are_counted():
    cfg = Config(reuse="allow", weights={**Config().weights, "repeat": .3, "overlap": 2.}).validate()
    previous = _candidate(4, 0., end=2.)
    current = _candidate(4, 1., end=3.)
    # One prior use adds repeat=.3; half of this two-second trim overlaps.
    assert usage_cost((previous,), current, cfg) == .3 + 2. * .5
    cfg.reuse = "forbid"
    assert usage_cost((previous,), current, cfg) is None


def test_pruning_preserves_clip_diversity_and_is_seeded():
    candidates = [_candidate(clip, start, cost=float(start) / 10)
                  for clip in range(3) for start in range(5)]
    first = prune_diverse(candidates, 6, np.random.default_rng(19))
    second = prune_diverse(candidates, 6, np.random.default_rng(19))
    assert [id(c) for c in first] == [id(c) for c in second]
    assert {c.clip for c in first} == {0, 1, 2}


def test_black_boundary_is_penalized_and_valid_candidate_costs_less():
    cfg = Config(fps=10, speed_min=1., speed_max=1.).validate()
    black_image = np.zeros((64, 96, 3), dtype=np.uint8)
    valid_image = np.full((64, 96, 3), (60, 100, 170), dtype=np.uint8)
    cv2.circle(valid_image, (48, 32), 20, (250, 235, 70), -1)
    black_desc, valid_desc = describe(black_image), describe(valid_image)
    assert boundary_quality(black_desc) > boundary_quality(valid_desc)
    black_clip = _clip(black_image)
    valid_clip = _clip(valid_image)
    black_candidate = make_candidate(0, black_clip, .2, 1., 0, 10, cfg)
    valid_candidate = make_candidate(1, valid_clip, .2, 1., 0, 10, cfg)
    assert black_candidate.segment_cost > valid_candidate.segment_cost


def test_beam_width_two_finds_better_complete_path_than_greedy():
    # Greedy takes A because its first-slot cost is lowest, then gets trapped
    # with A->Y at cost 1. A beam of two retains B and finds B->X at cost .1.
    a, b = _candidate(0, 0, tag="A"), _candidate(1, 0, cost=.1, tag="B")
    x, y = _candidate(2, 0, tag="X"), _candidate(3, 0, cost=1., tag="Y")
    graph = {("A", "X"): 10., ("A", "Y"): 0., ("B", "X"): 0., ("B", "Y"): 10.}
    original = planner.transition_components
    planner.transition_components = lambda previous, current, cfg: {"weighted": graph[(previous.exit["tag"], current.entry["tag"])]}
    try:
        cfg = Config(beam_width=2, refine_top=2, reuse="allow").validate()
        states = beam_search([[a, b], [x, y]], cfg, keep=2)
        greedy_first = min([a, b], key=lambda c: c.segment_cost)
        greedy_second = min([x, y], key=lambda c: c.segment_cost + graph[(greedy_first.exit["tag"], c.entry["tag"])])
        greedy_cost = greedy_first.segment_cost + greedy_second.segment_cost + graph[(greedy_first.exit["tag"], greedy_second.entry["tag"])]
        assert greedy_cost == 1.
        assert states[0].path == (b, x)
        assert states[0].cost < greedy_cost
        assert cfg.beam_width >= 2
    finally:
        planner.transition_components = original
