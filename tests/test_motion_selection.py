"""Motion is an active planner term, beyond merely reporting a descriptor."""
from dataclasses import replace
import cv2
import numpy as np

from autoedit.config import Config
from autoedit.features import describe, motion_summary
from autoedit.models import Candidate
from autoedit.planner import beam_search


def test_measured_motion_changes_the_selected_path():
    texture = np.random.default_rng(7).integers(0, 256, (100, 100, 3), np.uint8)
    def shift(dx):
        return cv2.warpAffine(texture, np.float32([[1, 0, dx], [0, 1, 0]]), (100, 100), borderMode=cv2.BORDER_REFLECT)
    right = motion_summary([texture, shift(4)], [0, .2])
    left = motion_summary([texture, shift(-4)], [0, .2])
    descriptor = describe(texture)
    def candidate(clip, start_frame, motion):
        return Candidate(clip, 0, 1, 1, start_frame, 30, descriptor, descriptor, motion, motion,
                         {"entry_penalty": 0., "exit_penalty": 0., "interior_penalty": 0.}, 0, 29/30)
    first = candidate(0, 0, right)
    opposite = candidate(1, 30, left)
    compatible = candidate(2, 30, right)
    cfg = Config(reuse="forbid")
    pools = [[first], [opposite, compatible]]
    assert beam_search(pools, cfg)[0].path[-1] is compatible
    disabled = replace(cfg, weights={**cfg.weights, "motion": 0.})
    assert beam_search(pools, disabled)[0].path[-1] is opposite
