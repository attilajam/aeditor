"""Behavioral coverage for model-free, spatially aware visual matching."""

import base64
import copy
import json

import cv2
import numpy as np

from autoedit.config import Config
from autoedit.features import boundary_quality, decode_edge_level, describe, visual_distance
from autoedit.planner import beam_search, transition_components
from autoedit.models import Candidate


def _subject_frame(*, background=(25, 45, 70), subject=(225, 205, 175), pose="arms_out", center=(120, 120), scale=1.0):
    """Render a simple human silhouette with antialiased, model-free drawing."""
    mask = np.zeros((160, 120), dtype=np.uint8)
    cv2.circle(mask, (60, 22), 13, 255, -1, cv2.LINE_AA)
    cv2.fillPoly(mask, [np.array([(43, 43), (77, 43), (83, 91), (37, 91)], np.int32)], 255, cv2.LINE_AA)
    if pose == "arms_out":
        arms = [((43, 52), (15, 77)), ((77, 52), (105, 77))]
    elif pose == "one_arm_up":
        arms = [((43, 52), (15, 77)), ((77, 52), (103, 20))]
    else:
        arms = [((43, 52), (52, 90)), ((77, 52), (68, 90))]
    for a, b in arms:
        cv2.line(mask, a, b, 255, 11, cv2.LINE_AA)
    cv2.line(mask, (47, 88), (37, 145), 255, 13, cv2.LINE_AA)
    cv2.line(mask, (73, 88), (83, 145), 255, 13, cv2.LINE_AA)

    height, width = max(12, round(160 * scale)), max(12, round(120 * scale))
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)
    frame = np.empty((240, 240, 3), dtype=np.uint8)
    frame[:] = background
    x0, y0 = round(center[0] - width / 2), round(center[1] - height / 2)
    x1, y1 = max(0, x0), max(0, y0)
    x2, y2 = min(240, x0 + width), min(240, y0 + height)
    if x1 < x2 and y1 < y2:
        crop = mask[y1 - y0 : y2 - y0, x1 - x0 : x2 - x0]
        frame[y1:y2, x1:x2][crop > 0] = subject
    return frame


def _candidate(clip, descriptor):
    motion = {"vx": 0.0, "vy": 0.0, "magnitude": 0.0, "reliability": 1.0}
    return Candidate(
        clip=clip, source_start=0.0, source_end=1.0, speed=1.0,
        output_start_frame=0, output_frame_count=30,
        entry=descriptor, exit=descriptor, entry_motion=motion, exit_motion=motion,
        quality={}, entry_pts=0.0, exit_pts=1.0,
    )


def test_subject_identity_and_pose_outweigh_palette_and_background_in_planner():
    reference = describe(_subject_frame())
    recolored = describe(_subject_frame(background=(200, 150, 35), subject=(25, 235, 210)))
    wrong_pose = describe(_subject_frame(pose="one_arm_up"))

    assert visual_distance(reference, recolored)["edge"] < visual_distance(reference, wrong_pose)["edge"]

    cfg = Config(reuse="allow")
    start, same_subject, same_background = map(_candidate, (0, 1, 2), (reference, recolored, wrong_pose))
    same_transition = transition_components(start, same_subject, cfg)
    pose_transition = transition_components(start, same_background, cfg)
    assert same_transition["weighted"] < pose_transition["weighted"]
    chosen = beam_search([[start], [same_subject, same_background]], cfg, keep=1)[0]
    assert chosen.path[-1] is same_subject


def test_spatial_edge_distance_penalizes_subject_translation_and_scale_change():
    reference = describe(_subject_frame())
    shifted = describe(_subject_frame(center=(174, 120)))
    scaled = describe(_subject_frame(scale=1.55))
    same = visual_distance(reference, reference)["edge"]
    assert same < 0.005  # the JSON descriptor rounds pooled bins
    assert visual_distance(reference, shifted)["edge"] > same + 0.01
    assert visual_distance(reference, scaled)["edge"] > same + 0.01


def test_oriented_edge_descriptor_is_json_safe_normalized_and_polarity_invariant():
    # Equal luminance contrast in opposite directions should preserve unsigned edges.
    dark_bg = describe(_subject_frame(background=(35, 35, 35), subject=(220, 220, 220)))
    light_bg = describe(_subject_frame(background=(220, 220, 220), subject=(35, 35, 35)))
    encoded = json.dumps(dark_bg, allow_nan=False)
    assert json.loads(encoded)["oriented_edges"] == dark_bg["oriented_edges"]
    assert "oriented_edges" in dark_bg
    levels = dark_bg["oriented_edges"]
    assert [level["grid"] for level in levels] == [8, 16]
    assert all(level["bins"] == 8 for level in levels)
    for level in levels:
        raw = base64.b64decode(level["data"], validate=True)
        assert len(raw) == level["grid"] * level["grid"] * level["bins"] * 2
        values = decode_edge_level(level)
        assert values.shape == (level["grid"], level["grid"], level["bins"])
        assert np.all(np.isfinite(values))
        assert np.all((values >= 0) & (values <= 1))
        assert np.isclose(np.linalg.norm(values), 1.0, atol=0.002)
    assert visual_distance(dark_bg, light_bg)["edge"] < 0.03


def test_packed_edges_match_legacy_value_lists_and_reject_malformed_data():
    descriptor = describe(_subject_frame())
    packed = descriptor["oriented_edges"]
    legacy = copy.deepcopy(packed)
    for level in legacy:
        level["values"] = decode_edge_level(level).ravel().tolist()
        del level["encoding"]
        del level["data"]
    packed_distance = visual_distance(descriptor, descriptor)["edge"]
    old_format = {**descriptor, "oriented_edges": legacy}
    assert np.isclose(visual_distance(old_format, old_format)["edge"], packed_distance, atol=1e-6)

    malformed = copy.deepcopy(descriptor)
    malformed["oriented_edges"][0]["data"] = base64.b64encode(b"bad").decode("ascii")
    with np.testing.assert_raises(ValueError):
        visual_distance(malformed, malformed)


def test_black_and_uniform_frames_are_penalized_and_old_descriptors_still_compare():
    black = describe(np.zeros((240, 240, 3), dtype=np.uint8))
    uniform = describe(np.full((240, 240, 3), 128, dtype=np.uint8))
    useful = describe(_subject_frame())
    assert boundary_quality(black) > 0.9
    assert boundary_quality(uniform) > boundary_quality(useful)

    # Cached descriptors made before oriented_edges existed remain comparable.
    old_a = {key: useful[key] for key in ("color", "structure", "edge")}
    old_b = {key: black[key] for key in ("color", "structure", "edge")}
    distance = visual_distance(old_a, old_b)
    assert set(distance) >= {"color", "structure", "edge"}
    assert all(np.isfinite(value) and 0 <= value <= 1 for value in distance.values())
