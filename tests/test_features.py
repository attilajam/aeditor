import cv2
import numpy as np

from autoedit.features import boundary_quality, describe, motion_distance, motion_summary, visual_distance


def test_describe_outputs_normalized_json_safe_features():
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:, :32] = (20, 90, 220)
    d = describe(frame)
    assert len(d["color"]) == 4 * 8 * 4 * 2
    assert np.isclose(sum(d["color"]), 1)
    assert len(d["structure"]) == len(d["edge"]) == 144
    assert all(0 <= v <= 1 for v in d["structure"] + d["edge"])
    assert 0 <= d["quality"]["black"] <= 1
    assert 0 <= d["quality"]["uniform"] <= 1
    assert 0 <= d["quality"]["texture"] <= 1


def test_spatial_structure_distinguishes_same_color_distribution():
    vertical = np.zeros((96, 96), dtype=np.uint8)
    horizontal = np.zeros((96, 96), dtype=np.uint8)
    # Each 48x48 quadrant has the same 50/50 color mass, arranged differently.
    for y in (0, 48):
        for x in (0, 48):
            vertical[y : y + 48, x : x + 24] = 255
            horizontal[y : y + 24, x : x + 48] = 255
    a = describe(cv2.cvtColor(vertical, cv2.COLOR_GRAY2BGR))
    b = describe(cv2.cvtColor(horizontal, cv2.COLOR_GRAY2BGR))
    distance = visual_distance(a, b)
    assert distance["color"] < 0.02
    assert distance["structure"] > 0.03


def test_motion_distance_handles_compatible_opposite_stationary_and_uncertain():
    right = {"vx": 0.2, "vy": 0.0, "magnitude": 0.2, "reliability": 1.0}
    slower_right = {"vx": 0.1, "vy": 0.0, "magnitude": 0.1, "reliability": 1.0}
    left = {"vx": -0.2, "vy": 0.0, "magnitude": 0.2, "reliability": 1.0}
    still = {"vx": 0.0, "vy": 0.0, "magnitude": 0.0, "reliability": 1.0}
    unknown = {"vx": 0.0, "vy": 0.0, "magnitude": 0.0, "reliability": 0.0}
    assert motion_distance(right, slower_right) < motion_distance(right, left)
    assert motion_distance(right, left) > 0.5
    assert motion_distance(still, still) < 0.01
    assert motion_distance(right, unknown) == 0.5


def test_motion_summary_stationary_textured_clip_is_reliable():
    rng = np.random.default_rng(4)
    frame = rng.integers(0, 256, (100, 120, 3), dtype=np.uint8)
    motion = motion_summary([frame, frame.copy(), frame.copy()], [0.0, 0.5, 1.0])
    assert motion["magnitude"] < 0.02
    assert motion["reliability"] > 0.5
    flat = np.zeros_like(frame)
    assert motion_summary([flat, flat], [0, 1])["reliability"] < 0.05
    split = np.zeros_like(frame)
    split[:, :60] = 255
    assert motion_summary([split, split], [0, 1])["reliability"] < 0.1


def test_extracted_motion_detects_translation_direction():
    rng = np.random.default_rng(9)
    texture = rng.integers(20, 236, (160, 160), dtype=np.uint8)

    def translated(dx):
        matrix = np.float32([[1, 0, dx], [0, 1, 0]])
        gray = cv2.warpAffine(texture, matrix, (160, 160), borderMode=cv2.BORDER_REFLECT)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    right = motion_summary([translated(0), translated(5)], [0.0, 0.5])
    right_again = motion_summary([translated(0), translated(5)], [0.0, 0.5])
    left = motion_summary([translated(0), translated(-5)], [0.0, 0.5])
    assert right["reliability"] > 0.5
    assert right["vx"] > 0.01
    assert left["vx"] < -0.01
    assert motion_distance(right, right_again) < motion_distance(right, left)


def test_black_and_uniform_boundary_frames_are_penalized():
    black = describe(np.zeros((64, 64, 3), dtype=np.uint8))
    flat = describe(np.full((64, 64, 3), 128, dtype=np.uint8))
    detailed = np.random.default_rng(8).integers(100, 256, (64, 64, 3), dtype=np.uint8)
    good = describe(detailed)
    assert boundary_quality(black) > 0.9
    assert boundary_quality(flat) > boundary_quality(good)
    # Consistently soft footage is judged relative to itself.
    assert boundary_quality(good, reference_sharpness=good["quality"]["sharpness"]) < 0.2
