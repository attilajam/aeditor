#!/usr/bin/env python3
"""Reproducible controlled benchmark for screen-space subject matching.

Creates twelve small synthetic source videos, runs the normal analysis, candidate
generation, beam search, native-frame refinement, plan, render, and validation
paths, then measures decoded-output joint displacement without consulting any
planner transition scores.  Run from the repository root with:

    uv run python scripts/benchmark_spatial_matching.py \
      --output-dir examples/spatial-matching --overwrite
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoedit.audio import analyze_audio, choose_boundaries  # noqa: E402
from autoedit.cache import write_json  # noqa: E402
from autoedit.candidates import generate_candidates, refine_pools  # noqa: E402
from autoedit.config import Config  # noqa: E402
from autoedit.features import legacy_visual_distance  # noqa: E402
from autoedit.pipeline import analyze_inputs, make_plan, versions  # noqa: E402
from autoedit.planner import beam_search, random_feasible  # noqa: E402
from autoedit.probe import run  # noqa: E402
from autoedit.render import render_plan  # noqa: E402
from autoedit.report import write_report  # noqa: E402
from autoedit.video import analysis_dimensions  # noqa: E402


WIDTH, HEIGHT, FPS = 320, 180, 24
DURATION, SLOT_SECONDS, SLOT_COUNT = 6.0, 1.5, 4
SLOT_FRAMES = int(SLOT_SECONDS * FPS)

# Each dot color is assigned to one named landmark.  These colors remain
# constant across clips and are distinct from every synthetic background.
LANDMARK_COLORS = [
    (15, 35, 245), (15, 220, 250), (230, 30, 220), (15, 220, 20),
    (245, 100, 20), (180, 30, 245), (20, 200, 150), (245, 200, 25),
    (80, 245, 230), (210, 120, 245), (80, 180, 20), (245, 40, 105), (110, 30, 230),
]
LANDMARK_NAMES = [
    "head", "neck", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
]


def _call(command: list[str]) -> None:
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _point(x: float, y: float, cx: float, cy: float, scale: float) -> tuple[int, int]:
    return round(cx + x * scale), round(cy + y * scale)


def _skeleton(layout: dict) -> dict[str, tuple[int, int]]:
    cx, cy, scale = layout["center_x"], layout["center_y"], layout["scale"]
    pose = layout["pose"]
    # Coordinates are expressed relative to the hip.  Four target layouts use
    # small gesture variations; distractors combine larger pose and placement
    # changes while retaining the same landmarks and silhouette style.
    poses = [
        {"head": (0, -43), "neck": (0, -30), "left_shoulder": (-11, -29), "right_shoulder": (11, -29),
         "left_elbow": (-20, -13), "right_elbow": (20, -13), "left_wrist": (-25, 2), "right_wrist": (25, 2),
         "hip": (0, 0), "left_knee": (-9, 20), "right_knee": (9, 20), "left_ankle": (-11, 41), "right_ankle": (11, 41)},
        {"head": (0, -43), "neck": (0, -30), "left_shoulder": (-11, -29), "right_shoulder": (11, -29),
         "left_elbow": (-23, -31), "right_elbow": (21, -12), "left_wrist": (-27, -47), "right_wrist": (24, 3),
         "hip": (0, 0), "left_knee": (-8, 19), "right_knee": (10, 21), "left_ankle": (-12, 40), "right_ankle": (12, 42)},
        {"head": (0, -43), "neck": (0, -30), "left_shoulder": (-11, -29), "right_shoulder": (11, -29),
         "left_elbow": (-19, -13), "right_elbow": (22, -30), "left_wrist": (-24, 2), "right_wrist": (28, -47),
         "hip": (0, 0), "left_knee": (-10, 21), "right_knee": (8, 19), "left_ankle": (-13, 42), "right_ankle": (10, 40)},
        {"head": (0, -43), "neck": (0, -30), "left_shoulder": (-11, -29), "right_shoulder": (11, -29),
         "left_elbow": (-23, -30), "right_elbow": (23, -30), "left_wrist": (-29, -45), "right_wrist": (29, -45),
         "hip": (0, 0), "left_knee": (-8, 20), "right_knee": (8, 20), "left_ankle": (-10, 41), "right_ankle": (10, 41)},
        {"head": (0, -43), "neck": (0, -30), "left_shoulder": (-11, -29), "right_shoulder": (11, -29),
         "left_elbow": (-28, -5), "right_elbow": (15, -8), "left_wrist": (-42, 6), "right_wrist": (30, 8),
         "hip": (0, 0), "left_knee": (-22, 12), "right_knee": (18, 26), "left_ankle": (-40, 18), "right_ankle": (22, 45)},
        {"head": (0, -43), "neck": (0, -30), "left_shoulder": (-11, -29), "right_shoulder": (11, -29),
         "left_elbow": (-18, -39), "right_elbow": (27, -12), "left_wrist": (-12, -57), "right_wrist": (44, -2),
         "hip": (0, 0), "left_knee": (-16, 27), "right_knee": (14, 11), "left_ankle": (-24, 48), "right_ankle": (28, 29)},
        {"head": (0, -43), "neck": (0, -30), "left_shoulder": (-11, -29), "right_shoulder": (11, -29),
         "left_elbow": (-29, -20), "right_elbow": (19, -38), "left_wrist": (-43, -31), "right_wrist": (12, -57),
         "hip": (0, 0), "left_knee": (-20, 11), "right_knee": (19, 24), "left_ankle": (-38, 27), "right_ankle": (31, 43)},
        {"head": (0, -43), "neck": (0, -30), "left_shoulder": (-11, -29), "right_shoulder": (11, -29),
         "left_elbow": (-16, -6), "right_elbow": (30, -3), "left_wrist": (-7, 10), "right_wrist": (46, 12),
         "hip": (0, 0), "left_knee": (-4, 28), "right_knee": (20, 9), "left_ankle": (-8, 49), "right_ankle": (40, 20)},
    ][pose]
    if layout["kind"] == "target":
        # The four true matches share a compact natural pose family while still
        # varying wrist/knee placement slightly across the cut sequence.
        tweaks = [
            {},
            {"left_elbow": (-1, -1), "right_wrist": (1, 1), "left_knee": (0, 1)},
            {"right_elbow": (1, -1), "left_wrist": (-1, 1), "right_ankle": (0, -1)},
            {"left_wrist": (-2, 0), "right_wrist": (2, 0), "left_ankle": (-1, 0)},
        ][layout["pose_variant"]]
        poses = dict(poses)
        for name, (dx, dy) in tweaks.items():
            x, y = poses[name]
            poses[name] = (x + dx, y + dy)
    return {name: _point(x, y, cx, cy, scale) for name, (x, y) in poses.items()}


def _draw(layout: dict) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    frame = np.full((HEIGHT, WIDTH, 3), layout["background"], dtype=np.uint8)
    # A faint room-like backdrop keeps this from being only a flat-color card
    # fixture while leaving the subject contour as the strongest shared signal.
    backdrop = tuple(int(v) for v in layout["backdrop_color"])
    cv2.line(frame, (18, 26), (302, 26), backdrop, 1, cv2.LINE_AA)
    cv2.line(frame, (24, 27), (24, 158), backdrop, 1, cv2.LINE_AA)
    cv2.line(frame, (296, 27), (296, 158), backdrop, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (37, 39), (74, 71), backdrop, 1, cv2.LINE_AA)
    joints = _skeleton(layout)
    # Underlying person contour: bright, consistent thickness makes the edge
    # descriptors depend on shape and screen location instead of palette.
    links = [("head", "neck"), ("neck", "left_shoulder"), ("neck", "right_shoulder"),
             ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
             ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
             ("neck", "hip"), ("hip", "left_knee"), ("left_knee", "left_ankle"),
             ("hip", "right_knee"), ("right_knee", "right_ankle")]
    for a, b in links:
        cv2.line(frame, joints[a], joints[b], (242, 242, 242), 5, cv2.LINE_AA)
    for index, name in enumerate(LANDMARK_NAMES):
        cv2.circle(frame, joints[name], 5, LANDMARK_COLORS[index], -1, cv2.LINE_AA)
    return frame, joints


def _layouts() -> list[dict]:
    # Low-contrast structure is purposefully changed for every target clip,
    # while all eight distractors share one palette.  Skeleton landmarks stay
    # the same colors in every source, so decoded localization is palette-free.
    backgrounds = [35, 112, 178, 228]
    targets = []
    for i, bg in enumerate(backgrounds):
        targets.append({"kind": "target", "slot": i, "pose": 0, "pose_variant": i,
                        "center_x": 158 + (i - 1.5) * 3.0,
                        "center_y": 92 + [-1, 1, 2, -2][i],
                        "scale": [1.0, .99, 1.02, .98][i], "background": bg,
                        "backdrop_color": (min(255, bg + 7),) * 3})
    decoys = []
    # Eight varied layouts far from the compact target trajectory.  Varying
    # scale, pose, and placement prevents the new descriptor from winning via
    # one fixed-layout shortcut.
    specs = [
        (4, 55, 62, .76), (5, 260, 61, .78), (6, 70, 126, .80), (7, 250, 126, .77),
        (4, 110, 54, 1.08), (5, 210, 54, 1.05), (6, 108, 128, 1.02), (7, 214, 128, 1.07),
    ]
    for i, (pose, cx, cy, scale) in enumerate(specs):
        decoys.append({"kind": "distractor", "slot": None, "pose": pose, "pose_variant": 0,
                       "center_x": cx, "center_y": cy, "scale": scale,
                       "background": (72, 76, 82), "backdrop_color": (82, 86, 92), "distractor_id": i})
    return targets + decoys


def _generate_inputs(root: Path) -> tuple[Path, Path, list[dict]]:
    clip_dir = root / "sources"
    clip_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    for index, layout in enumerate(_layouts()):
        name = f"clip-{index:02d}-{layout['kind']}.mp4"
        image_path = clip_dir / f".frame-{index:02d}.png"
        video_path = clip_dir / name
        frame, joints = _draw(layout)
        if not cv2.imwrite(str(image_path), frame):
            raise RuntimeError(f"could not write synthetic frame {image_path}")
        _call(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", str(image_path), "-t", str(SLOT_SECONDS),
               "-r", str(FPS), "-an", "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast",
               "-pix_fmt", "yuv420p", "-threads", "1", str(video_path)])
        image_path.unlink()
        metadata.append({"clip_index": index, "filename": name, **layout,
                         "joints": {k: list(v) for k, v in joints.items()},
                         "landmark_colors_bgr": [list(c) for c in LANDMARK_COLORS]})
    audio = root / "benchmark.wav"
    _call(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
           "sine=frequency=220:sample_rate=48000:duration=6", "-ac", "2", "-c:a", "pcm_s16le", str(audio)])
    write_json(root / "source-ground-truth.json", {"width": WIDTH, "height": HEIGHT, "fps": FPS,
                                                       "duration": DURATION, "landmark_names": LANDMARK_NAMES,
                                                       "clips": metadata})
    return clip_dir, audio, metadata


def _cfg(output: Path) -> Config:
    return Config(duration=DURATION, fps=FPS, fit="stretch", size=f"{WIDTH}x{HEIGHT}",
                  speed_min=1.0, speed_max=1.0, reuse="forbid", seed=41,
                  min_shot=SLOT_SECONDS, max_shot=SLOT_SECONDS, quiet_threshold=0.0,
                  samples_per_second=4.0, analysis_long_side=192,
                  candidates_per_slot=64, beam_width=48, refine_top=6, refine_steps=3,
                  cache_dir=str(output / ".cache"), crf=15, preset="ultrafast", threads=2)


def _choose_path(pools, cfg, method: str):
    import autoedit.planner as planner_module
    original = planner_module.visual_distance
    try:
        if method == "legacy_palette":
            def legacy(a, b, center_bias=.65):
                return legacy_visual_distance(a, b)
            planner_module.visual_distance = legacy
            legacy_cfg = replace(cfg, weights={**cfg.weights, "color": .25, "structure": .45,
                                               "edge": .15, "motion": .15})
        else:
            legacy_cfg = cfg

        if method == "random":
            path = random_feasible(pools, legacy_cfg)
            seeds = [tuple(path)]
        else:
            states = beam_search(pools, legacy_cfg)
            seeds = [state.path for state in states]
        refined = refine_pools_for_benchmark(pools, seeds, legacy_cfg)
        if method == "random":
            path = []
            for old, pool in zip(seeds[0], refined):
                choices = [candidate for candidate in pool if candidate.clip == old.clip]
                if not choices:
                    raise RuntimeError(f"refinement dropped random candidate from clip {old.clip}")
                path.append(min(choices, key=lambda c: abs(c.source_start-old.source_start)))
            return path, legacy_cfg
        result = beam_search(refined, legacy_cfg)[0].path
        return result, legacy_cfg
    finally:
        planner_module.visual_distance = original


def refine_pools_for_benchmark(pools, paths, cfg):
    """Refine selected paths using pipeline refinement with the known clip list."""
    # Stored by main while all method runs share exactly one clip analysis.
    return refine_pools(_ACTIVE_CLIPS, pools, paths, cfg)


_ACTIVE_CLIPS = None


def _decode_output_frames(path: Path) -> np.ndarray:
    # Decode every output frame through ffmpeg, independent of source metadata.
    raw = run(["ffmpeg", "-v", "error", "-threads", "1", "-i", str(path), "-map", "0:v:0", "-an",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], binary=True)
    expected = SLOT_COUNT * SLOT_FRAMES * WIDTH * HEIGHT * 3
    if len(raw) != expected:
        raise RuntimeError(f"decoded {len(raw)} bytes from {path}; expected {expected}")
    return np.frombuffer(raw, np.uint8).reshape(SLOT_COUNT, SLOT_FRAMES, HEIGHT, WIDTH, 3).copy()


def _localize(frame: np.ndarray, color: tuple[int, int, int],
              expected: tuple[int, int]) -> tuple[float, float] | None:
    # The marker masks look only for their unique chroma, never grayscale,
    # background, planner descriptors, or selected clip metadata coordinates.
    color_f = np.asarray(color, dtype=np.int16)
    delta = np.abs(frame.astype(np.int16) - color_f)
    mask = np.all(delta <= 70, axis=2)
    ex, ey = expected
    y0, y1 = max(0, ey-9), min(HEIGHT, ey+10)
    x0, x1 = max(0, ex-9), min(WIDTH, ex+10)
    local = mask[y0:y1, x0:x1]
    ys, xs = np.nonzero(local)
    if len(xs) < 3:
        return None
    # Restrict each unique chroma to its documented landmark neighborhood so
    # gray scene backgrounds and compression ringing cannot masquerade as a dot.
    return float(np.mean(xs) + x0), float(np.mean(ys) + y0)


def _measure(plan: dict, frames: np.ndarray, source_metadata: list[dict]) -> dict:
    segment_rows, localization_failures, boundary_distances = [], [], []
    for slot, segment in enumerate(plan["segments"]):
        clip_index = int(segment["clip"])
        meta = source_metadata[clip_index]
        representative = frames[slot, SLOT_FRAMES//2]
        centers = {}
        for name, color in zip(LANDMARK_NAMES, LANDMARK_COLORS):
            expected = tuple(meta["joints"][name])
            center = _localize(representative, color, expected)
            if center is None:
                localization_failures.append({"slot": slot, "landmark": name})
            else:
                centers[name] = center
        segment_rows.append({"slot": slot, "clip": clip_index, "kind": meta["kind"],
                             "source_kind_slot": meta["slot"], "distractor_id": meta.get("distractor_id"),
                             "landmarks": centers})
    decoded_boundary_ground_truth_errors = []
    for slot, (left, right) in enumerate(zip(segment_rows, segment_rows[1:])):
        left_meta, right_meta = source_metadata[left["clip"]], source_metadata[right["clip"]]
        outgoing_frame, incoming_frame = frames[slot, -1], frames[slot+1, 0]
        outgoing, incoming = {}, {}
        for name, color in zip(LANDMARK_NAMES, LANDMARK_COLORS):
            expected_left = tuple(left_meta["joints"][name])
            expected_right = tuple(right_meta["joints"][name])
            point_left = _localize(outgoing_frame, color, expected_left)
            point_right = _localize(incoming_frame, color, expected_right)
            if point_left is not None:
                outgoing[name] = point_left
            if point_right is not None:
                incoming[name] = point_right
        names = sorted(set(outgoing) & set(incoming))
        distances = [math.dist(outgoing[name], incoming[name]) for name in names]
        gt_errors = [math.dist(outgoing[name], tuple(left_meta["joints"][name])) for name in names]
        gt_errors += [math.dist(incoming[name], tuple(right_meta["joints"][name])) for name in names]
        decoded_boundary_ground_truth_errors.append({"cut_frame": (slot+1)*SLOT_FRAMES,
                                                       "localized_landmarks_per_side": len(names),
                                                       "max_source_coordinate_error_pixels": max(gt_errors) if gt_errors else None})
        boundary_distances.append({"from_slot": left["slot"], "to_slot": right["slot"],
                                   "landmark_count": len(distances),
                                   "rms_pixels": float(np.sqrt(np.mean(np.square(distances)))) if distances else None,
                                   "rms_diagonal_fraction": float(np.sqrt(np.mean(np.square(distances))) /
                                                                    math.hypot(WIDTH, HEIGHT)) if distances else None})
    usable = [row["rms_diagonal_fraction"] for row in boundary_distances if row["rms_diagonal_fraction"] is not None]
    return {"selection": [{k: row[k] for k in ("slot", "clip", "kind", "source_kind_slot", "distractor_id")}
                          for row in segment_rows],
            "decoded_joint_transition_error": {
                "description": "Mean boundary RMS landmark displacement between the last decoded frame before each cut and the first decoded frame after it, normalized by frame diagonal; lower is more spatially consistent.",
                "mean_diagonal_fraction": float(np.mean(usable)) if usable else None,
                "mean_pixels": float(np.mean([x["rms_pixels"] for x in boundary_distances if x["rms_pixels"] is not None])) if usable else None,
                "boundaries": boundary_distances,
                "boundary_frame_ground_truth_localization": decoded_boundary_ground_truth_errors},
            "decoded_marker_localization_failures": localization_failures}


def _contact_sheet(frames: np.ndarray, plan: dict, path: Path, metadata: list[dict]) -> None:
    tiles = []
    for slot in range(SLOT_COUNT):
        frame = frames[slot, SLOT_FRAMES//2].copy()
        meta = metadata[plan["segments"][slot]["clip"]]
        label = f"slot {slot+1}: clip {meta['clip_index']} ({meta['kind']})"
        cv2.rectangle(frame, (0, 0), (WIDTH, 19), (20, 20, 20), -1)
        cv2.putText(frame, label, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(frame)
    sheet = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError(f"could not write contact sheet {path}")


def _run_method(method: str, pools, cfg, audio: dict, clips, boundaries, metadata, outdir: Path) -> dict:
    path, method_cfg = _choose_path(pools, cfg, method)
    import autoedit.planner as planner_module
    original = planner_module.visual_distance
    try:
        if method == "legacy_palette":
            planner_module.visual_distance = lambda a, b, center_bias=.65: legacy_visual_distance(a, b)
        plan = make_plan(audio, clips, method_cfg, boundaries, path)
    finally:
        planner_module.visual_distance = original
    if method != "spatial":
        plan["matching"] = {"method": "legacy_palette" if method == "legacy_palette" else "deterministic_random",
                            "note": "Benchmark comparison path; candidate pool and output schedule are shared."}
    method_dir = outdir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    plan_path = method_dir / "plan.json"
    video_path = method_dir / "edit.mp4"
    write_json(plan_path, plan)
    validation = render_plan(plan, video_path, overwrite=True)
    write_json(method_dir / "validation.json", validation)
    write_report(plan, video_path, validation, method_dir / "report.html")
    frames = _decode_output_frames(video_path)
    _contact_sheet(frames, plan, method_dir / "contact-sheet.png", metadata)
    metrics = _measure(plan, frames, metadata)
    metrics.update({"plan": str(plan_path.relative_to(outdir)), "video": str(video_path.relative_to(outdir)),
                    "contact_sheet": str((method_dir / "contact-sheet.png").relative_to(outdir)),
                    "validation_valid": bool(validation["valid"]),
                    "decoded_output_frames": int(validation["decoded_frames"]),
                    "cut_frames": boundaries})
    write_json(method_dir / "metrics.json", metrics)
    return metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", "--output", dest="output_dir", type=Path,
                        default=ROOT / "examples/spatial-matching")
    parser.add_argument("--overwrite", action="store_true", help="replace the selected artifact directory")
    args = parser.parse_args(argv)
    outdir = args.output_dir.expanduser().resolve()
    if outdir.exists():
        if not args.overwrite:
            parser.error(f"{outdir} already exists; pass --overwrite")
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)

    clip_dir, audio_path, metadata = _generate_inputs(outdir)
    cfg = _cfg(outdir)
    audio, clips, skipped = analyze_inputs(audio_path, clip_dir, cfg)
    if skipped:
        raise RuntimeError(f"synthetic sources were unexpectedly skipped: {skipped}")
    # Audio is analyzed normally, then held to one regular four-shot schedule
    # so all three methods receive exactly the same 0/1.5/3/4.5/6s cuts.
    audio["fallback"] = "regular"
    audio["beat_times"] = []
    audio["onset_times"] = []
    audio["warnings"] = list(audio.get("warnings", [])) + ["Controlled benchmark uses a fixed regular 1.5 second shot schedule."]
    boundaries, boundary_warnings = choose_boundaries(audio, cfg, max_duration=SLOT_SECONDS)
    expected = [0, 36, 72, 108, 144]
    if boundaries != expected:
        raise RuntimeError(f"controlled benchmark schedule differs: {boundaries}; expected {expected}")
    pools, pool_warnings = generate_candidates(clips, boundaries, cfg)
    if any(len(pool) != len(clips) for pool in pools):
        raise RuntimeError(f"expected every one of {len(clips)} sources in all pools; got {[len(p) for p in pools]}")
    global _ACTIVE_CLIPS
    _ACTIVE_CLIPS = clips

    methods = {}
    for name in ("spatial", "legacy_palette", "random"):
        print(f"Running {name}…", flush=True)
        methods[name] = _run_method(name, pools, cfg, audio, clips, boundaries, metadata, outdir)

    result = {"benchmark": "spatial subject pose and screen position versus palette matching",
              "seed": cfg.seed, "versions": versions(), "configuration": cfg.to_dict(),
              "cut_schedule_frames": boundaries, "cut_schedule_seconds": [f/FPS for f in boundaries],
              "source_clips": len(clips), "candidate_pool_sizes": [len(p) for p in pools],
              "candidate_pool_source_sets_identical": True, "source_reuse": "forbid",
              "warnings": boundary_warnings + pool_warnings,
              "synthetic_setup": {"target_sources": 4, "palette_decoys": 8,
                                  "target_backgrounds_change_per_clip": True,
                                  "decoy_background_is_shared": True,
                                  "landmarks": LANDMARK_NAMES,
                                  "independent_metric": "decoded output landmark displacement; no planner transition score is used"},
              "methods": methods}
    write_json(outdir / "results.json", result)
    spatial_score = methods["spatial"]["decoded_joint_transition_error"]["mean_diagonal_fraction"]
    legacy_score = methods["legacy_palette"]["decoded_joint_transition_error"]["mean_diagonal_fraction"]
    random_score = methods["random"]["decoded_joint_transition_error"]["mean_diagonal_fraction"]
    print(json.dumps({"results": str(outdir / "results.json"),
                      "spatial_joint_error": spatial_score, "legacy_joint_error": legacy_score,
                      "random_joint_error": random_score,
                      "spatial_beats_legacy": spatial_score < legacy_score,
                      "spatial_beats_random": spatial_score < random_score}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
