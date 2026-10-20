"""Benchmark visual matching against a seeded feasible random baseline.

The fixture has three clips with the same composition and two mismatched
compositions. Four 1.5-second output slots must use four distinct sources, so
the baseline usually spends more slots on mismatched clips. Both methods use
the same beat-derived cut schedule, candidate pools, and no-reuse constraint.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from autoedit.audio import analyze_audio, choose_boundaries
from autoedit.candidates import generate_candidates, refine_pools
from autoedit.config import Config
from autoedit.planner import beam_search, random_feasible
from autoedit.pipeline import make_plan
from autoedit.probe import MediaError
from autoedit.render import render_plan
from autoedit.video import inspect_directory
from generate_test_media import generate_audio


SEED = 20261018


def _fixture_frame(index: int, total: int, kind: str, variant: int, seed: int) -> np.ndarray:
    """Make controlled source images with clear shared or mismatched layout."""
    rng = np.random.default_rng(seed + index)
    frame = np.zeros((240, 160, 3), dtype=np.uint8)
    if kind == "match":
        frame[:] = (42 + variant * 3, 88 + variant * 2, 142 + variant * 2)
        cv2.rectangle(frame, (10, 12), (149, 228), (226, 223, 214), 2)
        cv2.circle(frame, (80, 90), 31, (221, 169 + variant, 52), -1)
        cv2.rectangle(frame, (27, 158), (133, 181), (23, 28, 34), -1)
        x = int(23 + (index % 12) * 9)
        cv2.circle(frame, (x, 125), 8, (30, 218, 221), -1)
    elif kind == "portrait-a":
        frame[:] = (142, 35, 78)
        cv2.rectangle(frame, (12, 18), (148, 222), (245, 206, 62), -1)
        cv2.circle(frame, (80, 68), 25, (30, 25, 225), -1)
        cv2.line(frame, (24, 182), (138, 182), (20, 230, 50), 8)
    else:  # portrait-b: strong palette and layout mismatch
        frame[:] = (24, 134, 46)
        cv2.rectangle(frame, (8, 108), (151, 230), (190, 32, 164), -1)
        cv2.circle(frame, (45, 52), 26, (20, 220, 250), -1)
        cv2.line(frame, (78, 24), (142, 92), (245, 240, 225), 9)
    # A modest deterministic texture makes feature and motion estimates stable.
    noise = rng.integers(-5, 6, frame.shape, dtype=np.int16)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def make_fixture(output: Path, duration: float, fps: int) -> tuple[Path, Path, list[float]]:
    media = output / "fixture"
    video_dir = media / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    audio = media / "transients.wav"
    transient_times = generate_audio(audio, duration=duration, bpm=120, offset=0.25)
    specifications = [
        ("match-a.mp4", "match", 0), ("match-b.mp4", "match", 1),
        ("match-c.mp4", "match", 2), ("mismatch-a.mp4", "portrait-a", 0),
        ("mismatch-b.mp4", "portrait-b", 0),
    ]
    for source_index, (name, kind, variant) in enumerate(specifications):
        path = video_dir / name
        frame_count = round(duration * fps)
        raw = b"".join(_fixture_frame(i, frame_count, kind, variant, SEED + source_index).tobytes()
                        for i in range(frame_count))
        command = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pixel_format", "bgr24",
                   "-video_size", "160x240", "-framerate", str(fps), "-i", "pipe:0",
                   "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-pix_fmt", "yuv420p", str(path)]
        subprocess.run(command, input=raw, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return audio, video_dir, transient_times


def _boundary_metrics(video_path: Path, cuts: list[int], fps: int, contact_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    frames: dict[int, np.ndarray] = {}
    wanted = sorted({frame for cut in cuts[1:-1] for frame in (cut - 1, cut)})
    result = {}
    next_target = 0
    index = 0
    while next_target < len(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        if index == wanted[next_target]:
            frames[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            next_target += 1
        index += 1
    cap.release()
    if len(frames) != len(wanted):
        raise RuntimeError(f"could not decode boundary windows from {video_path}")
    panels = []
    values = []
    for cut in cuts[1:-1]:
        left, right = frames[cut - 1], frames[cut]
        # Independent full-resolution RGB MAE and Sobel-edge MAE; neither uses
        # planner descriptors or its reported transition scores.
        rgb_mae = float(np.abs(left.astype(np.float32) - right.astype(np.float32)).mean() / 255)
        left_luma = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        right_luma = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        def grad(image):
            gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8)
            gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8)
            return np.sqrt(gx * gx + gy * gy).clip(0, 1)
        edge_mae = float(np.abs(grad(left_luma) - grad(right_luma)).mean())
        metric = {"cut_frame": cut, "cut_seconds": cut / fps, "rgb_mae": rgb_mae, "edge_mae": edge_mae}
        result[str(cut)] = metric
        values.append(metric)
        panels.extend([left, right])
    thumb_w, thumb_h = 160, 240
    canvas = np.zeros((thumb_h * 2, thumb_w * (len(panels)//2), 3), dtype=np.uint8)
    for i, frame in enumerate(panels):
        thumb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), (thumb_w, thumb_h))
        y = (i % 2) * thumb_h
        x = (i // 2) * thumb_w
        canvas[y:y + thumb_h, x:x + thumb_w] = thumb
    cv2.imwrite(str(contact_path), canvas)
    result["mean"] = {key: float(np.mean([item[key] for item in values])) for key in ("rgb_mae", "edge_mae")}
    return result


def run_benchmark(output: Path, *, duration: float = 6.0, fps: int = 24,
                  width: int = 160, height: int = 240, seed: int = SEED) -> dict:
    if duration != 6.0:
        raise ValueError("the controlled benchmark currently requires duration=6 seconds for four 1.5s slots")
    if fps != 24 or (width, height) != (160, 240):
        raise ValueError("the default controlled benchmark uses 160x240 at 24 fps")
    output.mkdir(parents=True, exist_ok=True)
    audio_path, video_dir, transient_times = make_fixture(output, duration, fps)
    cfg = Config(duration=duration, pacing="medium", size=f"{width}x{height}", fps=fps,
                 min_shot=1.4, max_shot=1.6, samples_per_second=6.0,
                 analysis_long_side=120, candidates_per_slot=64, beam_width=48,
                 refine_top=8, refine_steps=3, reuse="forbid", seed=seed,
                 cache_dir=str(output / "cache"), preset="ultrafast", crf=23, threads=1)
    cfg.validate()
    audio_started = time.perf_counter()
    audio = analyze_audio(audio_path, cfg)
    boundaries, boundary_warnings = choose_boundaries(audio, cfg)
    clips, skipped = inspect_directory(video_dir, cfg, exclude=(audio_path,))
    pools, pool_warnings = generate_candidates(clips, boundaries, cfg)
    coarse = beam_search(pools, cfg, keep=max(1, cfg.refine_top - 1))
    # Draw the baseline from the exact same coarse candidate pools, then refine
    # its selected candidates alongside beam paths. This keeps the random path
    # available after refinement without restricting it to beam survivors.
    random_coarse = tuple(random_feasible(pools, cfg))
    refined = refine_pools(clips, pools, [state.path for state in coarse] + [random_coarse], cfg)
    selected = beam_search(refined, cfg, keep=1)[0].path
    random_path = tuple(
        min((candidate for candidate in refined[slot] if candidate.clip == original.clip),
            key=lambda candidate: abs(candidate.source_start - original.source_start) +
                                 abs(candidate.speed - original.speed))
        for slot, original in enumerate(random_coarse)
    )
    planner_seconds = time.perf_counter() - audio_started

    plans = {
        "enabled": make_plan(audio, clips, cfg, boundaries, selected, boundary_warnings + pool_warnings, skipped),
        "random": make_plan(audio, clips, cfg, boundaries, random_path, boundary_warnings + pool_warnings, skipped),
    }
    (output / "selected.plans").write_text(json.dumps(plans, indent=2) + "\n", encoding="utf-8")
    for name, plan in plans.items():
        (output / f"{name}.plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        render_plan(plan, output / f"{name}.mp4", overwrite=True)

    audio_cuts = [cut / fps for cut in boundaries[1:-1]]
    offsets = [min(abs(cut - event) for event in transient_times) for cut in audio_cuts]
    metrics = {
        name: _boundary_metrics(output / f"{name}.mp4", boundaries, fps, output / f"{name}.contact.png")
        for name in plans
    }
    baseline = metrics["random"]["mean"]["rgb_mae"]
    enabled = metrics["enabled"]["mean"]["rgb_mae"]
    improvement = 0.0 if baseline <= 1e-12 else 100.0 * (baseline - enabled) / baseline
    report = {
        "seed": seed,
        "fixture": {"duration": duration, "fps": fps, "size": [width, height],
                    "similar_sources": 3, "mismatched_sources": 2, "slots": len(boundaries) - 1,
                    "reuse": cfg.reuse, "known_transient_times": transient_times},
        "cuts": {"frames": boundaries, "seconds": audio_cuts,
                 "nearest_transient_offsets_seconds": offsets,
                 "mean_nearest_transient_offset_seconds": float(np.mean(offsets)) if offsets else 0.0,
                 "selected_beats": audio.get("beat_times", []), "audio_method": audio["reliability"]["method"]},
        "selected_sources": {name: [clips[c["clip"]]["identity"]["path"] for c in plans[name]["segments"]]
                             for name in plans},
        "selected_source_classes": {name: ["similar" if Path(clips[c["clip"]]["identity"]["path"]).name.startswith("match-") else "mismatched"
                                           for c in plans[name]["segments"]] for name in plans},
        "measured": metrics,
        "rgb_mae_improvement_percent": improvement,
        "planning_seconds_including_analysis": planner_seconds,
    }
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmark-output"),
                        help="directory for fixture, plans, rendered videos and JSON metrics")
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.output, duration=args.duration, fps=args.fps,
                                   width=args.width, height=args.height, seed=args.seed), indent=2))


if __name__ == "__main__":
    try:
        main()
    except (MediaError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
