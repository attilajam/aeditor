"""Create small, deterministic audio/video fixtures for editor tests.

Requires numpy, OpenCV (cv2), and the ffmpeg executable.  The media is synthetic
and intentionally includes edge cases useful for checking beat alignment and
source-frame selection.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SEED = 20261018
SAMPLE_RATE = 22_050
WIDTH, HEIGHT = 240, 160


def generate_audio(
    path: str | Path,
    duration: float = 6.0,
    bpm: float = 120.0,
    offset: float = 0.0,
    kind: str = "rhythmic",
) -> list[float]:
    """Write a WAV and return its transient times in seconds.

    ``kind`` is ``rhythmic``, ``weak`` (quiet, less regular pulses), or
    ``silence``. ``offset`` shifts the first pulse without shifting the file.
    """
    if duration <= 0 or bpm <= 0:
        raise ValueError("duration and bpm must be positive")
    if kind not in {"rhythmic", "weak", "silence"}:
        raise ValueError("kind must be rhythmic, weak, or silence")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = round(duration * SAMPLE_RATE)
    audio = np.zeros(count, dtype=np.float32)
    beat = 60.0 / bpm
    if kind == "rhythmic":
        times = [offset + n * beat for n in range(math.ceil(max(0, duration-offset) / beat)) if 0 <= offset+n*beat < duration]
        amplitude, decay, tone = 0.88, 0.045, 95.0
    elif kind == "weak":
        # Slightly swung, quiet impulses make onset confidence deliberately low.
        times = [offset + n * beat * (1.0 + (0.035 if n % 2 else -0.035)) for n in range(math.ceil(max(0, duration-offset) / beat)) if 0 <= offset+n*beat < duration]
        amplitude, decay, tone = 0.09, 0.022, 170.0
    else:
        times, amplitude, decay, tone = [], 0.0, 0.0, 0.0
    pulse_n = max(1, round(decay * SAMPLE_RATE))
    pulse_t = np.arange(pulse_n, dtype=np.float32) / SAMPLE_RATE
    pulse = (amplitude * np.sin(2 * np.pi * tone * pulse_t) * np.exp(-pulse_t * (8.0 / decay))
             if amplitude else np.zeros(pulse_n, dtype=np.float32))
    # A very soft bed keeps weak-rhythm files nonzero while remaining quiet.
    if kind == "weak":
        audio += (0.0015 * np.sin(2 * np.pi * 220 * np.arange(count) / SAMPLE_RATE)).astype(np.float32)
    for time_s in times:
        start = round(time_s * SAMPLE_RATE)
        end = min(count, start + pulse_n)
        if start < count:
            audio[start:end] += pulse[: end - start]
    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = np.round(pcm * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm16.tobytes())
    return times


def _frame(index: int, total: int, label: str, mode: str, direction: int, rng: np.random.Generator) -> np.ndarray:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    if mode == "black":
        return frame
    palette = {
        "similar-a": (36, 88, 148), "similar-b": (42, 91, 151),
        "different": (70, 32, 145), "motion": (35, 120, 70), "motion-left": (35, 120, 70),
        "rotated": (130, 75, 30), "vfr": (85, 60, 120),
        "short": (150, 105, 25),
    }
    color = palette.get(mode, (65, 80, 95))
    frame[:] = color
    # Stable layout across similar clips, with a subtly different accent.
    cv2.rectangle(frame, (9, 9), (WIDTH-10, HEIGHT-10), (235, 230, 215), 2)
    cv2.circle(frame, (WIDTH//2, HEIGHT//2), 27, (225, 170 if mode == "similar-a" else 176, 55), -1)
    cv2.rectangle(frame, (24, 105), (WIDTH-24, 125), (25, 30, 36), -1)
    if mode in {"motion", "motion-left"}:
        x = int(15 + (WIDTH - 45) * (index / max(1, total - 1)))
        if direction < 0:
            x = WIDTH - x
        cv2.circle(frame, (x, 75), 12, (10, 245, 245), -1)
    # Numeric IDs expose which source frame/time was selected.
    cv2.putText(frame, f"{label}  F{index:03d}", (15, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
    # Deterministic small markers help identify neighboring source regions.
    cv2.circle(frame, (20 + (index * 17) % 195, 35 + (index * 11) % 80), 3,
               tuple(int(v) for v in rng.integers(170, 256, size=3)), -1)
    return frame


def _write_video(path: Path, duration: float, fps: int, mode: str, *, audio: Path | None = None,
                 rotation: int = 0, vfr: bool = False, seed: int = SEED,
                 size: tuple[int, int] = (WIDTH, HEIGHT)) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = max(1, round(duration * fps))
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory(prefix="beat-fixture-") as temp_name:
        raw_path = Path(temp_name) / "frames.bgr"
        with raw_path.open("wb") as raw:
            for i in range(frame_count):
                image = _frame(i, frame_count, path.stem[:12], mode,
                               -1 if mode == "motion-left" else 1, rng)
                if size != (WIDTH, HEIGHT):
                    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
                raw.write(image.tobytes())
        # motion-left shares the moving scene renderer.
        vf = "settb=1/1000,setpts=PTS+4*floor(N/4)" if vfr else "null"
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pixel_format", "bgr24", "-video_size", f"{size[0]}x{size[1]}", "-framerate", str(fps),
               "-i", str(raw_path)]
        if audio is not None:
            cmd += ["-i", str(audio)]
        cmd += ["-vf", vf, "-fps_mode", "passthrough" if vfr else "cfr", "-c:v", "libx264", "-preset", "ultrafast",
                "-crf", "25", "-pix_fmt", "yuv420p"]
        if vfr:
            cmd += ["-enc_time_base:v", "1/1000", "-video_track_timescale", "90000"]
        if audio is not None:
            cmd += ["-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-shortest"]
        if rotation:
            cmd += ["-metadata:s:v:0", f"rotate={rotation}"]
        cmd += [str(path)]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if rotation:
            rotated = Path(temp_name) / "rotated.mp4"
            remux = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                     "-display_rotation:v:0", str(rotation), "-i", str(path),
                     "-map", "0", "-c", "copy", str(rotated)]
            subprocess.run(remux, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            shutil.move(str(rotated), path)
    return {"path": str(path), "duration": duration, "width": size[0], "height": size[1],
            "fps": fps, "has_audio": audio is not None, "rotation": rotation,
            "vfr": vfr, "frames": frame_count, "composition": mode}


def generate_media(root: str | Path, duration: float = 6.0) -> dict[str, Any]:
    """Generate the complete fixture set under ``root`` and return a manifest."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    audio_dir = root / "audio fixtures"
    video_dir = root / "video fixtures"
    rhythmic = audio_dir / "rhythmic 120 bpm.wav"
    weak = audio_dir / "weak rhythm.wav"
    silence = audio_dir / "silence.wav"
    rhythmic_times = generate_audio(rhythmic, duration, bpm=120, offset=0.25)
    weak_times = generate_audio(weak, duration, bpm=93, offset=0.41, kind="weak")
    generate_audio(silence, duration, kind="silence")
    videos = []
    specs = [
        ("similar scene A.mp4", duration, 24, "similar-a", rhythmic, 0, False),
        ("similar scene B.mp4", duration, 30, "similar-b", None, 0, False),
        ("different composition.mp4", duration, 24, "different", None, 0, False),
        ("motion left.mp4", duration, 24, "motion-left", None, 0, False),
        ("motion right.mp4", duration, 30, "motion", None, 0, False),
        ("black video.mp4", duration, 24, "black", None, 0, False),
        ("rotated metadata.mp4", duration, 24, "rotated", None, 90, False),
        ("variable frame rate.mp4", duration, 24, "vfr", None, 0, True),
        ("very short clip.mp4", min(0.7, duration), 30, "short", None, 0, False),
    ]
    for i, (name, seconds, fps, mode, audio, rotation, vfr) in enumerate(specs):
        videos.append(_write_video(video_dir / name, seconds, fps, mode, audio=audio,
                                   rotation=rotation, vfr=vfr, seed=SEED + i,
                                   size=(160, 240) if name == "different composition.mp4" else (WIDTH, HEIGHT)))
    corrupt = video_dir / "corrupt.mp4"
    corrupt.write_bytes(b"This is intentionally not a video file.\n")
    videos.append({"path": str(corrupt), "duration": None, "width": None, "height": None,
                   "fps": None, "has_audio": False, "rotation": 0, "vfr": False,
                   "frames": 0, "composition": "corrupt"})
    return {
        "seed": SEED,
        "duration": duration,
        "audio": {"rhythmic": str(rhythmic), "weak": str(weak), "silence": str(silence),
                  "rhythmic_transient_times": rhythmic_times, "weak_transient_times": weak_times},
        "videos": videos,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="directory for generated fixtures")
    parser.add_argument("--duration", type=float, default=6.0, help="base audio/video duration in seconds")
    args = parser.parse_args()
    manifest = generate_media(args.output, args.duration)
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
