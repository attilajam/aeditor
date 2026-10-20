"""Command-line interface for scene-split."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scene-split",
        description="Detect scene cuts in a video and save each scene as an MP4 clip.",
    )
    parser.add_argument("video", type=Path, help="video file to split")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="directory for clips (default: <video-name>_clips next to the video)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=27.0,
        help="cut sensitivity; lower values detect more cuts (default: 27)",
    )
    parser.add_argument(
        "--min-scene-len",
        default="0.6s",
        help="minimum scene length in frames, seconds, or timecode (default: 0.6s)",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=20,
        help="H.264 quality, from 0 (best) to 51 (smallest; default: 20)",
    )
    parser.add_argument(
        "--preset",
        default="veryfast",
        choices=(
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        ),
        help="encoding speed/size tradeoff (default: veryfast)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    video = args.video.expanduser().resolve()
    if not video.is_file():
        parser.error(f"video file does not exist: {video}")
    if args.crf < 0 or args.crf > 51:
        parser.error("--crf must be between 0 and 51")
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg is required; install it and make sure it is on PATH")

    output = args.output.expanduser().resolve() if args.output else video.parent / f"{video.stem}_clips"
    output.mkdir(parents=True, exist_ok=True)

    filename = f"{video.stem}_scene_$SCENE_NUMBER.mp4"
    command = [
        sys.executable,
        "-m",
        "scenedetect",
        "-i",
        str(video),
        "-o",
        str(output),
        "-m",
        args.min_scene_len,
        "detect-content",
        "--threshold",
        str(args.threshold),
        "split-video",
        "--filename",
        filename,
        "--preset",
        args.preset,
        "--rate-factor",
        str(args.crf),
    ]

    print(f"Splitting {video.name} into {output}", flush=True)
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error


if __name__ == "__main__":
    main()
