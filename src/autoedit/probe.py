"""FFmpeg subprocesses, content identities, and presentation timestamp inspection."""
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import shutil
import subprocess


class MediaError(RuntimeError):
    pass


def require_tools():
    for name in ("ffmpeg", "ffprobe"):
        if shutil.which(name) is None:
            raise MediaError(f"{name} is required on PATH; install FFmpeg first")


def run(args, *, binary=False):
    try:
        result = subprocess.run([str(a) for a in args], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, check=False)
    except OSError as exc:
        raise MediaError(f"cannot run {args[0]}: {exc}") from exc
    if result.returncode:
        detail = result.stderr.decode(errors="replace")[-6000:]
        raise MediaError(f"{args[0]} failed ({result.returncode}) for {args!r}\n{detail}")
    return result.stdout if binary else result.stdout.decode(errors="replace")


def fingerprint(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise MediaError(f"missing source file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "size": path.stat().st_size}


def verify_identity(identity):
    actual = fingerprint(identity["path"])
    if actual["sha256"] != identity["sha256"] or actual["size"] != identity["size"]:
        raise MediaError(f"source changed since analysis: {identity['path']}; run create again")


def probe(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise MediaError(f"input file does not exist: {path}")
    info = json.loads(run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", path]))
    videos = [s for s in info.get("streams", []) if s["codec_type"] == "video" and not s.get("disposition", {}).get("attached_pic")]
    audios = [s for s in info.get("streams", []) if s["codec_type"] == "audio"]
    video = videos[0] if videos else None
    audio = audios[0] if audios else None
    chosen = video or audio or {}
    duration = float(chosen.get("duration", info.get("format", {}).get("duration", 0)))
    if duration <= 0:
        raise MediaError(f"no positive media duration in {path}")
    rotation = 0
    if video:
        rotation = int(float(video.get("tags", {}).get("rotate", 0)))
        for side in video.get("side_data_list", []):
            if "rotation" in side:
                rotation = int(float(side["rotation"]))
    width, height = (video.get("width", 0), video.get("height", 0)) if video else (0, 0)
    sar = (video or {}).get("sample_aspect_ratio", "1:1")
    try:
        sar_value = float(Fraction(sar.replace(":", "/")))
    except (ValueError, ZeroDivisionError):
        sar_value = 1.
    display = (round(width * sar_value), height)
    if abs(rotation) % 180 == 90:
        display = display[::-1]
    return {"path": str(path), "duration": duration, "video": video, "audio": audio,
            "rotation": rotation, "display_width": display[0], "display_height": display[1],
            "streams": info.get("streams", []), "format": info.get("format", {})}


def frame_timestamps(path):
    data = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_streams",
                           "-show_entries", "stream=time_base:frame=best_effort_timestamp,duration,pkt_duration", "-of", "json", path]))
    records = [r for r in data.get("frames", []) if "best_effort_timestamp" in r]
    if not records:
        raise MediaError(f"no decodable video frames: {path}")
    tb = Fraction(data["streams"][0]["time_base"])
    origin_pts = int(records[0]["best_effort_timestamp"])
    origin = float(origin_pts * tb)
    times = [float((int(r["best_effort_timestamp"]) - origin_pts) * tb) for r in records]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise MediaError(f"nonmonotonic presentation timestamps in {path}")
    last = records[-1]
    tail = float(int(last.get("duration", last.get("pkt_duration", 0))) * tb)
    if tail <= 0:
        tail = times[-1] - times[-2] if len(times) > 1 else 1 / 30
    return times, times[-1] + tail, origin


def fit_filter(width, height, fit):
    # FFmpeg autorotates before this filter. Normalize non-square source pixels first.
    base = "scale=iw*sar:ih,setsar=1,"
    if fit == "stretch":
        return base + f"scale={width}:{height},setsar=1"
    if fit == "contain":
        return base + f"scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    return base + f"scale={width}:{height}:force_original_aspect_ratio=increase:force_divisible_by=2,crop={width}:{height},setsar=1"
