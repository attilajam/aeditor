"""Low resolution, timestamp-aware footage analysis and boundary refinement."""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import json
import math
import os
import tempfile
import numpy as np

from .cache import key_for, read_cache, write_json
from .features import DESCRIPTOR_VERSION, describe, motion_summary, legacy_visual_distance
from .probe import MediaError, fingerprint, fit_filter, frame_timestamps, probe, run

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mts", ".m2ts", ".mpeg", ".mpg", ".ogv", ".ts"}
SCENE_DETECTOR_VERSION = 2


def analysis_dimensions(cfg):
    w, h = cfg.dimensions
    factor = cfg.analysis_long_side / max(w, h)
    return max(16, round(w * factor / 2) * 2), max(16, round(h * factor / 2) * 2)


def content_frame(frame, info, cfg):
    """Return visible source content without deliberate contain padding."""
    if cfg.fit == "contain":
        h, w = frame.shape[:2]
        dw, dh = info["display_width"], info["display_height"]
        factor = min(w/dw, h/dh)
        cw, ch = max(2, int(dw*factor)//2*2), max(2, int(dh*factor)//2*2)
        x, y = (w-cw)//2, (h-ch)//2
        return frame[y:y+ch, x:x+cw]
    return frame


def composition_descriptor(frame, info, cfg):
    """Match the whole composition; measure quality on content without fit padding."""
    descriptor = describe(frame)
    if cfg.fit == "contain":
        descriptor["quality"] = describe(content_frame(frame, info, cfg), spatial=False)["quality"]
    return descriptor


def sample_indices(times, rate):
    result = []
    last = -math.inf
    for i, t in enumerate(times):
        if t - last >= 1 / rate - 1e-6:
            result.append(i)
            last = t
    return result


def decode_samples(path, times, cfg, rate=None):
    rate = rate or cfg.samples_per_second
    w, h = analysis_dimensions(cfg)
    vf = f"setpts=PTS-STARTPTS,select='isnan(prev_selected_t)+gte(t-prev_selected_t,{1/rate-1e-6:.12f})'," + fit_filter(w, h, cfg.fit)
    raw = run(["ffmpeg", "-v", "error", "-threads", cfg.threads, "-i", path, "-map", "0:v:0", "-an",
               "-vf", vf, "-fps_mode", "passthrough", "-threads", cfg.threads,
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], binary=True)
    indices = sample_indices(times, rate)
    expected = len(indices) * w * h * 3
    if len(raw) != expected:
        raise MediaError(f"decoded samples disagree with presentation timestamps in {path}: {len(raw)} bytes, expected {expected}")
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).copy(), [times[i] for i in indices]


def detect_scene_boundaries(frames, times, info, cfg):
    """Combine composition changes with abrupt pixel changes relative to motion.

    A cut between related shots may retain nearly the same color histogram.
    Detect isolated spatial changes too, while sustained motion raises the
    temporal baseline. Deliberate output padding is excluded from detection.
    """
    content = [content_frame(frame, info, cfg) for frame in frames]
    descriptors = [describe(frame, spatial=False) for frame in content]
    pixel_changes = [float(np.abs(b.astype(np.float32)-a.astype(np.float32)).mean()/255)
                     for a, b in zip(content, content[1:])]
    pixel_threshold = max(cfg.scene_threshold/3, 4*float(np.median(pixel_changes))) if pixel_changes else 1.
    cuts = []
    for i, pixel_change in enumerate(pixel_changes, 1):
        distances = legacy_visual_distance(descriptors[i-1], descriptors[i])
        change = .45*distances["color"] + .45*distances["structure"] + .10*distances["edge"]
        if change >= cfg.scene_threshold or pixel_change >= pixel_threshold:
            cuts.append([times[i-1], times[i]])
    return cuts


def invalidate_cross_cut_motion(clip):
    times, cuts = clip["times"], clip["scene_boundaries"]
    for i in range(len(times)):
        lo, hi = max(0, i-1), min(len(times)-1, i+1)
        if any(times[lo] <= b and times[hi] >= a for a, b in cuts):
            clip["motions"][i] = {"vx": 0., "vy": 0., "magnitude": 0., "reliability": 0.}


def analyze_clip(path, cfg, info=None):
    identity = fingerprint(path)
    settings = {"samples": cfg.samples_per_second, "dimensions": analysis_dimensions(cfg),
                "fit": cfg.fit, "scene_threshold": cfg.scene_threshold,
                # Keep the decoded-frame cache identity stable. Descriptor and
                # scene versions below are independently migrated from frames.
                "ffmpeg": run(["ffmpeg", "-version"]).splitlines()[0], "feature_version": 3}
    import cv2
    settings.update(opencv=cv2.__version__, numpy=np.__version__)
    key = key_for(identity, settings, "video")
    root = Path(cfg.cache_dir)
    metadata = root / f"{key}.json"
    arrays = root / f"{key}.npz"
    cached = read_cache(metadata)
    if cached and arrays.is_file():
        try:
            with np.load(arrays, allow_pickle=False) as data:
                cached["frames"] = data["frames"].copy()
            cached["identity"] = identity
            cached["probe"]["path"] = str(Path(path).resolve())
            cached["probe"]["format"]["filename"] = str(Path(path).resolve())
            changed = False
            if cached.get("descriptor_version", 3) != DESCRIPTOR_VERSION:
                cached["descriptors"] = [composition_descriptor(f, cached["probe"], cfg) for f in cached["frames"]]
                cached["descriptor_version"] = DESCRIPTOR_VERSION
                cached["reference_sharpness"] = float(np.median([d["quality"]["sharpness"] for d in cached["descriptors"]]))
                changed = True
            if cached.get("scene_detector_version", 1) != SCENE_DETECTOR_VERSION:
                # Frames and descriptors remain valid; migrate only scene
                # metadata instead of decoding all footage again.
                cached["scene_boundaries"] = detect_scene_boundaries(cached["frames"], cached["times"], cached["probe"], cfg)
                cached["scene_detector_version"] = SCENE_DETECTOR_VERSION
                invalidate_cross_cut_motion(cached)
                changed = True
            if changed:
                write_json(metadata, {k: v for k, v in cached.items() if k != "frames"})
            return cached
        except (OSError, ValueError, KeyError):
            pass
    info = info or probe(path)
    if not info["video"]:
        raise MediaError("no video stream")
    pts, duration, origin = frame_timestamps(path)
    frames, times = decode_samples(path, pts, cfg)
    if len(frames) < 1:
        raise MediaError("no decodable footage")
    descriptors = [composition_descriptor(f, info, cfg) for f in frames]
    cuts = detect_scene_boundaries(frames, times, info, cfg)
    motions = []
    for i in range(len(frames)):
        lo, hi = max(0, i-1), min(len(frames), i+2)
        if any(times[lo] <= b and times[hi-1] >= a for a, b in cuts):
            motions.append({"vx": 0., "vy": 0., "magnitude": 0., "reliability": 0.})
        else:
            motions.append(motion_summary(list(frames[lo:hi]), times[lo:hi]))
    sharp = [d["quality"]["sharpness"] for d in descriptors]
    result = {"identity": identity, "probe": info, "duration": duration, "pts_origin": origin,
              "pts": pts, "times": times, "descriptors": descriptors, "motions": motions,
              "scene_boundaries": cuts, "scene_detector_version": SCENE_DETECTOR_VERSION,
              "descriptor_version": DESCRIPTOR_VERSION,
              "reference_sharpness": float(np.median(sharp))}
    root.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".npz", dir=root)
    os.close(fd)
    try:
        np.savez_compressed(tmp, frames=frames)
        os.replace(tmp, arrays)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    write_json(metadata, result)
    result["frames"] = frames
    return result


def inspect_directory(directory, cfg, exclude=()):
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise MediaError(f"clips directory does not exist: {directory}")
    exclusions = {str(Path(p).resolve()) for p in exclude}
    usable, skipped, paths = [], [], []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or str(path) in exclusions:
            continue
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            skipped.append({"path": str(path), "reason": "unsupported file extension"})
            continue
        paths.append(path)
    inspected = {}
    if cfg.size == "auto":
        for path in paths:
            try:
                info = probe(path)
                if not info["video"]:
                    raise MediaError("no video stream")
                inspected[path] = info
            except (MediaError, ValueError) as exc:
                skipped.append({"path": str(path), "reason": str(exc)})
        paths = list(inspected)
        if paths:
            cfg.size = source_output_size(list(inspected.values()))
            cfg.validate()
            print(f"Output size: {cfg.size} (source aspect ratio)", flush=True)
    for path in paths:
        try:
            print(f"Analyzing {path.name}", flush=True)
            usable.append(analyze_clip(path, cfg, inspected.get(path)))
        except (MediaError, ValueError) as exc:
            skipped.append({"path": str(path), "reason": str(exc)})
    if not usable:
        reasons = "; ".join(f"{Path(s['path']).name}: {s['reason'][:200]}" for s in skipped[:5])
        raise MediaError(f"no usable footage in {directory}" + (f" ({reasons})" if reasons else " (directory is empty)"))
    return usable, skipped


def source_output_size(infos):
    """Choose the dominant displayed aspect, then its most common native size.

    Display sizes already account for rotation and sample aspect ratio. Ties
    favor the larger native size. Odd dimensions round down for H.264/yuv420p.
    """
    groups = defaultdict(list)
    for info in infos:
        w, h = info["display_width"], info["display_height"]
        if w > 0 and h > 0:
            groups[Fraction(w, h)].append((w, h))
    if not groups:
        raise MediaError("cannot infer output dimensions: no valid displayed video dimensions")
    sizes = max(groups.values(), key=lambda group: (len(group), max(w*h for w, h in group)))
    counts = Counter(sizes)
    w, h = max(counts, key=lambda size: (counts[size], size[0]*size[1], size))
    return f"{max(16, w//2*2)}x{max(16, h//2*2)}"


def visible_source_index(pts, source_start, output_index, speed, fps):
    """Mirror fps round=up: a source frame becomes available at ceil(relative PTS*fps).

    Trimming removes frames before source_start. fps start_time=0 backfills the
    first retained frame until its timestamp. Selection uses PTS, including VFR.
    """
    clock = 90_000_000
    start_tick = round(source_start * clock)
    first = bisect_left(pts, (start_tick-.5)/clock)
    threshold = source_start + output_index / fps * speed
    index = max(first, min(len(pts)-1, bisect_right(pts, threshold) - 1))
    def available(i):
        # setpts truncates to the 90 MHz filter time base before fps round=up.
        tick = int((round(pts[i]*clock)-start_tick)/speed)
        return tick * fps <= output_index * clock
    while index+1 < len(pts) and available(index+1):
        index += 1
    while index > first and not available(index):
        index -= 1
    return index


def closest_sample(clip, timestamp):
    times = clip["times"]
    i = bisect_left(times, timestamp)
    return min(range(max(0, i-1), min(len(times), i+1)), key=lambda j: abs(times[j]-timestamp))


def decode_boundary_neighborhood(clip, start_index, count, cfg):
    """Decode actual adjacent native frames, never cross a cut for motion."""
    stop = min(len(clip["pts"]), start_index + count)
    w, h = analysis_dimensions(cfg)
    vf = f"select='between(n,{start_index},{stop-1})'," + fit_filter(w, h, cfg.fit)
    raw = run(["ffmpeg", "-v", "error", "-threads", cfg.threads, "-i", clip["identity"]["path"],
               "-an", "-vf", vf, "-frames:v", stop-start_index, "-fps_mode", "passthrough",
               "-threads", cfg.threads, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], binary=True)
    if len(raw) != (stop-start_index)*w*h*3:
        raise MediaError(f"cannot decode boundary in {clip['identity']['path']}")
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).copy(), clip["pts"][start_index:stop]


def serializable_analysis(clip):
    return {k: v for k, v in clip.items() if k != "frames"}
