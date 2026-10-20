"""Decode verification, native-source boundary checks, and soundtrack alignment."""
from fractions import Fraction
from pathlib import Path
import json
import cv2
import numpy as np
from scipy.signal import correlate, correlation_lags

from .audio import _decode, cut_is_active
from .probe import MediaError, fit_filter, frame_timestamps, probe, run
from .render import check_plan, decoded_frame_count, verify_sources
from .video import analysis_dimensions, visible_source_index


def validate_user_timing(plan):
    """Validate saved manual timing provenance and return its beat/quiet report."""
    timing = plan.get("user_timing")
    if timing is None:
        return {"explicit": False, "valid": True, "errors": [], "manual_adjustments": [],
                "manual_cut_frames": [], "max_auto_beat_error_seconds": 0.}
    errors = []
    fps = plan.get("configuration", {}).get("fps")
    total = plan.get("output", {}).get("total_frames")
    try:
        if not isinstance(timing, dict) or timing.get("version") != 1:
            raise ValueError("unsupported or malformed user_timing version")
        for key in ("base_cut_frames", "cut_frames", "adjustments", "no_cut_ranges"):
            if key not in timing:
                raise ValueError(f"user_timing is missing {key}")
        base = timing["base_cut_frames"]
        proposed = timing["cut_frames"]
        adjustments = timing["adjustments"]
        ranges = timing["no_cut_ranges"]
        if type(fps) is not int or fps < 1 or type(total) is not int:
            raise ValueError("invalid frame rate or total frame count")
        for name, values in (("base_cut_frames", base), ("cut_frames", proposed)):
            if (not isinstance(values, list) or len(values) < 2 or
                    any(type(frame) is not int for frame in values) or values[0] != 0 or values[-1] != total or
                    any(b <= a for a, b in zip(values, values[1:]))):
                raise ValueError(f"{name} must be strictly increasing exact integer frame boundaries")
        if not isinstance(ranges, list):
            raise ValueError("no_cut_ranges must be a list")
        normalized_ranges = []
        for pair in ranges:
            if (not isinstance(pair, list) or len(pair) != 2 or any(type(v) is not int for v in pair) or
                    pair[0] < 0 or pair[0] >= pair[1] or pair[1] > total):
                raise ValueError("no_cut_ranges must contain valid [start_frame, end_frame) integer pairs")
            normalized_ranges.append(pair)
        if not isinstance(adjustments, list):
            raise ValueError("adjustments must be a list")
        ids, active = set(), []
        for item in adjustments:
            if not isinstance(item, dict) or not {"id", "kind", "detected_time", "original_frame", "frame", "enabled"} <= set(item):
                raise ValueError("each adjustment needs id, kind, detected_time, original_frame, frame, and enabled")
            if (not isinstance(item["id"], str) or not item["id"] or item["id"] in ids or
                    item["kind"] not in {"manual", "automatic", "beat", "kick", "onset"} or
                    type(item["original_frame"]) is not int or type(item["frame"]) is not int or
                    not 0 <= item["original_frame"] <= total or not 0 <= item["frame"] <= total or
                    type(item["enabled"]) is not bool):
                raise ValueError("adjustment fields have invalid types or duplicate ids")
            detected_time = item["detected_time"]
            if item["kind"] == "manual":
                if detected_time is not None:
                    raise ValueError(f"manual adjustment {item['id']} must have null detected_time")
            elif item["kind"] in {"beat", "kick", "onset"}:
                event_key = "onset_times" if item["kind"] == "onset" else "beat_times"
                events = plan.get("audio", {}).get(event_key, [])
                if (type(detected_time) not in (int, float) or not np.isfinite(detected_time) or
                        not any(abs(float(detected_time) - event) <= 1e-7 for event in events) or
                        round(float(detected_time) * fps) != item["original_frame"]):
                    raise ValueError(f"{item['kind']} adjustment {item['id']} is not backed by its detected event")
            else:
                base_interiors = set(base[1:-1])
                if item["original_frame"] not in base_interiors:
                    raise ValueError(f"automatic adjustment {item['id']} has no original base cut")
                if detected_time is not None:
                    raise ValueError(f"automatic adjustment {item['id']} has invalid detected_time")
            ids.add(item["id"])
            frame = item["frame"]
            if item["enabled"] and 0 < frame < total and not any(a <= frame < b for a, b in normalized_ranges):
                active.append(frame)
        reconstructed = [0, *sorted(active), total]
        if len(set(reconstructed)) != len(reconstructed):
            raise ValueError("multiple enabled markers reconstruct the same cut frame")
        if reconstructed != proposed:
            raise ValueError("adjustments and no_cut_ranges do not reconstruct user_timing.cut_frames")
        if proposed != plan.get("cut_frames"):
            raise ValueError("plan cut_frames disagree with user_timing.cut_frames")
        cuts = proposed[1:-1]
        for a, b in normalized_ranges:
            if any(a <= frame < b for frame in cuts):
                raise ValueError(f"cut frame inside strict no_cut_range [{a}, {b})")
        manual = [item for item in adjustments if item["enabled"] and item["frame"] in cuts and
                  (item["kind"] == "manual" or item["frame"] != item["original_frame"])]
        unchanged = [item["frame"] for item in adjustments if item["enabled"] and item["frame"] in cuts and
                     item["kind"] in {"automatic", "beat", "kick", "onset"} and item["frame"] == item["original_frame"]]
        beats = plan.get("audio", {}).get("beat_times", [])
        fallback = plan.get("audio", {}).get("fallback")
        auto_errors = []
        if fallback == "none":
            for frame in unchanged:
                distance = min((abs(frame / fps - beat) for beat in beats), default=float("inf"))
                auto_errors.append(distance)
                if distance > 1 / fps + 1e-9:
                    errors.append(f"untouched detected cut at frame {frame} is not on the detected beat grid")
        return {"explicit": True, "valid": not errors, "errors": errors,
                "manual_adjustments": manual, "manual_cut_frames": [item["frame"] for item in manual],
                "max_auto_beat_error_seconds": max(auto_errors, default=0.),
                "no_cut_ranges": normalized_ranges}
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        errors.append(str(exc))
        return {"explicit": True, "valid": False, "errors": errors,
                "manual_adjustments": [], "manual_cut_frames": [],
                "max_auto_beat_error_seconds": 0., "no_cut_ranges": []}


def decode_frames(path, dimensions, indices=None, fit=None):
    w, h = dimensions
    filters = []
    if indices is not None:
        indices = sorted(set(indices))
        filters.append("select='" + "+".join(f"eq(n,{i})" for i in indices) + "'")
    filters.append(fit_filter(w, h, fit) if fit else f"scale={w}:{h}")
    raw = run(["ffmpeg", "-v", "error", "-threads", 2, "-i", path, "-map", "0:v:0", "-an", "-vf", ",".join(filters),
               "-fps_mode", "passthrough", "-threads", 2, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], binary=True)
    if len(raw) % (w*h*3):
        raise MediaError(f"incomplete raw frame from {path}")
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    if indices is not None:
        if len(frames) != len(indices):
            raise MediaError(f"could not decode all requested frames in {path}")
        return dict(zip(indices, frames))
    return frames


def audio_alignment(plan, video):
    duration = min(5., plan["audio"]["duration"])
    original = _decode(plan["audio"]["identity"]["path"], plan["audio"]["audio_start"], duration)
    rendered = _decode(video, 0., duration)
    n = min(len(original), len(rendered))
    if n < 64:
        return {"valid": False, "reason": "too few decoded audio samples"}
    a, b = original[:n].astype(np.float64), rendered[:n].astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-5 or nb < 1e-5:
        equal_silence = bool(max(na, nb) < 1e-3)
        return {"valid": equal_silence, "correlation": None, "lag_seconds": 0., "silence": True}
    corr = correlate(b, a, mode="full", method="fft")
    lags = correlation_lags(n, n, mode="full")
    mask = np.abs(lags) <= 2205  # at most 100 ms of codec-delay search
    selected = np.flatnonzero(mask)[int(np.argmax(corr[mask]))]
    score = float(corr[selected]/(na*nb))
    lag = int(lags[selected])/22050
    return {"valid": bool(score >= .70 and abs(lag) <= .05), "correlation": score,
            "lag_seconds": lag, "silence": False, "comparison_seconds": duration}


def validate_video(plan, video, verify_inputs=True):
    cfg = check_plan(plan)
    if verify_inputs:
        verify_sources(plan)
    video = Path(video).resolve()
    info = probe(video)
    errors, warnings = [], []
    stream, audio = info["video"], info["audio"]
    out = plan["output"]
    if not stream:
        raise MediaError("rendered output contains no video stream")
    if stream["codec_name"] != "h264" or stream.get("pix_fmt") != "yuv420p":
        errors.append("output must be H.264 yuv420p")
    if (stream["width"], stream["height"]) != (out["width"], out["height"]):
        errors.append("output dimensions disagree with plan")
    if abs(float(Fraction(stream["avg_frame_rate"]))-cfg.fps) > 1e-6:
        errors.append("output frame rate disagrees with plan")
    if not audio or audio["codec_name"] != "aac":
        errors.append("output must contain AAC soundtrack")
    elif len([s for s in info["streams"] if s["codec_type"] == "audio"]) != 1:
        errors.append("expected exactly one soundtrack stream")
    count = decoded_frame_count(video)
    if count != out["total_frames"]:
        errors.append(f"decoded frame count {count} differs from planned {out['total_frames']}")
    # Fail on decode errors even if ffprobe happened to count damaged frames.
    run(["ffmpeg", "-v", "error", "-xerror", "-i", video, "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"])
    pts, duration, _ = frame_timestamps(video)
    if any(abs(t-i/cfg.fps) > 1/90000+.000001 for i, t in enumerate(pts)):
        errors.append("video timestamps contain a gap or timing drift")
    if abs(duration-out["total_frames"]/cfg.fps) > 1/cfg.fps:
        errors.append("decoded duration disagrees with planned frame timeline")
    timing_report = validate_user_timing(plan)
    if not timing_report["valid"]:
        errors.extend(f"invalid user timing: {error}" for error in timing_report["errors"])
    beat_errors = []
    activity = plan["audio"].get("cut_activity")
    quiet_violations = []
    if activity and activity["enabled"] and not timing_report["explicit"]:
        forced = set(activity.get("forced_cut_frames", []))
        quiet_violations = [frame for frame in plan["cut_frames"][1:-1]
                            if frame not in forced and not cut_is_active(activity, frame/cfg.fps)]
        if quiet_violations:
            errors.append(f"unrecorded transitions inside protected quiet sections at frames {quiet_violations}")
        if forced:
            warnings.append(f"{len(forced)} quiet-section cut(s) required by available footage; recorded as explicit relaxations")
    if plan["audio"]["fallback"] == "none":
        beats = plan["audio"]["beat_times"]
        manual_frames = set(timing_report.get("manual_cut_frames", []))
        for frame in plan["cut_frames"][1:-1]:
            if frame in manual_frames:
                continue
            distance = min((abs(frame/cfg.fps-beat) for beat in beats), default=float("inf"))
            beat_errors.append(distance)
            if distance > 1/cfg.fps+1e-9:
                errors.append(f"cut at frame {frame} is not on the detected beat grid")
    dims = analysis_dimensions(cfg)
    indices = {c["output_start_frame"] for c in plan["segments"]}
    indices.update(c["output_start_frame"]+c["output_frame_count"]-1 for c in plan["segments"])
    actual = decode_frames(video, dims, sorted(indices))
    wanted = {}
    positions = []
    for i, segment in enumerate(plan["segments"]):
        source = plan["sources"][segment["clip"]]
        pts_source = source["pts"]
        for end in (False, True):
            native = visible_source_index(pts_source, segment["source_start"], segment["output_frame_count"]-1 if end else 0,
                                          segment["speed"], cfg.fps)
            wanted.setdefault(segment["clip"], set()).add(native)
            output_index = segment["output_start_frame"] + (segment["output_frame_count"]-1 if end else 0)
            positions.append((i, "exit" if end else "entry", segment["clip"], native, output_index))
    sources = {clip_id: decode_frames(plan["sources"][clip_id]["identity"]["path"], dims, sorted(native), cfg.fit)
               for clip_id, native in wanted.items()}
    checks = []
    for slot, kind, clip_id, native, index in positions:
        original = sources[clip_id][native]
        rendered = actual[index]
        error = float(np.mean(np.abs(rendered.astype(float)-original.astype(float)))/255)
        black_actual = float(np.mean(cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY) < 12))
        black_source = float(np.mean(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) < 12))
        if error > .065:
            errors.append(f"slot {slot} {kind} frame differs from planned source (MAE={error:.4f})")
        if black_actual > .98 and black_source < .95:
            errors.append(f"unexpected black frame at output index {index}")
        checks.append({"slot": slot, "kind": kind, "output_frame": index, "source_frame": native,
                       "normalized_pixel_mae": error, "black_fraction": black_actual})
    alignment = audio_alignment(plan, video) if audio else {"valid": False}
    if not alignment["valid"]:
        errors.append("decoded soundtrack does not match the selected audio offset")
    if plan["audio"]["fallback"] != "none":
        warnings.append(f"rhythm fallback: {plan['audio']['fallback']}; generated boundaries are not detected beats")
    max_auto_beat_error = (timing_report["max_auto_beat_error_seconds"] if timing_report["explicit"]
                           else max(beat_errors, default=0.))
    return {"version": 1, "valid": not errors, "video": str(video), "errors": errors, "warnings": warnings,
            "decoded_frames": count, "duration": duration, "expected_frames": out["total_frames"],
            "max_beat_error_seconds": max(beat_errors, default=0.),
            "max_auto_beat_error_seconds": max_auto_beat_error,
            "manual_adjustments": timing_report.get("manual_adjustments", []), "audio_alignment": alignment,
            "quiet_sections": {"checked": bool(activity and activity["enabled"]),
                               "quiet_intervals": activity.get("quiet_intervals", []) if activity else [],
                               "forced_cut_frames": activity.get("forced_cut_frames", []) if activity else [],
                               "violating_cut_frames": quiet_violations},
            "boundary_checks": checks, "fully_decoded": True,
            "video_codec": stream["codec_name"], "pixel_format": stream.get("pix_fmt"),
            "audio_codec": audio["codec_name"] if audio else None,
            "width": stream["width"], "height": stream["height"], "fps": cfg.fps}
