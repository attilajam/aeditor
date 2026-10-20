"""Render planned intervals, count every intermediate frame, then atomically publish."""
from pathlib import Path
import json
import os
import shutil
import tempfile

from .cache import write_json
from .config import Config
from .probe import MediaError, fit_filter, run, verify_identity

PLAN_VERSION = 1


def load_plan(path):
    try:
        plan = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise MediaError(f"cannot read edit plan {path}: {exc}") from exc
    if plan.get("version") != PLAN_VERSION:
        raise MediaError(f"unsupported plan version: {plan.get('version')}")
    check_plan(plan)
    return plan


def check_plan(plan):
    try:
        cfg = Config(**plan["configuration"]).validate()
        output = plan["output"]
        if output["fps"] != cfg.fps or [output["width"], output["height"]] != list(cfg.dimensions):
            raise ValueError("output settings disagree with configuration")
        cuts = plan["cut_frames"]
        if cuts[0] != 0 or cuts[-1] != output["total_frames"] or any(b <= a for a, b in zip(cuts, cuts[1:])):
            raise ValueError("invalid frame boundaries")
        if len(plan["segments"]) != len(cuts)-1:
            raise ValueError("segment count disagrees with cuts")
        used = set()
        for i, c in enumerate(plan["segments"]):
            if c["output_start_frame"] != cuts[i] or c["output_frame_count"] != cuts[i+1]-cuts[i]:
                raise ValueError("segment frame range disagrees with cuts")
            if not cfg.speed_min-1e-8 <= c["speed"] <= cfg.speed_max+1e-8:
                raise ValueError("segment speed outside configured limits")
            if not 0 <= c["clip"] < len(plan["sources"]):
                raise ValueError("unknown clip index")
            source = plan["sources"][c["clip"]]
            if c["source_start"] < 0 or c["source_end"] > source["duration"] + 1e-7:
                raise ValueError("source interval out of bounds")
            consumed = c["output_frame_count"]/cfg.fps*c["speed"]
            if abs(c["source_end"]-c["source_start"]-consumed) > 1e-7:
                raise ValueError("source duration does not match output duration * speed")
            if cfg.reuse == "forbid" and c["clip"] in used:
                raise ValueError("forbidden clip reuse")
            used.add(c["clip"])
        if abs(plan["audio"]["duration"]-output["total_frames"]/cfg.fps) > 1e-7:
            raise ValueError("audio and video timelines disagree")
        if "user_timing" in plan:
            from .review_planner import validate_user_timing
            timing = validate_user_timing(plan)
            if not timing["valid"]:
                raise ValueError("invalid user timing: " + "; ".join(timing["errors"]))
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise MediaError(f"invalid edit plan: {exc}") from exc
    return cfg


def verify_sources(plan):
    verify_identity(plan["audio"]["identity"])
    for source in plan["sources"]:
        verify_identity(source["identity"])


def segment_filter(segment, cfg, dimensions=None):
    clock = 90_000_000
    start = round(segment["source_start"]*clock)
    end = round(segment["source_end"]*clock)
    frames = segment["output_frame_count"]
    w, h = dimensions or cfg.dimensions
    # Source timestamps are normalized to first presentation time, including VFR.
    # round=up takes the latest available source frame at each output timestamp.
    return (f"settb=1/{clock},setpts=PTS-STARTPTS,trim=start_pts={start}:end_pts={end},"
            f"setpts=(PTS-{start})/{segment['speed']:.12f},"
            f"fps={cfg.fps}:start_time=0:round=up:eof_action=pass,trim=end_frame={frames},"
            f"setpts=N/({cfg.fps}*TB)," + fit_filter(w, h, cfg.fit) + ",format=yuv420p")


def decoded_frame_count(path):
    data = json.loads(run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                           "-show_entries", "stream=nb_read_frames", "-of", "json", path]))
    try:
        return int(data["streams"][0]["nb_read_frames"])
    except (KeyError, ValueError, IndexError):
        raise MediaError(f"could not count decoded frames in {path}") from None


def render_plan(plan, output, overwrite=False):
    from .validate import validate_video
    cfg = check_plan(plan)
    verify_sources(plan)
    output = Path(output).expanduser().resolve()
    if output.suffix.lower() != ".mp4":
        raise MediaError("output must have .mp4 extension")
    if output.exists() and not overwrite:
        raise MediaError(f"output already exists: {output}; use --overwrite")
    source_paths = {s["identity"]["path"] for s in plan["sources"]} | {plan["audio"]["identity"]["path"]}
    if str(output) in source_paths:
        raise MediaError("output must not replace an input source")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".autoedit-", dir=output.parent))
    logs = []
    try:
        segments = []
        for i, segment in enumerate(plan["segments"]):
            part = temporary_root / f"part-{i:05}.mp4"
            source = plan["sources"][segment["clip"]]["identity"]["path"]
            command = ["ffmpeg", "-v", "error", "-y", "-threads", cfg.threads, "-i", source, "-map", "0:v:0", "-an",
                       "-vf", segment_filter(segment, cfg), "-c:v", "libx264", "-crf", 0, "-preset", "ultrafast",
                       "-threads", cfg.threads, "-pix_fmt", "yuv420p", "-video_track_timescale", cfg.fps*3000, "-map_metadata", "-1", part]
            logs.append([str(x) for x in command])
            run(command)
            actual = decoded_frame_count(part)
            if actual != segment["output_frame_count"]:
                raise MediaError(f"segment {i} decoded {actual} frames; expected {segment['output_frame_count']}. "
                                 "Insufficient decodable footage; no freeze or black filler was inserted.")
            segments.append(part)
        # ffconcat files are generated names only; input paths never enter its syntax.
        concat = temporary_root / "segments.ffconcat"
        concat.write_text("ffconcat version 1.0\n" + "".join(
            f"file '{part.name}'\nduration {segment['output_frame_count']/cfg.fps:.12f}\n"
            for part, segment in zip(segments, plan["segments"])))
        final = temporary_root / "finished.mp4"
        audio = plan["audio"]
        duration = plan["output"]["total_frames"] / cfg.fps
        # Decode interval precisely; atrim uses sample timestamps, resets the audio clock.
        command = ["ffmpeg", "-v", "error", "-y", "-threads", cfg.threads, "-f", "concat", "-safe", "1", "-i", concat,
                   "-i", audio["identity"]["path"], "-map", "0:v:0", "-map", "1:a:0",
                   "-vf", f"settb=1/{cfg.fps*3000},setpts=N*3000", "-r", cfg.fps, "-fps_mode", "cfr",
                   "-c:v", "libx264", "-crf", cfg.crf, "-preset", cfg.preset, "-threads", cfg.threads,
                   "-pix_fmt", "yuv420p",
                   "-af", f"asetpts=PTS-STARTPTS,atrim=start={audio['audio_start']:.12f}:duration={duration:.12f},asetpts=PTS-STARTPTS",
                   "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-map_metadata", "-1",
                   "-video_track_timescale", cfg.fps*3000, "-movflags", "+faststart", final]
        logs.append([str(x) for x in command])
        run(command)
        report = validate_video(plan, final, verify_inputs=False)
        if not report["valid"]:
            raise MediaError("render failed validation: " + "; ".join(report["errors"]))
        os.replace(final, output)
        report["video"] = str(output)
        write_json(output.with_suffix(".validation.json"), report)
        output.with_suffix(".error.json").unlink(missing_ok=True)
        return report
    except Exception as exc:
        write_json(output.with_suffix(".error.json"), {"error": str(exc), "commands": logs})
        raise
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
