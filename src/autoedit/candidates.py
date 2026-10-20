"""Feasible source intervals, balanced pruning, and native-frame refinement."""
from bisect import bisect_left
from collections import defaultdict
import math
import numpy as np

from .features import boundary_quality, describe, motion_summary
from .models import Candidate
from .probe import MediaError, fit_filter, run
from .video import analysis_dimensions, closest_sample, composition_descriptor, visible_source_index


def regions(clip):
    bounds = [0.]
    result = []
    for a, b in clip["scene_boundaries"]:
        if a > bounds[-1]:
            result.append((bounds[-1], a))
        bounds.append(b)
    if clip["duration"] > bounds[-1]:
        result.append((bounds[-1], clip["duration"]))
    return result


def max_slot_duration(clips, cfg):
    return max(b-a for clip in clips for a, b in regions(clip)) / cfg.speed_min


def make_candidate(clip_id, clip, start, speed, out_start, count, cfg, allow_cuts=False):
    source_duration = count / cfg.fps * speed
    end = start + source_duration
    if start < -1e-8 or end > clip["duration"] + 1e-8 or not cfg.speed_min-1e-9 <= speed <= cfg.speed_max+1e-9:
        return None
    crossing = any(start < b-1e-8 and end > a+1e-8 for a, b in clip["scene_boundaries"])
    if crossing and not allow_cuts:
        return None
    pts = clip["pts"]
    entry = visible_source_index(pts, start, 0, speed, cfg.fps)
    exit_ = visible_source_index(pts, start, count-1, speed, cfg.fps)
    if entry >= len(pts) or pts[entry] >= end-1e-8 or pts[exit_] >= end+1e-8:
        return None
    i, j = closest_sample(clip, pts[entry]), closest_sample(clip, pts[exit_])
    entry_desc, exit_desc = clip["descriptors"][i], clip["descriptors"][j]
    quality_entry = boundary_quality(entry_desc, clip["reference_sharpness"])
    quality_exit = boundary_quality(exit_desc, clip["reference_sharpness"])
    inside = [d for t, d in zip(clip["times"], clip["descriptors"]) if start <= t < end]
    interior_badness = float(np.mean([boundary_quality(d, clip["reference_sharpness"]) for d in inside])) if inside else (quality_entry+quality_exit)/2
    quality = {"entry_penalty": quality_entry, "exit_penalty": quality_exit, "interior_penalty": interior_badness,
               "entry": entry_desc["quality"], "exit": exit_desc["quality"]}
    warnings = ["segment crosses an estimated internal scene cut"] if crossing else []
    candidate = Candidate(clip_id, start, end, speed, out_start, count, entry_desc, exit_desc,
                          clip["motions"][i], clip["motions"][j], quality, pts[entry], pts[exit_], warnings=warnings)
    update_cost(candidate, cfg)
    return candidate


def update_cost(candidate, cfg):
    q = candidate.quality
    candidate.segment_cost = cfg.weights["quality"] * (.4*q["entry_penalty"] + .4*q["exit_penalty"] + .2*q["interior_penalty"])
    candidate.segment_cost += cfg.weights["speed"] * abs(math.log(candidate.speed))
    candidate.segment_cost += 1.0 if candidate.warnings else 0.


def prune_diverse(candidates, limit, rng):
    """Round-robin clip groups, retaining quality and region/speed diversity."""
    groups = defaultdict(list)
    for c in candidates:
        groups[c.clip].append(c)
    per_clip_limit = min(limit, max(3, math.ceil(limit/max(1, len(groups)))*3))
    for items in groups.values():
        # Stable seed jitter breaks exact quality ties without hard-coded ordering.
        noise = {id(c): rng.random()*1e-5 for c in items}
        items.sort(key=lambda c: c.segment_cost + noise[id(c)])
        # Retain first then spread starts rather than many neighboring trims.
        if len(items) > 2:
            chosen, rest = [items[0]], items[1:]
            while rest and len(chosen) < per_clip_limit:
                best = min(rest, key=lambda c: c.segment_cost + .05/(.01 + min(abs(c.source_start-s.source_start) for s in chosen)))
                chosen.append(best)
                rest.remove(best)
            groups[items[0].clip] = chosen
    keys = list(groups)
    rng.shuffle(keys)
    keys.sort(key=lambda k: groups[k][0].segment_cost)
    result = []
    depth = 0
    while len(result) < limit:
        added = False
        for key in keys:
            if depth < len(groups[key]):
                result.append(groups[key][depth])
                added = True
                if len(result) == limit:
                    break
        if not added:
            break
        depth += 1
    return result


def generate_candidates(clips, boundaries, cfg):
    rng = np.random.default_rng(cfg.seed)
    pools, warnings = [], []
    speeds = sorted(set([cfg.speed_min, min(cfg.speed_max, max(cfg.speed_min, 1.)), cfg.speed_max]))
    for slot, (a, b) in enumerate(zip(boundaries, boundaries[1:])):
        generated = []
        for relaxed in (False, True):
            for clip_id, clip in enumerate(clips):
                usable = [(0., clip["duration"])] if relaxed else regions(clip)
                for lo, hi in usable:
                    for speed in speeds:
                        consumed = (b-a) / cfg.fps * speed
                        if hi-lo < consumed-1e-8:
                            continue
                        starts = [t for t in clip["times"] if lo-1e-8 <= t <= hi-consumed+1e-8]
                        # Include the far end; use an actual presentation timestamp.
                        end_index = max(0, bisect_left(clip["pts"], hi-consumed+1e-8)-1)
                        if clip["pts"][end_index] >= lo-1e-8:
                            starts.append(clip["pts"][end_index])
                        for start in sorted(set(starts)):
                            c = make_candidate(clip_id, clip, start, speed, a, b-a, cfg, allow_cuts=relaxed)
                            if c:
                                generated.append(c)
            if generated:
                if relaxed:
                    warnings.append(f"slot {slot}: internal source-cut avoidance relaxed; no continuous region filled this slot")
                break
        if not generated:
            raise MediaError(f"no source segment can fill slot {slot} ({(b-a)/cfg.fps:.3f}s) at allowed speeds")
        pools.append(prune_diverse(generated, cfg.candidates_per_slot, rng))
    return pools, warnings


def refine_pools(clips, pools, paths, cfg):
    """Refine source trims of several complete coarse paths, preserving output cuts.

    Native source frames (not nominal FPS) are decoded in one batch per clip.
    Farneback uses only sequences inside each refined source interval.
    """
    selected = [set() for _ in pools]
    for path in paths[:cfg.refine_top]:
        for slot, candidate in enumerate(path):
            selected[slot].add(id(candidate))
    result = [[] for _ in pools]
    for slot, pool in enumerate(pools):
        for candidate in pool:
            if id(candidate) not in selected[slot]:
                continue
            clip = clips[candidate.clip]
            shifts = np.linspace(-.5/cfg.samples_per_second, .5/cfg.samples_per_second, cfg.refine_steps)
            starts = {candidate.source_start}
            for shift in shifts:
                i = bisect_left(clip["pts"], candidate.source_start + float(shift))
                if i < len(clip["pts"]):
                    starts.add(clip["pts"][i])
            for start in sorted(starts):
                c = make_candidate(candidate.clip, clip, start, candidate.speed, candidate.output_start_frame,
                                   candidate.output_frame_count, cfg, bool(candidate.warnings))
                if c:
                    result[slot].append(c)
    needed = defaultdict(set)
    sequences = {}
    for pool in result:
        for c in pool:
            clip = clips[c.clip]
            first = visible_source_index(clip["pts"], c.source_start, 0, c.speed, cfg.fps)
            last = visible_source_index(clip["pts"], c.source_start, c.output_frame_count-1, c.speed, cfg.fps)
            end_index = bisect_left(clip["pts"], c.source_end-1e-9)
            incoming = list(range(first, min(first+3, end_index)))
            outgoing = list(range(max(first, last-2), last+1))
            sequences[id(c)] = incoming, outgoing
            needed[c.clip].update(incoming + outgoing)
    decoded = {}
    w, h = analysis_dimensions(cfg)
    for clip_id, indices in needed.items():
        indices = sorted(indices)
        # Group contiguous native indices to keep filter expressions short.
        groups = []
        for i in indices:
            if groups and groups[-1][1]+1 == i:
                groups[-1][1] = i
            else:
                groups.append([i, i])
        select = "+".join(f"between(n,{a},{b})" for a, b in groups)
        vf = f"select='{select}'," + fit_filter(w, h, cfg.fit)
        raw = run(["ffmpeg", "-v", "error", "-threads", cfg.threads, "-i", clips[clip_id]["identity"]["path"],
                   "-an", "-vf", vf, "-fps_mode", "passthrough", "-threads", cfg.threads,
                   "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], binary=True)
        if len(raw) != len(indices)*w*h*3:
            raise MediaError("native frame refinement decode did not return the expected frames")
        frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
        decoded[clip_id] = {i: frame for i, frame in zip(indices, frames)}
    for pool in result:
        for c in pool:
            incoming, outgoing = sequences[id(c)]
            data = decoded[c.clip]
            clip = clips[c.clip]
            c.entry = composition_descriptor(data[incoming[0]], clip["probe"], cfg)
            c.exit = composition_descriptor(data[outgoing[-1]], clip["probe"], cfg)
            def measure(indices):
                if any(clip["pts"][indices[0]] <= b and clip["pts"][indices[-1]] >= a for a, b in clip["scene_boundaries"]):
                    return {"vx": 0., "vy": 0., "magnitude": 0., "reliability": 0.}
                return motion_summary([data[i] for i in indices], [clip["pts"][i] for i in indices])
            c.entry_motion = measure(incoming)
            c.exit_motion = measure(outgoing)
            c.quality.update(entry_penalty=boundary_quality(c.entry, clip["reference_sharpness"]),
                             exit_penalty=boundary_quality(c.exit, clip["reference_sharpness"]),
                             entry=c.entry["quality"], exit=c.exit["quality"])
            c.refined = True
            update_cost(c, cfg)
    return result
