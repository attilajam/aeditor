"""Fixed-boundary planning and provenance checks for user-reviewed cuts."""
from copy import deepcopy
from dataclasses import replace

from .config import Config
from .pipeline import make_plan
from .probe import MediaError
from .validate import validate_user_timing

def build_review_plan(base_plan, cut_frames, *, adjustments=None, no_cut_ranges=None):
    """Replan source footage against fixed, user-reviewed output boundaries."""
    from .config import Config
    from .probe import verify_identity
    from .video import analyze_clip

    # Verify all recorded identities before any cached/repeated analysis.
    verify_identity(base_plan["audio"]["identity"])
    for source in base_plan["sources"]:
        verify_identity(source["identity"])
    try:
        cfg = Config(**base_plan["configuration"]).validate()
    except (KeyError, TypeError, ValueError) as exc:
        raise MediaError(f"invalid base plan configuration: {exc}") from exc
    upgrade_notes = []
    old_visual_defaults = {"color": .25, "structure": .45, "edge": .15, "motion": .15}
    if not base_plan.get("matching") and all(cfg.weights[key] == value for key, value in old_visual_defaults.items()):
        # Review runs a new selection. Upgrade the former automatic defaults,
        # retaining custom objectives and every user-reviewed output boundary.
        cfg.weights = {**cfg.weights, **{key: Config().weights[key] for key in old_visual_defaults}}
        upgrade_notes.append("Legacy automatic visual weights upgraded to spatial edge matching; user cut timing is unchanged.")
    boundaries = _frame_boundaries(cut_frames, base_plan["output"]["total_frames"])
    ranges = _validate_no_cut_ranges([] if no_cut_ranges is None else no_cut_ranges,
                                    base_plan["output"]["total_frames"])
    audio = deepcopy(base_plan["audio"])
    for start, end in ranges:
        offending = [frame for frame in boundaries[1:-1] if start <= frame < end]
        if offending:
            raise MediaError(f"fixed cut frame {offending[0]} lies inside no_cut_range [{start}, {end})")
    paths = [source["identity"]["path"] for source in base_plan["sources"]]
    clips = [analyze_clip(path, cfg) for path in paths]

    # User timing is fixed; only source choices are searched. The normal
    # candidate generator may fall back to crossing a scene cut, so reject
    # every such candidate and fail with its slot if no continuous pool exists.
    search_cfg = cfg
    if cfg.reuse == "forbid" and sum(clip["duration"] for clip in clips) / cfg.speed_min < audio["duration"] - 1e-8:
        raise MediaError("fixed review schedule cannot fit continuously with reuse forbidden at allowed playback speeds")
    from .candidates import generate_candidates, refine_pools
    from .planner import beam_search
    relaxation_notes = []
    for attempt in range(2):
        bounded_cfg = search_cfg if attempt == 0 else replace(
            search_cfg, candidates_per_slot=search_cfg.candidates_per_slot * 2,
            beam_width=search_cfg.beam_width * 2)
        try:
            proposed, _warnings = generate_candidates(clips, boundaries, bounded_cfg)
            clean = []
            for slot, pool in enumerate(proposed):
                accepted = [candidate for candidate in pool if not candidate.warnings]
                if not accepted:
                    raise MediaError(
                        f"fixed review schedule has no continuous source for slot {slot} "
                        f"(frames {boundaries[slot]}–{boundaries[slot+1]}); "
                        "all candidates cross an estimated internal source cut or no continuous region fits"
                    )
                clean.append(accepted)
            states = beam_search(clean, bounded_cfg)
            pools, search_cfg = clean, bounded_cfg
            if attempt:
                relaxation_notes.append("Candidate and beam search bounds doubled once after no feasible fixed-schedule path.")
            break
        except MediaError as exc:
            if attempt:
                raise MediaError(f"fixed review schedule is infeasible after one bounded search widening: {exc}") from exc
            continue
    refined = refine_pools(clips, pools, [state.path for state in states], search_cfg)
    final = beam_search(refined, search_cfg)[0]
    plan = make_plan(audio, clips, cfg, boundaries, final.path,
                     ["User-reviewed cut timing is fixed; detected rhythm was preserved." , *upgrade_notes, *relaxation_notes],
                     base_plan.get("skipped_sources", ()))
    plan["configuration"] = search_cfg.to_dict()
    plan["seed"] = base_plan["seed"]
    plan["review_base_versions"] = deepcopy(base_plan.get("versions", {}))
    plan["cut_origin"] = "user_review"
    if adjustments is not None:
        recorded_adjustments = deepcopy(adjustments)
    else:
        beats = audio.get("beat_times", [])
        kick_frames = {round(float(t) * cfg.fps) for t, lab in zip(beats, audio.get("beat_labels", [])) if lab}
        recorded_adjustments = []
        for i, frame in enumerate(boundaries[1:-1]):
            original = min(base_plan["cut_frames"][1:-1], key=lambda old: abs(old-frame), default=frame)
            beat = min(beats, key=lambda value: abs(value*cfg.fps-original), default=None)
            onset = min(audio.get("onset_times", []), key=lambda value: abs(value*cfg.fps-original), default=None)
            if audio.get("fallback") == "none" and beat is not None and round(beat*cfg.fps) == original:
                kind, detected_time = ("kick", beat) if original in kick_frames else ("beat", beat)
            elif audio.get("fallback") == "onsets" and onset is not None and round(onset*cfg.fps) == original:
                kind, detected_time = "onset", onset
            elif original in base_plan["cut_frames"][1:-1]:
                kind, detected_time = "automatic", None
            else:
                kind, detected_time = "manual", None
            recorded_adjustments.append({"id": f"review-{i}",
                "kind": kind, "detected_time": detected_time,
                "original_frame": original, "frame": frame, "enabled": True})
    plan["user_timing"] = {
        "version": 1,
        "base_cut_frames": list(base_plan["cut_frames"]),
        "adjustments": recorded_adjustments,
        "no_cut_ranges": ranges,
        "cut_frames": boundaries,
    }
    # The marker list is the source of truth: reject stale UI calculations.
    timing_report = validate_user_timing(plan)
    if not timing_report["valid"]:
        raise MediaError("invalid user timing: " + "; ".join(timing_report["errors"]))
    return plan


def _frame_boundaries(values, total):
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        raise MediaError("cut_frames must contain exact integer boundaries including 0 and total_frames")
    if any(type(value) is not int for value in values):
        raise MediaError("cut_frames must contain exact integer frame boundaries")
    if values[0] != 0 or values[-1] != total or any(b <= a for a, b in zip(values, values[1:])):
        raise MediaError("cut_frames must be strictly increasing from 0 through total_frames")
    return list(values)


def _validate_no_cut_ranges(values, total):
    if not isinstance(values, (list, tuple)):
        raise MediaError("no_cut_ranges must be a list of [start_frame, end_frame] pairs")
    result = []
    for item in values:
        if (not isinstance(item, (list, tuple)) or len(item) != 2 or
                any(type(x) is not int for x in item) or item[0] < 0 or item[1] > total or item[0] >= item[1]):
            raise MediaError("no_cut_ranges must contain valid integer [start_frame, end_frame) pairs")
        result.append(list(item))
    return result

