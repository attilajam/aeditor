"""Analysis, coarse search, trim refinement and reproducible plan construction."""
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

from . import __version__
from .audio import analyze_audio, choose_boundaries
from .candidates import generate_candidates, max_slot_duration, refine_pools
from .features import DESCRIPTOR_VERSION, EDGE_BINS, EDGE_GRIDS
from .planner import beam_search, path_details
from .probe import MediaError, require_tools, run
from .video import inspect_directory


def versions():
    import platform
    return {"autoedit": __version__, "python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "librosa", "opencv-python", "scipy", "PyYAML")},
            "ffmpeg": run(["ffmpeg", "-version"]).splitlines()[0],
            "ffprobe": run(["ffprobe", "-version"]).splitlines()[0]}


def cached_audio(path, cfg):
    return analyze_audio(path, cfg)


def analyze_inputs(audio_path, clips_path, cfg, exclude=()):
    require_tools()
    audio = cached_audio(audio_path, cfg)
    clips, skipped = inspect_directory(clips_path, cfg, exclude)
    return audio, clips, skipped


def make_plan(audio, clips, cfg, boundaries, path, warnings=(), skipped=()):
    details = path_details(path, cfg)
    w, h = cfg.dimensions
    all_warnings = list(warnings) + list(audio["warnings"])
    all_warnings += [warning for segment in path for warning in segment.warnings]
    if skipped:
        all_warnings.append(f"Skipped {len(skipped)} unsupported or corrupt input file(s); reasons recorded in skipped_sources.")
    for i, transition in enumerate(details["transitions"]):
        if transition["weighted"] > .30:
            all_warnings.append(f"transition {i+1} has a weak measured match ({transition['weighted']:.3f}); no perceptual quality guarantee")
    return {"version": 1, "tool": "autoedit", "versions": versions(), "configuration": cfg.to_dict(), "seed": cfg.seed,
            "audio": audio, "output": {"width": w, "height": h, "fps": cfg.fps, "total_frames": audio["total_frames"],
                                      "duration": audio["total_frames"]/cfg.fps, "fit": cfg.fit},
            "sources": [{"identity": c["identity"], "duration": c["duration"], "pts": c["pts"],
                         "probe": c["probe"], "scene_boundaries": c["scene_boundaries"],
                         "scene_detector_version": c.get("scene_detector_version", 1)} for c in clips],
            "matching": {"method": "spatial_unsigned_edge_pyramid", "descriptor_version": DESCRIPTOR_VERSION,
                         "grids": list(EDGE_GRIDS), "orientation_bins": EDGE_BINS,
                         "scale_weights": [.35, .65], "edge_center_bias": cfg.edge_center_bias,
                         "coordinates": "absolute normalized output frame; no subject alignment",
                         "distance": "center-weighted Hellinger; color and luminance are secondary cues"},
            "skipped_sources": list(skipped), "cut_frames": boundaries,
            "selected_cut_times": [frame/cfg.fps for frame in boundaries],
            "cut_origin": "detected_beats" if audio["fallback"] == "none" else audio["fallback"],
            "segments": [c.to_dict() for c in path], "scores": details,
            "warnings": list(dict.fromkeys(all_warnings)),
            "search": {"method": "coarse beam search, native-frame trim refinement, refined beam search",
                       "beam_width": cfg.beam_width, "candidates_per_slot": cfg.candidates_per_slot,
                       "refine_top": cfg.refine_top, "refine_steps": cfg.refine_steps}}


def build_plan(audio, clips, cfg, skipped=()):
    if cfg.reuse == "forbid" and sum(c["duration"] for c in clips)/cfg.speed_min < audio["duration"]-1e-8:
        raise MediaError("insufficient total footage with reuse forbidden at allowed playback speeds")
    cap = max_slot_duration(clips, cfg)
    relaxations = []
    try:
        boundaries, warnings = choose_boundaries(audio, cfg, max_duration=cap)
    except MediaError as exc:
        # Preserve the complete interval when the detected beat spacing exceeds
        # all continuous footage. Clearly label the replacement regular schedule.
        audio = {**audio, "fallback": "regular", "warnings": audio["warnings"] +
                 [f"Detected rhythm cannot fit available source regions: {exc}. Used regular footage-feasible pacing."]}
        boundaries, warnings = choose_boundaries(audio, cfg, max_duration=cap)
    pools, pool_warnings = generate_candidates(clips, boundaries, cfg)
    search_cfg = cfg
    try:
        states = beam_search(pools, search_cfg)
    except MediaError:
        # First broaden the bounded candidate pool; then reconsider beat grouping.
        search_cfg = replace(cfg, candidates_per_slot=cfg.candidates_per_slot*2, beam_width=cfg.beam_width*2)
        pools, pool_warnings = generate_candidates(clips, boundaries, search_cfg)
        try:
            states = beam_search(pools, search_cfg)
            relaxations.append("Search bounds doubled after the initial bounded search found no feasible complete path.")
        except MediaError:
            if cfg.reuse != "forbid":
                raise
            target = audio["duration"] / len(clips)
            grouping_cfg = replace(search_cfg, min_shot=min(cap, max(cfg.shot_limits[0], target)),
                                   max_shot=max(cfg.shot_limits[1], target*1.5))
            boundaries, warnings = choose_boundaries(audio, grouping_cfg, max_duration=cap)
            pools, pool_warnings = generate_candidates(clips, boundaries, search_cfg)
            states = beam_search(pools, search_cfg)
            relaxations.append("Beat grouping reconsidered for whole-edit reuse prohibition; pacing preferences relaxed.")
    refined = refine_pools(clips, pools, [state.path for state in states], search_cfg)
    final = beam_search(refined, search_cfg)[0]
    plan = make_plan(audio, clips, cfg, boundaries, final.path, warnings+pool_warnings+relaxations, skipped)
    plan["search"].update(beam_width=search_cfg.beam_width, candidates_per_slot=search_cfg.candidates_per_slot,
                         coarse_cost=states[0].cost, refined_cost=final.cost)
    return plan
