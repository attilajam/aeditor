"""Bounded beam search with complete-path source usage constraints."""
from collections import Counter
from dataclasses import dataclass
import math
import numpy as np

from .features import motion_distance, visual_distance
from .probe import MediaError


def transition_components(previous, current, cfg):
    parts = visual_distance(previous.exit, current.entry, cfg.edge_center_bias)
    parts["motion"] = motion_distance(previous.exit_motion, current.entry_motion, previous.speed, current.speed)
    parts["speed_change"] = min(1., abs(math.log(current.speed/previous.speed)))
    parts["weighted"] = sum(cfg.weights[k]*value for k, value in parts.items())
    return parts


def usage_cost(path, candidate, cfg):
    previous = [c for c in path if c.clip == candidate.clip]
    if cfg.reuse == "forbid" and previous:
        return None
    repeat = len(previous) * cfg.weights["repeat"]
    overlap = sum(max(0., min(c.source_end, candidate.source_end)-max(c.source_start, candidate.source_start)) /
                  max(1e-8, candidate.source_end-candidate.source_start) for c in previous)
    return repeat + cfg.weights["overlap"] * overlap


@dataclass
class SearchState:
    slot: int
    cost: float
    path: tuple
    usage: tuple


def beam_search(pools, cfg, keep=None):
    states = [SearchState(0, 0., (), ())]
    transition_cache = {}
    for slot, pool in enumerate(pools):
        expanded = []
        for state in states:
            for c in pool:
                usage = usage_cost(state.path, c, cfg)
                if usage is None:
                    continue
                transition = 0.
                if state.path:
                    key = (id(state.path[-1]), id(c))
                    if key not in transition_cache:
                        transition_cache[key] = transition_components(state.path[-1], c, cfg)["weighted"]
                    transition = transition_cache[key]
                counts = dict(state.usage)
                counts[c.clip] = counts.get(c.clip, 0) + 1
                expanded.append(SearchState(slot+1, state.cost+c.segment_cost+transition+usage,
                                            state.path+(c,), tuple(sorted(counts.items()))))
        if not expanded:
            raise MediaError(f"beam search found no complete feasible path at slot {slot}; reuse={cfg.reuse}. "
                             "Try shorter duration, more footage, reuse allow, or a wider beam/candidate pool.")
        expanded.sort(key=lambda s: s.cost)
        # Equal ending candidates may have different histories. Preserve them when
        # usage differs, but don't fill the beam with near-identical source trims.
        states, signatures = [], Counter()
        for state in expanded:
            last = state.path[-1]
            signature = (last.clip, round(last.source_start*cfg.samples_per_second), state.usage)
            if signatures[signature] >= 2:
                continue
            signatures[signature] += 1
            states.append(state)
            if len(states) >= cfg.beam_width:
                break
    states.sort(key=lambda s: s.cost)
    return states[:keep or cfg.refine_top]


def path_details(path, cfg):
    transitions = []
    reuse = []
    total = 0.
    for i, c in enumerate(path):
        penalty = usage_cost(path[:i], c, cfg)
        total += c.segment_cost + (penalty or 0.)
        reuse.append({"slot": i, "previous_clip_uses": sum(p.clip == c.clip for p in path[:i]), "penalty": penalty})
        if i:
            parts = transition_components(path[i-1], c, cfg)
            parts["cut_frame"] = c.output_start_frame
            transitions.append(parts)
            total += parts["weighted"]
    return {"cost": total, "transitions": transitions, "reuse": reuse}


def random_feasible(pools, cfg):
    """Deterministic random baseline with the same candidates and usage rules."""
    rng = np.random.default_rng(cfg.seed)
    for _ in range(256):
        path = []
        for pool in pools:
            options = [c for c in pool if usage_cost(path, c, cfg) is not None]
            if not options:
                break
            path.append(options[int(rng.integers(len(options)))])
        if len(path) == len(pools):
            return path
    raise MediaError("no random feasible path found within 256 bounded attempts")
