"""Validated, reproducible analysis and planning settings."""
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import math
import yaml


@dataclass
class Config:
    duration: float | None = None
    audio_start: float = 0.0
    pacing: str = "medium"
    size: str = "auto"
    fps: int = 30
    fit: str = "contain"
    speed_min: float = 0.85
    speed_max: float = 1.15
    reuse: str = "forbid"
    beat_sensitivity: float = 1.0
    kick_focus: bool = False
    quiet_threshold: float = 0.60
    quiet_window: float = 0.40
    quiet_release: float = 0.30
    seed: int = 0
    samples_per_second: float = 6.0
    analysis_long_side: int = 192
    edge_center_bias: float = 0.65
    min_shot: float | None = None
    max_shot: float | None = None
    scene_threshold: float = 0.30
    candidates_per_slot: int = 48
    beam_width: int = 24
    refine_top: int = 3
    refine_steps: int = 3
    cache_dir: str = ".autoedit-cache"
    crf: int = 19
    preset: str = "veryfast"
    threads: int = 2
    weights: dict = field(default_factory=lambda: {
        "color": 0.03, "structure": 0.07, "edge": 0.75, "motion": 0.15,
        "quality": 1.0, "repeat": 0.30, "overlap": 1.0,
        "speed": 0.08, "speed_change": 0.12, "duration": 0.10,
    })

    @property
    def dimensions(self):
        if self.size == "auto":
            raise ValueError("automatic output size must be resolved from input footage first")
        return tuple(map(int, self.size.lower().split("x")))

    @property
    def shot_limits(self):
        lo, hi = {"slow": (2., 4.), "medium": (1., 2.), "fast": (.5, 1.)}[self.pacing]
        factor = math.sqrt(self.beat_sensitivity)
        return self.min_shot or lo/factor, self.max_shot or hi/factor

    def validate(self):
        for key in ("fps", "analysis_long_side", "candidates_per_slot", "beam_width", "refine_top", "refine_steps", "threads", "seed"):
            if type(getattr(self, key)) is not int:
                raise ValueError(f"{key} must be an integer")
        if type(self.kick_focus) is not bool:
            raise ValueError("kick_focus must be a boolean")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        for key, value in asdict(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if not isinstance(self.size, str):
            raise ValueError("size must be auto or WIDTHxHEIGHT")
        self.size = self.size.lower()
        if self.size != "auto":
            try:
                w, h = self.dimensions
            except (ValueError, TypeError):
                raise ValueError("size must be auto or WIDTHxHEIGHT, for example 1920x1080") from None
            if w < 16 or h < 16 or w % 2 or h % 2:
                raise ValueError("output width and height must be even integers >=16")
        if self.pacing not in {"slow", "medium", "fast"} or self.fit not in {"contain", "cover", "stretch"}:
            raise ValueError("invalid pacing or fit")
        if self.reuse not in {"allow", "forbid"}:
            raise ValueError("reuse must be allow or forbid")
        if not .5 <= self.beat_sensitivity <= 3.:
            raise ValueError("beat_sensitivity must be between 0.5 and 3.0")
        if not 0 <= self.quiet_threshold <= 1:
            raise ValueError("quiet_threshold must be between 0 and 1; 0 disables the quiet gate")
        if not .05 <= self.quiet_window <= 5 or not 0 <= self.quiet_release <= 5:
            raise ValueError("quiet_window must be 0.05..5 seconds and quiet_release 0..5 seconds")
        if not 1 <= self.fps <= 120 or self.audio_start < 0:
            raise ValueError("fps must be 1..120 and audio_start nonnegative")
        if self.duration is not None and self.duration <= 0:
            raise ValueError("duration must be positive")
        if not 0 < self.speed_min <= self.speed_max <= 4:
            raise ValueError("require 0 < speed_min <= speed_max <= 4")
        if not .5 <= self.samples_per_second <= 60 or self.analysis_long_side < 48:
            raise ValueError("samples_per_second must be .5..60; analysis_long_side >=48")
        if not 0 <= self.edge_center_bias <= 1:
            raise ValueError("edge_center_bias must be between 0 and 1")
        if (self.min_shot is not None and self.min_shot <= 0) or (self.max_shot is not None and self.max_shot <= 0):
            raise ValueError("shot limits must be positive")
        lo, hi = self.shot_limits
        if lo > hi or hi * self.fps < 1:
            raise ValueError("min_shot must not exceed max_shot")
        if not 0 < self.scene_threshold <= 1:
            raise ValueError("scene_threshold must be in (0,1]")
        if min(self.candidates_per_slot, self.beam_width, self.refine_top, self.refine_steps, self.threads) < 1:
            raise ValueError("candidate, beam, refinement and thread limits must be positive")
        if not 0 <= self.crf <= 51 or self.preset not in {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"}:
            raise ValueError("invalid encoder preset or crf")
        expected = Config().weights
        if set(self.weights) != set(expected) or any(not math.isfinite(v) or v < 0 for v in self.weights.values()):
            raise ValueError("weights must contain all documented nonnegative finite components")
        return self

    def to_dict(self):
        return asdict(self)


def load_config(path=None, overrides=None):
    data = {}
    if path:
        data = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError("configuration must be a YAML mapping")
    unknown = set(data) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"unknown configuration keys: {', '.join(sorted(unknown))}")
    if "weights" in data:
        data["weights"] = {**Config().weights, **data["weights"]}
    data.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return Config(**data).validate()
