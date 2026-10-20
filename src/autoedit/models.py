"""Candidate representation shared by analysis, planning, reporting and replay."""
from dataclasses import asdict, dataclass, field


@dataclass
class Candidate:
    clip: int
    source_start: float
    source_end: float
    speed: float
    output_start_frame: int
    output_frame_count: int
    entry: dict
    exit: dict
    entry_motion: dict
    exit_motion: dict
    quality: dict
    entry_pts: float
    exit_pts: float
    segment_cost: float = 0.
    refined: bool = False
    warnings: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)
