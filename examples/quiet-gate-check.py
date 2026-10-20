"""Reproduce measurements and an opening contact sheet for the supplied edit."""
import json
from pathlib import Path

import cv2
import numpy as np

from autoedit.validate import decode_frames


root = Path(__file__).resolve().parents[1]
plan = json.loads((root / "edit.plan.json").read_text())
validation = json.loads((root / "edit.validation.json").read_text())
previous = json.loads((root / "examples/before-quiet-gate/edit.plan.json").read_text())
first = plan["cut_frames"][1]
fps = plan["output"]["fps"]
segments = plan["segments"]
activity = plan["audio"]["cut_activity"]
assert validation["valid"], validation["errors"]
assert plan["audio"]["beat_times"] == previous["audio"]["beat_times"]
assert plan["cut_frames"][1:] == [f for f in previous["cut_frames"] if f >= first]
assert len(segments) == len({s["clip"] for s in segments})
assert len(segments) == len({plan["sources"][s["clip"]]["identity"]["sha256"] for s in segments})
assert not activity["forced_cut_frames"]
source = plan["sources"][segments[0]["clip"]]
assert not any(segments[0]["source_start"] < b and segments[0]["source_end"] > a
               for a, b in source["scene_boundaries"])

frames = decode_frames(root / "edit.mp4", (192, 108))
changes = np.abs(np.diff(frames[:first].astype(np.float32), axis=0)).mean(axis=(1, 2, 3))/255
opening = {"source": source["identity"]["path"],
           "source_start": segments[0]["source_start"], "source_end": segments[0]["source_end"],
           "speed": segments[0]["speed"], "detected_internal_cuts": source["scene_boundaries"],
           "max_adjacent_rgb_mae": float(changes.max()),
           "max_change_frame": int(np.argmax(changes)+1)}
summary = {"output": plan["output"], "shots": len(segments), "distinct_sources": len(segments),
           "first_cut_frame": first, "first_cut_seconds": first/fps,
           "beat_sensitivity": plan["configuration"]["beat_sensitivity"],
           "beat_grid_unchanged": True, "post_drop_cut_grid_unchanged": True,
           "cut_activity": activity, "opening": opening,
           "max_beat_error_seconds": validation["max_beat_error_seconds"],
           "audio_alignment": validation["audio_alignment"],
           "max_boundary_pixel_mae": max(c["normalized_pixel_mae"] for c in validation["boundary_checks"]),
           "validation_valid": validation["valid"], "warnings": plan["warnings"]}
(root / "examples/quiet-gate-summary.json").write_text(json.dumps(summary, indent=2)+"\n")
indices = [0, 2*fps, 4*fps, 6*fps, first-1, first]
tiles = []
for i in indices:
    tile = cv2.resize(frames[i], (384, 216))
    tile = cv2.copyMakeBorder(tile, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
    cv2.putText(tile, f"{i/fps:.3f}s | frame {i}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                .5, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(tile)
contact = np.concatenate([np.concatenate(tiles[:3], axis=1), np.concatenate(tiles[3:], axis=1)], axis=0)
cv2.imwrite(str(root / "examples/quiet-gate-opening.png"), contact)
print(json.dumps(summary, indent=2))
