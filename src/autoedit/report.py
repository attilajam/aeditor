"""A self-contained inspection report using actual rendered boundary frames."""
from pathlib import Path
import base64
import html
import cv2

from .validate import decode_frames


def _thumbnail(frame):
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError("could not encode boundary thumbnail")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode()


def write_report(plan, video, validation, output):
    esc = html.escape
    dims = (216, 384) if plan["output"]["height"] > plan["output"]["width"] else (384, 216)
    cuts = plan["cut_frames"][1:-1]
    indices = sorted({f for cut in cuts for f in (cut-1, cut)})
    frames = decode_frames(video, dims, indices) if indices else {}
    fps = plan["output"]["fps"]
    matching = plan.get("matching", {})
    matching_summary = "Visual scores use the matcher and weights recorded in this plan."
    if matching.get("method") == "spatial_unsigned_edge_pyramid":
        weights = plan["configuration"]["weights"]
        matching_summary = ("Subject composition matching: unsigned contour orientations on 8×8 and 16×16 screen grids. "
                            "Position and scale are retained; this is an approximate pose cue, not skeleton detection. "
                            f"Weights: contours {weights['edge']:.2f}, color {weights['color']:.2f}, "
                            f"brightness {weights['structure']:.2f}, motion {weights['motion']:.2f}. "
                            f"Speed-change weight {weights['speed_change']:.2f}. "
                            f"Broad center emphasis {matching['edge_center_bias']:.2f} (0 gives equal screen weighting).")
    timeline, rows, pairs = [], [], []
    colors = ["#456b9c", "#7e548d", "#467d70", "#947137", "#8a5359", "#4e7480"]
    for i, segment in enumerate(plan["segments"]):
        name = Path(plan["sources"][segment["clip"]]["identity"]["path"]).name
        count = segment["output_frame_count"]
        timeline.append(f'<a href="#slot-{i}" style="flex:{count};background:{colors[segment["clip"]%len(colors)]}" title="{esc(name)}">{i+1}</a>')
        rows.append(f'<tr id="slot-{i}"><td>{i+1}</td><td>{segment["output_start_frame"]/fps:.3f}s</td><td>{count/fps:.3f}s / {count}f</td><td>{esc(name)}</td><td>{segment["source_start"]:.4f}–{segment["source_end"]:.4f}</td><td>{segment["speed"]:.3f}×</td><td>{segment["segment_cost"]:.3f}</td></tr>')
    for i, cut in enumerate(cuts):
        score = plan["scores"]["transitions"][i]
        labels = {"edge": "contours" if matching.get("method") == "spatial_unsigned_edge_pyramid" else "edge",
                  "structure": "brightness"}
        parts = " · ".join(f"{labels.get(key, key)} {score[key]:.3f}" for key in ("color", "structure", "edge", "motion"))
        pairs.append(f'<article><h3>Cut {i+1} · {cut/fps:.3f}s · frame {cut}</h3><div class="pair"><figure><img src="{_thumbnail(frames[cut-1])}"><figcaption>Outgoing · frame {cut-1}</figcaption></figure><figure><img src="{_thumbnail(frames[cut])}"><figcaption>Incoming · frame {cut}</figcaption></figure></div><p>{parts}<br>Weighted transition: {score["weighted"]:.3f} (lower is better)</p></article>')
    warnings = "".join(f"<li>{esc(w)}</li>" for w in plan["warnings"])
    alignment = validation["audio_alignment"]
    activity = plan["audio"].get("cut_activity", {})
    if activity.get("enabled"):
        spans = ", ".join(f"{a:.3f}–{b:.3f}s" for a, b in activity.get("quiet_intervals", [])) or "none"
        quiet_summary = (f"Protected quiet sections: {spans}. Relative RMS threshold {activity['relative_threshold']:.2f}; "
                         f"{len(activity.get('suppressed_event_frames', []))} cut opportunities suppressed; "
                         f"{len(activity.get('forced_cut_frames', []))} cuts required by footage limits.")
    else:
        quiet_summary = "Quiet-section gating is disabled or was not recorded in this plan."
    timing_summary = ""
    beat_label = "maximum selected beat quantization error"
    if plan.get("user_timing"):
        count = len(validation.get("manual_adjustments", []))
        ranges = plan["user_timing"].get("no_cut_ranges", [])
        timing_summary = (f"<p>User-reviewed timeline: {count} manually placed or adjusted cut(s); "
                          f"{len(ranges)} strict no-cut range(s). Original detected beats remain in the plan. "
                          "Explicit user choices take precedence over automatic quiet gating.</p>")
        beat_label = "maximum untouched detected-beat quantization error"
    body = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Autoedit inspection</title>
<style>body{{font:15px system-ui;color:#e7eaf0;background:#171b23;margin:24px auto;max-width:1100px;padding:0 16px}}h1,h2,h3{{font-weight:600}}a{{color:inherit}}.timeline{{display:flex;height:42px;gap:2px;margin:24px 0}}.timeline a{{text-align:center;padding-top:10px;min-width:12px;text-decoration:none}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border-bottom:1px solid #414651;text-align:left;padding:8px}}.scroll{{overflow-x:auto}}.pairs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}}article{{background:#252c38;border-radius:8px;padding:16px}}.pair{{display:flex;gap:10px}}figure{{margin:0;width:50%}}img{{width:100%;display:block}}figcaption{{font-size:12px;color:#b8bfcb;margin-top:6px}}p{{line-height:1.55}}video{{max-width:100%;max-height:540px}}</style>
<h1>Autoedit inspection</h1><p>{plan['output']['duration']:.3f}s · {len(plan['segments'])} shots · {plan['output']['total_frames']} frames · {plan['output']['width']}×{plan['output']['height']} at {fps} fps</p>
<p>Rhythm: {esc(plan['cut_origin'])}. Reliability heuristic {plan['audio']['reliability']['score']:.3f}; this is not a calibrated probability. Seed {plan['seed']}.</p>
<p>{esc(quiet_summary)}</p>
{timing_summary}
<p>Validation: {'passed' if validation['valid'] else 'FAILED'} · fully decoded {validation['decoded_frames']} frames · {beat_label} {validation['max_beat_error_seconds']*1000:.2f} ms · soundtrack lag {alignment.get('lag_seconds',0)*1000:.2f} ms.</p>
<div class="timeline">{''.join(timeline)}</div><div class="scroll"><table><tr><th>Shot</th><th>Output start</th><th>Length</th><th>Source</th><th>Source interval (s)</th><th>Speed</th><th>Segment cost</th></tr>{''.join(rows)}</table></div>
<h2>Actual rendered boundary frames</h2><p>{esc(matching_summary)}</p><p>Each pair is decoded directly from the finished MP4. Optical flow is measured within each source shot; no flow is estimated across these cuts.</p><div class="pairs">{''.join(pairs) or '<p>Single-shot edit: no internal transitions.</p>'}</div>
<h2>Warnings and relaxations</h2><ul>{warnings or '<li>None</li>'}</ul><p>Skipped sources: {len(plan['skipped_sources'])}. Full source fingerprints, analysis, cost components and settings are in the adjacent JSON plan.</p></html>'''
    Path(output).write_text(body)
    return body
