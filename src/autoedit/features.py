"""Compact, JSON-safe visual and motion descriptors for normalized video frames.

The primary match descriptor is a pyramid of unsigned gradient orientations
in fixed screen coordinates. Palette and luminance are secondary cues. Legacy
12x12 maps remain available for scene detection and older descriptors.
"""

from __future__ import annotations

import base64
from functools import lru_cache
from typing import Any

import cv2
import numpy as np


_GRID = 12
DESCRIPTOR_VERSION = 5
EDGE_GRIDS = (8, 16)
EDGE_BINS = 8


def _oriented_edges(gray: np.ndarray) -> list[dict[str, Any]]:
    """Contrast-polarity independent contours without centering the subject.

    Smooth pixel noise, interpolate each gradient between adjacent angle bins,
    then pool at two spatial scales. Cap unusually dense cells to keep a small
    patch of background texture from dominating. Square-root unit-mass
    histograms permit a bounded Hellinger comparison, including edge location.
    """
    smooth = cv2.GaussianBlur(gray, (0, 0), 0.8)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8)
    magnitude = np.maximum(cv2.magnitude(gx, gy) - 0.004, 0)
    angle = (np.arctan2(gy, gx) % np.pi) * (EDGE_BINS / np.pi)
    lower = np.floor(angle).astype(np.int32) % EDGE_BINS
    fraction = angle - np.floor(angle)
    channels = [magnitude * (np.where(lower == i, 1 - fraction, 0) +
                            np.where((lower + 1) % EDGE_BINS == i, fraction, 0))
                for i in range(EDGE_BINS)]
    pyramid = []
    for grid in EDGE_GRIDS:
        cells = np.stack([cv2.resize(channel, (grid, grid), interpolation=cv2.INTER_AREA)
                          for channel in channels], axis=-1)
        mass = cells.sum(axis=-1)
        nonzero = mass[mass > 1e-8]
        if nonzero.size:
            cap = float(np.percentile(nonzero, 75))
            cells *= np.minimum(1., cap / np.maximum(mass, 1e-8))[:, :, None]
        total = float(cells.sum())
        root = np.sqrt(cells / max(total, 1e-8))
        # Retaining thousands of Python floats per sampled frame multiplies
        # library memory use. Packed uint16 roots are JSON-safe and keep the
        # quantization error below 1/65535 per bin before renormalization.
        packed = np.rint(np.clip(root, 0, 1) * 65535).astype("<u2")
        pyramid.append({"grid": grid, "bins": EDGE_BINS, "encoding": "uint16-le-base64",
                        "data": base64.b64encode(packed.tobytes()).decode("ascii")})
    return pyramid


def decode_edge_level(level: dict) -> np.ndarray:
    """Unpack a normalized root histogram; accept the earlier list representation."""
    shape = (level["grid"], level["grid"], level["bins"])
    if level.get("encoding") == "uint16-le-base64":
        raw = base64.b64decode(level["data"], validate=True)
        expected = shape[0] * shape[1] * shape[2]
        if len(raw) != expected * 2:
            raise ValueError("invalid spatial edge descriptor length")
        return np.frombuffer(raw, dtype="<u2").reshape(shape).astype(np.float32) / 65535
    if "values" in level:
        return np.asarray(level["values"], dtype=np.float32).reshape(shape)
    raise ValueError("unsupported spatial edge descriptor encoding")


@lru_cache(maxsize=32)
def _center_weights(grid: int, center_bias: float) -> np.ndarray:
    axis = (np.arange(grid, dtype=np.float32) + .5) / grid - .5
    prior = (1 - center_bias) + center_bias * np.exp(-(axis[:, None]**2 + axis[None, :]**2) / (2 * .28**2))
    return np.sqrt(prior)[:, :, None]


def _gray01(frame_bgr: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame_bgr)
    if frame.ndim != 3 or frame.shape[2] < 3 or frame.size == 0:
        raise ValueError("frame_bgr must be a non-empty HxWx3 BGR image")
    if frame.dtype == np.uint8:
        return cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    arr = frame[:, :, :3].astype(np.float32)
    if np.nanmax(arr) > 1.0:
        arr /= 255.0
    return cv2.cvtColor(np.nan_to_num(arr, nan=0.0).clip(0, 1), cv2.COLOR_BGR2GRAY)


def describe(frame_bgr: np.ndarray, *, spatial: bool = True) -> dict[str, Any]:
    """Describe screen-space contours, color, structure and frame quality."""
    frame = np.asarray(frame_bgr)
    gray = _gray01(frame)
    # Histograms use HSV hue (8), saturation (4), value (2) bins, per quadrant.
    bgr8 = frame[:, :, :3]
    if bgr8.dtype != np.uint8:
        vals = bgr8.astype(np.float32)
        if np.nanmax(vals) <= 1.0:
            vals *= 255.0
        bgr8 = np.nan_to_num(vals, nan=0).clip(0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(bgr8, cv2.COLOR_BGR2HSV)
    h, w = gray.shape
    color: list[float] = []
    for y0, y1 in ((0, h // 2), (h // 2, h)):
        for x0, x1 in ((0, w // 2), (w // 2, w)):
            region = hsv[y0:y1, x0:x1]
            hist = cv2.calcHist([region], [0, 1, 2], None, [8, 4, 2], [0, 180, 0, 256, 0, 256])
            color.extend(hist.ravel().astype(np.float64).tolist())
    total = sum(color)
    if total:
        color = [v / total for v in color]

    structure = cv2.resize(gray, (_GRID, _GRID), interpolation=cv2.INTER_AREA).clip(0, 1)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8)
    edge_full = np.sqrt(gx * gx + gy * gy).clip(0, 1)
    edge = cv2.resize(edge_full, (_GRID, _GRID), interpolation=cv2.INTER_AREA).clip(0, 1)

    std = float(np.std(gray))
    black = float(np.mean(gray < 0.08))
    # Uniform means low contrast and low local gradient; soft footage can still
    # have normal contrast, so blur alone does not make this score high.
    uniform = float(np.clip((1.0 - min(std / 0.12, 1.0)) * (1.0 - min(float(edge_full.mean()) / 0.025, 1.0)), 0, 1))
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    sharpness = float(np.var(lap))
    texture = float(np.clip(0.65 * min(std / 0.16, 1.0) + 0.35 * min(float(edge_full.mean()) / 0.08, 1.0), 0, 1))
    result = {
        "color": color,
        "structure": structure.astype(float).ravel().tolist(),
        "edge": edge.astype(float).ravel().tolist(),
        "quality": {"black": black, "uniform": uniform, "sharpness": sharpness, "texture": texture},
    }
    if spatial:
        result["oriented_edges"] = _oriented_edges(gray)
        result["descriptor_version"] = DESCRIPTOR_VERSION
    return result


def motion_summary(frames: list[np.ndarray], times: list[float]) -> dict[str, float]:
    """Estimate robust mean flow velocity for one clip, normalized by diagonal."""
    if len(frames) != len(times):
        raise ValueError("frames and times must have equal lengths")
    if len(frames) < 2:
        return {"vx": 0.0, "vy": 0.0, "magnitude": 0.0, "reliability": 0.0}
    h, w = frames[0].shape[:2]
    diag = max(float(np.hypot(w, h)), 1.0)
    velocities: list[tuple[float, float, float]] = []
    texture_scores = []
    for first, second, t0, t1 in zip(frames, frames[1:], times, times[1:]):
        dt = float(t1) - float(t0)
        if dt <= 0 or first.shape[:2] != (h, w) or second.shape[:2] != (h, w):
            continue
        a, b = _gray01(first), _gray01(second)
        # Keep flow affordable while preserving enough detail for textured footage.
        scale = min(1.0, 320.0 / max(h, w))
        size = (max(16, round(w * scale)), max(16, round(h * scale)))
        aa = np.rint(cv2.resize(a, size, interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)
        bb = np.rint(cv2.resize(b, size, interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)
        flow = cv2.calcOpticalFlowFarneback(aa, bb, None, 0.5, 3, 21, 3, 5, 1.2, 0)
        # Prefer textured pixels so large blank backgrounds do not overwhelm
        # small subjects or camera motion. Sparse edges imply low confidence.
        aa_f, bb_f = aa.astype(np.float32) / 255.0, bb.astype(np.float32) / 255.0
        grad_a = cv2.magnitude(cv2.Sobel(aa_f, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8),
                               cv2.Sobel(aa_f, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8))
        grad_b = cv2.magnitude(cv2.Sobel(bb_f, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8),
                               cv2.Sobel(bb_f, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8))
        mask = np.maximum(grad_a, grad_b) > 0.025
        if np.count_nonzero(mask) < 8:
            velocities.append((0.0, 0.0, dt))
            texture_scores.append(0.0)
            continue
        # Median rejects independent moving objects and occasional outliers.
        fx, fy = np.median(flow[mask], axis=0)
        size_diag = float(np.hypot(size[0], size[1]))
        vx, vy = float(fx / size_diag / dt), float(fy / size_diag / dt)
        velocities.append((vx, vy, dt))
        # A split-color blank frame has contrast but almost no textured area.
        texture_scores.append(float(np.clip(np.mean(mask) / 0.2, 0, 1)))
    if not velocities:
        return {"vx": 0.0, "vy": 0.0, "magnitude": 0.0, "reliability": 0.0}
    total_dt = sum(v[2] for v in velocities)
    vx = sum(v[0] * v[2] for v in velocities) / total_dt
    vy = sum(v[1] * v[2] for v in velocities) / total_dt
    texture = float(np.mean(texture_scores))
    # Texture makes both moving and stationary estimates meaningful.
    reliability = float(np.clip(texture, 0, 1))
    return {"vx": float(vx), "vy": float(vy), "magnitude": float(np.hypot(vx, vy)), "reliability": reliability}


def _motion_vector(m: dict[str, Any], speed: float) -> tuple[float, float, float]:
    vx, vy = float(m.get("vx", 0.0)) * speed, float(m.get("vy", 0.0)) * speed
    return vx, vy, float(np.hypot(vx, vy))


def motion_distance(a: dict[str, Any], b: dict[str, Any], speed_a: float = 1.0, speed_b: float = 1.0) -> float:
    """Bounded discrepancy blending uncertain flow toward a neutral 0.5."""
    ax, ay, am = _motion_vector(a, speed_a)
    bx, by, bm = _motion_vector(b, speed_b)
    # Saturating scale makes very fast motion differences remain bounded.
    mag_a, mag_b = np.tanh(am / 0.35), np.tanh(bm / 0.35)
    magnitude = abs(mag_a - mag_b)
    if am < 1e-4 and bm < 1e-4:
        direction = 0.0
    elif am < 1e-4 or bm < 1e-4:
        direction = 0.0  # stationarity is captured by the magnitude term
    else:
        cosine = np.clip((ax * bx + ay * by) / (am * bm), -1, 1)
        # Direction remains informative for comparable nonzero motion, even
        # when both speeds are modest; magnitude separately captures stationarity.
        direction = float((1.0 - cosine) * 0.5)
    certain_distance = float(np.clip(0.3 * magnitude + 0.7 * direction, 0, 1))
    ra = float(np.clip(a.get("reliability", 0.0), 0, 1))
    rb = float(np.clip(b.get("reliability", 0.0), 0, 1))
    confidence = ra * rb
    return float(np.clip(confidence * certain_distance + (1 - confidence) * 0.5, 0, 1))


def legacy_visual_distance(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    """Original palette/luminance/magnitude distances; also used for cut detection."""
    out: dict[str, float] = {}
    for key in ("color", "structure", "edge"):
        x, y = np.asarray(a.get(key, []), dtype=np.float64), np.asarray(b.get(key, []), dtype=np.float64)
        if x.size == 0 or x.shape != y.shape:
            out[key] = 0.0 if x.size == y.size == 0 else 1.0
        elif key == "color":
            # L1 distance between unit-mass histograms has maximum 2.
            out[key] = float(np.clip(np.abs(x - y).sum() / 2.0, 0, 1))
        else:
            out[key] = float(np.clip(np.abs(x - y).mean(), 0, 1))
    return out


def spatial_edge_distance(a: list[dict], b: list[dict], center_bias: float = .65) -> float:
    """Weighted Hellinger distance in [0,1], 35% coarse / 65% fine.

    Broad Gaussian emphasis favors a central subject while retaining every
    cell. No translation, scaling, reflection or pose alignment is applied:
    subject position and size must match in the rendered frame itself.
    """
    if len(a) != len(EDGE_GRIDS) or len(b) != len(EDGE_GRIDS):
        raise ValueError("incompatible spatial edge pyramid")
    distances = []
    for left, right, grid in zip(a, b, EDGE_GRIDS):
        if any(level.get("grid") != grid or level.get("bins") != EDGE_BINS for level in (left, right)):
            raise ValueError("incompatible spatial edge grid or orientation bins")
        x, y = decode_edge_level(left), decode_edge_level(right)
        weight = _center_weights(grid, center_bias)
        x, y = x * weight, y * weight
        nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
        if nx < 1e-7 or ny < 1e-7:
            distance = 0. if nx < 1e-7 and ny < 1e-7 else 1.
        else:
            # Difference form avoids cancellation near a perfect match.
            distance = float(np.sqrt(np.clip(.5 * np.sum((x/nx-y/ny)**2), 0, 1)))
        distances.append(distance)
    return .35 * distances[0] + .65 * distances[1]


def visual_distance(a: dict[str, Any], b: dict[str, Any], center_bias: float = .65) -> dict[str, float]:
    """Compare normalized cues; edge uses contours when both descriptors have them.

    Old plans and hand-authored legacy descriptors retain their original edge
    magnitude comparison. Mixing descriptor generations is an error, rather
    than quietly discarding the stronger cue.
    """
    out = legacy_visual_distance(a, b)
    if ("oriented_edges" in a) != ("oriented_edges" in b):
        raise ValueError("mixed visual descriptor generations; reanalyze the footage")
    if "oriented_edges" in a:
        out["edge"] = spatial_edge_distance(a["oriented_edges"], b["oriented_edges"], center_bias)
    return out


def boundary_quality(desc: dict[str, Any], reference_sharpness: float = 0.0) -> float:
    """Score a boundary frame; black/uniform frames are strongly penalized.

    Sharpness is judged relative to the clip reference so a consistently soft
    clip does not receive a blur penalty.
    """
    q = desc.get("quality", {})
    black = float(np.clip(q.get("black", 0), 0, 1))
    uniform = float(np.clip(q.get("uniform", 0), 0, 1))
    sharpness = max(0.0, float(q.get("sharpness", 0)))
    relative = 1.0 if reference_sharpness <= 1e-9 else float(np.clip(sharpness / reference_sharpness, 0, 1))
    if black >= .98:
        return 4.0
    if uniform >= .95:
        return 3.0
    goodness = (1.0 - black) ** 2 * (1.0 - 0.85 * uniform) * (0.7 + 0.3 * relative)
    return float(np.clip(1.0 - goodness, 0, 1))
