"""Audio analysis and beat-aware output-frame boundary planning.

The beat score returned here is an analysis heuristic, not a calibrated
probability. All event times are relative to the decoded audio interval; the
interval's position in the source is recorded separately as ``audio_start``.
"""
from __future__ import annotations

import math
import subprocess
from fractions import Fraction
from pathlib import Path

import librosa
import numpy as np

from .config import Config
from .cache import key_for, read_cache, write_json
from .probe import MediaError, fingerprint, probe, require_tools


_SAMPLE_RATE = 22_050
_HOP_LENGTH = 512
_ANALYSIS_VERSION = 9
_MIN_BPM = 55.0
_MAX_BPM = 215.0
_LAST_INTERPOLATED: set[float] = set()


def _sensitivity_controls(cfg: Config) -> dict:
    """Return deterministic beat/onset controls for the configured sensitivity.

    Higher sensitivity maps to a denser beat grid: a lower DBN observation
    threshold admits metrical subdivisions the tracker would otherwise skip,
    and a lower transition lambda tolerates tempo drift. Onset peak-picking
    keeps the librosa controls: a finer onset grid, a lower local peak
    threshold, and a compressed onset envelope that gives quieter transients
    more weight.
    """
    sensitivity = float(getattr(cfg, "beat_sensitivity", 1.0))
    if not math.isfinite(sensitivity) or not 0.5 <= sensitivity <= 3.0:
        raise ValueError("beat_sensitivity must be 0.5..3")
    hop = int(np.clip(round(_HOP_LENGTH / math.sqrt(sensitivity)), 128, 1024))
    # Match librosa 1.0's onset_detect defaults in seconds, adjusted only for
    # the smaller hop so peak timing stays independent of frame resolution.
    peak_wait = max(1, int(round(.03 * _SAMPLE_RATE / hop / math.sqrt(sensitivity))))
    density = float(np.clip((sensitivity - .5) / 2.5, 0, 1))
    return {
        "sensitivity": sensitivity,
        "hop_length": hop,
        "onset_detection": {
            "pre_max": max(1, int(round(.03 * _SAMPLE_RATE / hop))),
            "post_max": max(1, int(round(.00 * _SAMPLE_RATE / hop)) + 1),
            "pre_avg": max(1, int(round(.10 * _SAMPLE_RATE / hop))),
            "post_avg": max(1, int(round(.10 * _SAMPLE_RATE / hop)) + 1),
            "wait": peak_wait,
            "delta": .07 / sensitivity,
        },
        "density": density,
        "beat_threshold": .20 - .08 * density,
        "transition_lambda": 100 - 40 * density,
        "analysis_version": _ANALYSIS_VERSION,
    }


_MADMOM: tuple | None | False = None


def _madmom_state() -> tuple | None | False:
    """Lazily load madmom and its RNN processors once; ``False`` means unusable."""
    global _MADMOM
    if _MADMOM is None:
        try:
            import warnings
            import madmom
            from madmom.audio.signal import Signal
            from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                beat_rnn = RNNBeatProcessor()
            _MADMOM = (Signal, beat_rnn, DBNBeatTrackingProcessor, madmom.__version__)
        except Exception:
            _MADMOM = False
    return _MADMOM


def _bass_envelope(samples: np.ndarray) -> np.ndarray:
    """Half-wave rectified spectral-diff energy of the 30-150 Hz band.

    The kick/bass transient is what a cut should land on for CapCut-style
    timing; onset envelopes weight snares and hats equally, so track the low
    band separately when deciding beat phase.
    """
    spec = np.abs(librosa.stft(samples, n_fft=2048, hop_length=256))
    freqs = np.linspace(0, _SAMPLE_RATE / 2, spec.shape[0])
    band = spec[(freqs >= 30) & (freqs <= 150)].sum(axis=0)
    diff = np.maximum(np.diff(band, prepend=band[0]), 0)
    peak = float(np.max(diff)) if diff.size else 0.0
    return diff / peak if peak > 1e-12 else diff


def _treble_envelope(samples: np.ndarray) -> np.ndarray:
    """Half-wave rectified spectral-diff energy of the 2-8 kHz band.

    The snare/hat crack lives up here while a kick does not, so the pair of
    bands is what tells one instrument's beat from another's.
    """
    spec = np.abs(librosa.stft(samples, n_fft=512, hop_length=64))
    freqs = np.linspace(0, _SAMPLE_RATE / 2, spec.shape[0])
    band = spec[(freqs >= 2000) & (freqs <= 8000)].sum(axis=0)
    diff = np.maximum(np.diff(band, prepend=band[0]), 0)
    peak = float(np.max(diff)) if diff.size else 0.0
    return diff / peak if peak > 1e-12 else diff


def _band_peak(env: np.ndarray, hop: int, center: float, half: float) -> float:
    """Strongest frame of a band envelope within +-half seconds of a time."""
    lo = int(max(0.0, center - half) * _SAMPLE_RATE / hop)
    hi = int((center + half) * _SAMPLE_RATE / hop) + 1
    seg = env[max(0, lo):min(len(env), hi)]
    return float(np.max(seg)) if seg.size else 0.0


# A beat reads as a kick when its low band outweighs its high band locally.
_KICK_RATIO = 1.0
_KICK_GATE = .15      # fraction of the rolling +-2 s low-band peak a beat must reach
_KICK_FLOOR = .02     # absolute low-band floor: no transient, no kick label
_KICK_EPS = .10       # keeps an evidence-free beat from scoring an even balance


def _kick_beat_labels(beats: np.ndarray, samples: np.ndarray) -> tuple[list[bool], list[float]]:
    """Flag the beats whose low band dominates their high band.

    Editors line cuts up to one instrument's kick, not to every beat, so each
    beat is labelled by the local balance of its 30-150 Hz and 2-8 kHz
    transient support. Each side is scaled by what that band is doing within
    +-2 s of the beat, and the low side must clear a local gate: a global
    track-maximum gate would relabel a whole quiet section, and a beat
    carrying no transient at all must never read as an even balance.
    """
    beats = np.asarray(beats, dtype=float)
    if len(beats) == 0 or samples.size == 0:
        return [], []
    # _bass_envelope runs at hop 256 and _treble_envelope at hop 64.
    bass, treble = _bass_envelope(samples), _treble_envelope(samples)
    labels, ratios = [], []
    for t in beats:
        center = float(t)
        low = _band_peak(bass, 256, center, .05)
        high = _band_peak(treble, 64, center, .05)
        low_ref = _band_peak(bass, 256, center, 2.0)
        high_ref = _band_peak(treble, 64, center, 2.0)
        ratio = ((low / max(low_ref, 1e-12)) + _KICK_EPS) / ((high / max(high_ref, 1e-12)) + _KICK_EPS)
        labels.append(bool(ratio >= _KICK_RATIO and low >= max(_KICK_GATE * low_ref, _KICK_FLOOR)))
        ratios.append(float(ratio))
    return labels, ratios


def _fill_grid(beats: np.ndarray, duration: float, step: float,
               onsets: list[float] | None = None,
               return_meta: bool = False) -> np.ndarray | tuple[np.ndarray, set[float], list[list[float]]]:
    """Complete a beat grid using local neighbor intervals where onsets exist.

    Quiet intros, soft bridges, or a dropped pulse leave holes the DBN could
    not confirm. Insert local-step beats inside oversized gaps only when onset
    density is non-negligible; otherwise mark the gap to hold across breakdowns.
    """
    beats = np.asarray(beats, dtype=float)
    intervals = np.diff(beats)
    def local_step_at(idx: int) -> float:
        if not len(intervals):
            return step
        win = [float(intervals[k]) for k in range(max(0, idx - 3), min(len(intervals), idx + 4)) if k != idx]
        if not win:
            return float(intervals[idx]) if intervals[idx] <= 1.6 * step else step
        lo_ref = min(win)
        single = [w for w in win if w <= 1.6 * lo_ref]
        return float(np.median(single)) if single else step

    filled = []
    interpolated: set[float] = set()
    held_gaps: list[list[float]] = []
    for idx, (prev, nxt) in enumerate(zip(beats, beats[1:])):
        filled.append(float(prev))
        loc = local_step_at(idx)
        gap = float(nxt - prev)
        k = int(round(gap / loc)) if loc > 0 else 1
        if k >= 2 and gap > 1.6 * loc:
            if onsets is not None:
                interior = [o for o in onsets if prev + .35 * loc <= o <= nxt - .35 * loc]
                if len(interior) / max(loc, gap - .7 * loc) < .25 / loc:
                    held_gaps.append([float(prev), float(nxt)])
                    continue
            for i in range(1, k):
                b_new = float(prev + gap * i / k)
                filled.append(b_new)
                interpolated.add(b_new)
    filled.append(float(beats[-1]))
    first_step = local_step_at(0)
    last_step = local_step_at(len(intervals) - 1)
    for _ in range(2):
        cand0 = filled[0] - first_step
        if cand0 >= 0 and (onsets is None or any(abs(float(o) - cand0) <= .6 * first_step for o in onsets)):
            filled.insert(0, float(cand0))
            interpolated.add(float(cand0))
        cand1 = filled[-1] + last_step
        if cand1 < duration and (onsets is None or any(abs(float(o) - cand1) <= .6 * last_step for o in onsets)):
            filled.append(float(cand1))
            interpolated.add(float(cand1))
    res = np.asarray(sorted(filled), dtype=float)
    _LAST_INTERPOLATED.clear()
    _LAST_INTERPOLATED.update(interpolated)
    return (res, interpolated, held_gaps) if return_meta else res


def _octave_check(beats: np.ndarray, onsets: list[float], bass_support_fn) -> np.ndarray:
    """Compare beat vs 2x/0.5x grids by onset support in dense vs sparse sections."""
    if len(beats) < 4 or not onsets:
        return beats
    intervals = np.diff(beats)
    step = float(np.median(intervals))
    if step <= 0:
        return beats
    gap_ix = np.where(intervals > 1.6 * step)[0] + 1
    out = []
    for seg in np.split(beats, gap_ix):
        if len(seg) < 4:
            out.extend(seg.tolist())
            continue
        seg_step = float(np.median(np.diff(seg)))
        if seg_step <= 0:
            out.extend(seg.tolist())
            continue
        sec_onsets = [float(o) for o in onsets if seg[0] - .5 * seg_step <= o <= seg[-1] + .5 * seg_step]
        if not sec_onsets:
            out.extend(seg.tolist())
            continue
        n_iv = max(1.0, float(seg[-1] - seg[0]) / seg_step)
        obs_per_beat = len(sec_onsets) / n_iv
        def grid_sup(times):
            return float(np.mean([min(abs(o - b) for o in sec_onsets) < .08 for b in times])) if len(times) else 0.0
        def onset_cov(times):
            return float(np.mean([min(abs(o - b) for b in times) < .08 for o in sec_onsets])) if len(times) else 0.0
        base_sup = grid_sup(seg)
        if seg_step * 2 <= 60.0 / _MIN_BPM and obs_per_beat < .75 and base_sup < .65:
            g0, g1 = seg[::2], seg[1::2]
            half_grid = g0 if (grid_sup(g0), bass_support_fn(g0)) >= (grid_sup(g1), bass_support_fn(g1)) else g1
            if len(half_grid) >= 2 and grid_sup(half_grid) > base_sup + .20:
                out.extend(half_grid.tolist())
                continue
        if seg_step / 2 >= 60.0 / _MAX_BPM and obs_per_beat >= 1.75:
            mids = .5 * (seg[:-1] + seg[1:])
            dbl = np.sort(np.concatenate([seg, mids]))
            if grid_sup(mids) >= .75 and onset_cov(dbl) > onset_cov(seg) + .25 and bass_support_fn(mids) >= .65 * bass_support_fn(seg):
                out.extend(dbl.tolist())
                continue
        out.extend(seg.tolist())
    return np.asarray(sorted(out), dtype=float)


def _madmom_beats(samples: np.ndarray, duration: float, controls: dict,
                  onsets: list[float]) -> tuple[np.ndarray, float] | None:
    """Track beats with madmom's RNN + DBN; ``None`` falls back to librosa.

    Returns beat times relative to the decoded interval and an estimated BPM
    from the median inter-beat interval. When the tracker locked onto the
    off-beat (onset support clearly prefers the half-beat phase), the whole
    grid is shifted by half a beat.
    """
    state = _madmom_state()
    if state is False:
        return None
    Signal, beat_rnn, DBN, version = state
    try:
        act = beat_rnn(Signal(samples.astype(np.float64), sample_rate=_SAMPLE_RATE, num_channels=1))
        act = np.asarray(act, dtype=float)
        if act.size < 3 or not np.isfinite(act).all() or float(act.max()) < .05:
            return None
        # Absolute thresholds from the density mapping only make sense for
        # strong activations; rescale for quiet recordings.
        threshold = min(controls["beat_threshold"], .8 * float(act.max()))
        dbn = DBN(min_bpm=_MIN_BPM, max_bpm=_MAX_BPM, fps=100,
                  transition_lambda=controls["transition_lambda"], threshold=threshold)
        beats = np.asarray(dbn(act), dtype=float)
        beats = beats[(beats >= 0) & (beats < duration)]
        if len(beats) < 3:
            return None
        intervals = np.diff(beats)
        step = float(np.median(intervals)) if len(intervals) else 0.0
        if step > 0 and len(beats) >= 3:
            bass = _bass_envelope(samples)
            def bass_support(times):
                if not len(times) or not len(bass):
                    return 0.0
                vals = []
                for t_val in times:
                    ix = int(np.clip(round(float(t_val) * _SAMPLE_RATE / 256), 0, len(bass) - 1))
                    vals.append(float(np.max(bass[max(0, ix - 2):min(len(bass), ix + 3)])))
                return float(np.mean(vals))
            # Drop coasted DBN beats inside onset-empty breakdowns where RNN activation is weak.
            kept = []
            for b in beats:
                ix = int(np.clip(round(float(b) * 100), 0, len(act) - 1))
                loc_act = float(np.max(act[max(0, ix - 5):min(len(act), ix + 6)]))
                has_onset = any(abs(float(o) - b) <= .6 * step for o in onsets) if onsets else True
                if loc_act >= .25 * threshold or has_onset or bass_support([b]) >= .05:
                    kept.append(float(b))
            if len(kept) >= 3:
                beats = np.asarray(kept, dtype=float)
                intervals = np.diff(beats)
                step = float(np.median(intervals)) if len(intervals) else step
            single = intervals[intervals <= 1.6 * step]
            reg = float(np.std(single) / np.median(single)) if len(single) >= 2 else float(np.std(intervals) / step)
            gap_ix = np.where(intervals > 1.6 * step)[0] + 1
            segments = np.split(beats, gap_ix)
            shifted_all = []
            for seg in segments:
                if len(seg) < 2:
                    shifted_all.extend(seg.tolist())
                    continue
                seg_diffs = np.diff(seg)
                local_next = np.append(seg_diffs, seg_diffs[-1])
                local_prev = np.insert(seg_diffs, 0, seg_diffs[0])
                phrase_len = 8
                for start_i in range(0, len(seg), phrase_len):
                    end_i = len(seg) if len(seg) - (start_i + phrase_len) < 3 else start_i + phrase_len
                    win = seg[start_i:end_i]
                    half_p = .5 * local_next[start_i:end_i]
                    half_m = .5 * local_prev[start_i:end_i]
                    win_onsets = [o for o in onsets if win[0] - half_m[0] <= o <= win[-1] + half_p[-1]]
                    def support(times):
                        if not win_onsets:
                            return 0.0
                        gaps = [min(abs(float(o) - b) for b in times) for o in win_onsets]
                        return float(np.mean(np.asarray(gaps) < .08))
                    on_beat = support(win)
                    at_bass = bass_support(win)
                    do_shift = False
                    if on_beat < .3 and (support(win + half_p) > on_beat + .2 or support(win - half_m) > on_beat + .2):
                        do_shift = True
                    elif bass_support(win + half_p) > max(2 * at_bass, .01) or bass_support(win - half_m) > max(2 * at_bass, .01):
                        do_shift = True
                    if do_shift:
                        cand_m = win - half_m
                        if cand_m[0] >= 0:
                            shifted_all.extend(cand_m.tolist())
                        else:
                            shifted_all.extend(cand_m[1:].tolist())
                        tail = float(win[-1] + half_p[-1])
                        if end_i == len(seg) and tail < duration and (bass_support([tail]) > max(2 * at_bass, .01) or support([tail]) > on_beat + .2):
                            shifted_all.append(tail)
                    else:
                        shifted_all.extend(win.tolist())
                    if end_i == len(seg):
                        break
            beats = np.asarray(sorted(shifted_all), dtype=float)
            if reg < .30 and len(beats) >= 2:
                beats = _fill_grid(beats, duration, step, onsets=onsets if onsets else None)
            beats = _octave_check(beats, onsets, bass_support)
        beats = beats[(beats >= 0) & (beats < duration)]
        if len(beats) < 3:
            return None
        intervals = np.diff(beats)
        med_iv = float(np.median(intervals)) if len(intervals) else 0.0
        single_iv = intervals[intervals <= 1.6 * med_iv] if med_iv > 0 else intervals
        ref_step = float(np.median(single_iv)) if len(single_iv) else med_iv
        bpm = 60.0 / ref_step if ref_step > 0 else 0.0
        return beats, bpm
    except Exception:
        return None


def _snap_beats_to_attacks(beats: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Refine each beat to its local acoustic attack.

    The DBN grid smooths over tempo drift, so grid times can sit tens of
    milliseconds off the audible transient. Snap to the strongest 10 ms
    low-band or spectral-flux rise within +-50 ms (whichever band peaks nearer
    the grid beat); beats below the local +-2 s gate keep the grid time.
    """
    beats = np.asarray(beats, dtype=float)
    if len(beats) == 0 or samples.size == 0:
        return beats
    step = max(1, int(.010 * _SAMPLE_RATE))
    t_s = np.arange(len(samples)) / _SAMPLE_RATE
    bass = _bass_envelope(samples)
    bass_s = np.interp(t_s, (np.arange(len(bass)) + .85) * 256 / _SAMPLE_RATE, bass)
    cs_b = np.cumsum(bass_s)
    rise_b = np.zeros_like(bass_s)
    rise_b[step:] = cs_b[step:] - cs_b[:-step]

    spec = np.abs(librosa.stft(samples, n_fft=512, hop_length=64))
    freqs = np.linspace(0, _SAMPLE_RATE / 2, spec.shape[0])
    sub = spec[freqs > 150]
    flux = np.maximum(np.diff(sub, axis=1, prepend=sub[:, :1]), 0).sum(axis=0)
    f_peak = float(np.max(flux)) if flux.size else 0.0
    if f_peak > 1e-12:
        flux = flux / max(f_peak, 1e-3)
    flux_s = np.interp(t_s, (np.arange(len(flux)) + 1.5) * 64 / _SAMPLE_RATE, flux)
    cs_f = np.cumsum(flux_s)
    rise_f = np.zeros_like(flux_s)
    rise_f[step:] = cs_f[step:] - cs_f[:-step]

    if max(float(np.max(rise_b)), float(np.max(rise_f))) <= 0:
        return beats
    window = int(.050 * _SAMPLE_RATE)
    local_win = int(2.0 * _SAMPLE_RATE)
    floor = 1e-3 * step
    refined = []
    for beat in beats:
        center = int(round(float(beat) * _SAMPLE_RATE))
        lo, hi = max(0, center - window), min(len(samples), center + window)
        if hi <= lo + step:
            refined.append(float(beat))
            continue
        llo, lhi = max(0, center - local_win), min(len(samples), center + local_win)
        cands = []
        for r in (rise_b, rise_f):
            seg = r[lo:hi]
            j = int(np.argmax(seg))
            gate = max(.15 * float(np.max(r[llo:lhi])), floor)
            if seg[j] >= gate:
                cands.append(max(lo, lo + j - step) / _SAMPLE_RATE)
        refined.append(min(cands, key=lambda t: abs(t - float(beat))) if cands else float(beat))
    return np.asarray(refined, dtype=float)


def _decode(path: str | Path, start: float, duration: float | None) -> np.ndarray:
    """Decode exactly the requested source interval to mono float samples."""
    require_tools()
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.9f}", "-i", str(path),
           "-map", "0:a:0", "-vn"]
    if duration is not None:
        cmd += ["-t", f"{duration:.9f}"]
    cmd += ["-ac", "1", "-ar", str(_SAMPLE_RATE), "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except OSError as exc:
        raise MediaError(f"cannot run ffmpeg: {exc}") from exc
    if result.returncode:
        detail = result.stderr.decode(errors="replace")[-3000:]
        raise MediaError(f"ffmpeg audio decode failed ({result.returncode}) for {path}\n{detail}")
    if len(result.stdout) % 4:
        raise MediaError("ffmpeg returned an incomplete audio sample")
    return np.frombuffer(result.stdout, dtype="<f4").copy()


def _normalize(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return []
    vals = np.maximum(np.asarray(values, dtype=np.float64), 0)
    peak = float(np.max(vals))
    if not math.isfinite(peak) or peak <= 1e-12:
        return [0.0] * int(vals.size)
    return np.clip(vals / peak, 0, 1).astype(float).tolist()


def analyze_audio(path: str | Path, cfg: Config) -> dict:
    """Decode and analyze the configured music interval.

    If duration is omitted, decode up to 30 seconds. Metadata is used only to
    decide whether a bounded 30 second decode is possible; the returned length
    always comes from decoded samples, which avoids trusting padded MP3 duration
    metadata. A short source is decoded to EOF.
    """
    cfg.validate()
    path = Path(path).expanduser().resolve()
    meta = probe(path)
    if meta.get("audio") is None:
        raise MediaError(f"no audio stream in {path}")
    audio_start = float(cfg.audio_start)
    audio_stream = meta["audio"]
    try:
        stream_duration = audio_stream.get("duration")
        if not stream_duration and audio_stream.get("duration_ts") and audio_stream.get("time_base"):
            stream_duration = float(audio_stream["duration_ts"]) * float(Fraction(audio_stream["time_base"]))
        source_duration = float(stream_duration or meta.get("format", {}).get("duration") or meta["duration"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        source_duration = float(meta.get("format", {}).get("duration") or meta["duration"])
    source_end = float(audio_stream.get("start_time") or 0.0) + source_duration
    if audio_start >= source_end:
        raise MediaError(f"audio_start ({audio_start:.3f}s) is at or beyond source audio end ({source_end:.3f}s)")
    identity = fingerprint(path)
    controls = _sensitivity_controls(cfg)
    _madmom_state()
    madmom_version = _MADMOM[3] if isinstance(_MADMOM, tuple) else "unavailable"
    cache_settings = {"audio_start": audio_start, "duration": cfg.duration, "fps": cfg.fps,
                      "sample_rate": _SAMPLE_RATE, "sensitivity": controls["sensitivity"],
                      "hop_length": controls["hop_length"], "onset_detection": controls["onset_detection"],
                      "density": controls["density"], "beat_threshold": controls["beat_threshold"],
                      "transition_lambda": controls["transition_lambda"],
                      "analysis_version": controls["analysis_version"],
                      "librosa_version": librosa.__version__,
                      "madmom_version": madmom_version,
                      "ffmpeg_version": subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True).stdout.decode().splitlines()[0]}
    cache_key = key_for(identity, cache_settings, "audio-analysis")
    cache_path = Path(cfg.cache_dir) / "audio" / f"{cache_key}.json"
    cached = read_cache(cache_path)
    if cached is not None and cached.get("identity", {}).get("sha256") == identity["sha256"]:
        cached["identity"] = identity
        return cached
    requested_duration = cfg.duration
    if requested_duration is None:
        estimated_remaining = max(0.0, source_end - audio_start)
        decode_limit = 30.0 if estimated_remaining >= 30.0 else None
        requested_duration = 30.0
    else:
        # Decode through a rounded-up output boundary when one is requested,
        # provided source music is present there. We still validate the full
        # unrounded user request independently below.
        requested_frames = int(round(float(requested_duration) * int(cfg.fps)))
        decode_limit = max(float(requested_duration), requested_frames / int(cfg.fps))
    samples = _decode(path, audio_start, decode_limit)
    decoded_duration = samples.size / _SAMPLE_RATE
    if decoded_duration <= 0:
        if cfg.duration is not None:
            raise MediaError(f"requested audio duration {cfg.duration:.3f}s exceeds available audio after {audio_start:.3f}s (no samples decoded)")
        raise MediaError(f"no audio samples decoded from {path} at {audio_start:.3f}s")
    if cfg.duration is None:
        requested_duration = min(30.0, decoded_duration)
    if cfg.duration is not None and decoded_duration + (1 / _SAMPLE_RATE) < float(cfg.duration):
        raise MediaError(f"requested audio duration {cfg.duration:.3f}s exceeds available audio after {audio_start:.3f}s ({decoded_duration:.3f}s decoded)")
    fps = int(cfg.fps)
    reached_eof = decode_limit is None or decoded_duration + 1 / _SAMPLE_RATE < decode_limit
    requested_frames = int(round(float(requested_duration) * fps))
    if reached_eof:
        total_frames = int(math.floor(decoded_duration * fps + 1e-9))
    else:
        total_frames = min(requested_frames, int(round(decoded_duration * fps)))
    if total_frames < 1:
        raise MediaError("decoded audio interval is shorter than one output frame")
    duration = total_frames / fps
    analysis_warnings = []
    if _MADMOM is False:
        analysis_warnings.append("madmom could not be loaded; fell back to the librosa beat tracker.")
    if reached_eof and decoded_duration * fps - total_frames > 1e-6:
        analysis_warnings.append("The soundtrack ended between output frames; duration was floored to avoid extending beyond available music.")
    # Ignore the fractional sample/frame tail so every analysis timeline agrees
    # with the frame-quantized interval advertised to callers.
    usable_samples = min(samples.size, int(round(duration * _SAMPLE_RATE)))
    samples = samples[:usable_samples]
    hop = controls["hop_length"]

    envelope = librosa.onset.onset_strength(y=samples, sr=_SAMPLE_RATE, hop_length=hop)
    env_norm = np.asarray(_normalize(envelope), dtype=np.float64)
    # Gamma compression raises the relative salience of quieter transients
    # without manufacturing events in silence or changing the raw strength
    # values exposed to the user.
    detection_envelope = np.power(env_norm, 1.0 / controls["sensitivity"])
    env_times = librosa.frames_to_time(np.arange(len(envelope)), sr=_SAMPLE_RATE, hop_length=hop).astype(float).tolist()
    onset_frames = librosa.onset.onset_detect(onset_envelope=detection_envelope, sr=_SAMPLE_RATE,
                                               hop_length=hop, units="frames", backtrack=False,
                                               **controls["onset_detection"])
    onset_times = librosa.frames_to_time(onset_frames, sr=_SAMPLE_RATE, hop_length=hop).astype(float).tolist()
    detector, bpm = "librosa_beat_track", 0.0
    _LAST_INTERPOLATED.clear()
    madmom_result = _madmom_beats(samples, duration, controls, onset_times)
    if madmom_result is not None:
        beat_times_arr, bpm = madmom_result
        detector = "madmom_dbn"
    else:
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=detection_envelope, sr=_SAMPLE_RATE,
                                                     hop_length=hop, tightness=max(20.0, 100.0 / math.sqrt(controls["sensitivity"])),
                                                     units="frames", trim=True)
        del tempo
        beat_frames = np.asarray(beat_frames, dtype=int).reshape(-1)
        beat_times_arr = librosa.frames_to_time(beat_frames, sr=_SAMPLE_RATE, hop_length=hop)
        beat_times_arr = beat_times_arr[(beat_times_arr >= 0) & (beat_times_arr < duration)]
    interp_flags = [any(abs(float(t) - bi) <= .06 for bi in _LAST_INTERPOLATED) for t in beat_times_arr]
    beat_times_arr = _snap_beats_to_attacks(beat_times_arr, samples)
    bass_env = _bass_envelope(samples)
    beat_strengths = []
    beat_bass = []
    # Onset strength peaks a frame or two after the transient itself, and the
    # beat tracker may sit exactly on the transient. Take the local maximum so
    # support reflects the event, not detector phase (values are sensitivity-
    # independent because they use the raw normalized envelope).
    for t in beat_times_arr:
        ix = int(np.clip(round(float(t) * _SAMPLE_RATE / hop), 0, max(0, len(env_norm) - 1)))
        lo, hi = max(0, ix - 1), min(len(env_norm), ix + 3)
        beat_strengths.append(float(np.max(env_norm[lo:hi])) if len(env_norm) else 0.0)
        b_ix = int(np.clip(round(float(t) * _SAMPLE_RATE / 256), 0, max(0, len(bass_env) - 1)))
        beat_bass.append(float(np.max(bass_env[max(0, b_ix - 1):min(len(bass_env), b_ix + 3)])) if len(bass_env) else 0.0)
    beat_labels, _ = _kick_beat_labels(beat_times_arr, samples)

    rms = librosa.feature.rms(y=samples, frame_length=2048, hop_length=hop, center=True)[0]
    energy_times = librosa.frames_to_time(np.arange(len(rms)), sr=_SAMPLE_RATE, hop_length=hop).astype(float).tolist()
    energy = _normalize(rms)

    rms_peak = float(np.max(rms)) if len(rms) else 0.0
    active = rms_peak > 1e-5 and float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) > 1e-6
    beat_count = len(beat_times_arr)
    beat_support = float(np.mean(beat_strengths)) if beat_strengths else 0.0
    intervals = np.diff(beat_times_arr)
    interval_regularity = 0.0
    if len(intervals) >= 2 and float(np.mean(intervals)) > 0:
        med_iv = float(np.median(intervals))
        single_iv = intervals[intervals <= 1.6 * med_iv] if med_iv > 0 else intervals
        ref_iv = single_iv if len(single_iv) >= 2 else intervals
        interval_regularity = max(0.0, 1.0 - float(np.std(ref_iv) / np.mean(ref_iv)))
    onset_support = float(np.mean([env_norm[min(len(env_norm)-1, max(0, int(round(t * _SAMPLE_RATE / hop))))]
                                   for t in onset_times])) if onset_times and len(env_norm) else 0.0
    # Heuristic signals are exposed so consumers can explain the decision.
    activity_signal = min(1.0, len(onset_times) / max(1.0, duration * 0.8)) if active else 0.0
    score = float(np.clip(.40 * min(1.0, beat_count / 8) + .25 * beat_support +
                          .20 * interval_regularity + .15 * activity_signal, 0, 1))
    if not active:
        fallback = "regular"
        method = "silence"
        warnings = ["No meaningful audio activity was detected; regular boundaries will be used."]
        beat_times_arr = np.asarray([], dtype=float)
        beat_strengths = []
        beat_bass = []
        beat_labels = []
        interp_flags = []
        score = 0.0
    elif beat_count >= 3 and beat_support >= .10 and interval_regularity >= .15:
        fallback = "none"
        method = detector
        warnings = []
    elif onset_times:
        fallback = "onsets"
        method = "weak_or_ambiguous_rhythm"
        warnings = ["Beat tracking was weak or ambiguous; usable onsets will guide boundaries."]
    else:
        fallback = "regular"
        method = "no_usable_onsets"
        warnings = ["No usable rhythmic events were detected; regular boundaries will be used."]

    result = {
        "identity": identity, "audio_start": audio_start,
        "requested_duration": float(requested_duration), "duration": duration,
        "total_frames": total_frames, "fps": fps, "sample_rate": _SAMPLE_RATE,
        "onset_times": [float(t) for t in onset_times], "onset_strength": _normalize(envelope),
        "onset_envelope_times": [float(t) for t in env_times],
        "beat_times": [float(t) for t in beat_times_arr], "beat_strengths": beat_strengths,
        "beat_bass": beat_bass, "beat_labels": [bool(b) for b in beat_labels],
        "interpolated_beats": [bool(f) for f in interp_flags],
        "energy_times": [float(t) for t in energy_times], "energy": energy,
        "analysis": {**controls, "detector": detector, "bpm": float(bpm),
                     "librosa_version": librosa.__version__, "madmom_version": madmom_version},
        "reliability": {"score": score, "method": method,
                        "heuristic": "activity, beat support, beat count and interval regularity; not a calibrated probability",
                        "signals": {"active": bool(active), "beat_count": int(beat_count),
                                    "beat_support": beat_support, "interval_regularity": interval_regularity,
                                    "onset_activity": activity_signal}},
        "fallback": fallback, "warnings": analysis_warnings + warnings,
    }
    write_json(cache_path, result)
    return result


def _energy_at(audio: dict, seconds: float) -> float:
    times, values = audio.get("energy_times", []), audio.get("energy", [])
    if not times or not values:
        return 0.5
    ix = int(np.clip(np.searchsorted(times, seconds), 0, len(values) - 1))
    return float(values[ix])


def cut_activity(audio: dict, cfg: Config) -> dict:
    """A relative-energy gate for editing, separate from beat detection.

    Trailing smoothing avoids spreading a drop across a long centered window. The
    80th percentile supplies a robust foreground reference. Hysteresis and a
    delayed release keep ordinary gaps between rhythmic pulses from chattering.
    No beats are deleted or synthesized; only cut eligibility changes.
    """
    duration = audio["total_frames"] / audio.get("fps", cfg.fps)
    times = np.asarray(audio.get("energy_times", []), dtype=float)
    values = np.asarray(audio.get("energy", []), dtype=float)
    metadata = {"enabled": cfg.quiet_threshold > 0 and bool(len(times)) and len(times) == len(values),
                "method": "trailing RMS, foreground percentile, hysteresis and delayed release",
                "reference_percentile": 80, "reference_energy": 0.,
                "rms_measurement": "centered 2048-sample RMS (approximately 93 ms), then trailing power smoothing",
                "relative_threshold": cfg.quiet_threshold, "threshold": 0., "release_threshold": 0.,
                "smoothing_seconds": cfg.quiet_window, "release_seconds": cfg.quiet_release,
                "active_intervals": [], "quiet_intervals": []}
    if not metadata["enabled"]:
        metadata["active_intervals"] = [[0., duration]]
        return metadata
    valid = (times >= 0) & (times <= duration) & np.isfinite(times) & np.isfinite(values)
    times, values = times[valid], np.maximum(0., values[valid])
    if not len(times):
        metadata["enabled"] = False
        metadata["active_intervals"] = [[0., duration]]
        return metadata
    steps = np.diff(times)
    dt = float(np.median(steps[steps > 0])) if np.any(steps > 0) else cfg.quiet_window
    window = max(1, min(len(values), int(round(cfg.quiet_window/dt))))
    power = np.pad(values*values, (window-1, 0), mode="edge")
    smoothed = np.sqrt(np.maximum(0., np.convolve(power, np.ones(window)/window, mode="valid")))
    reference = float(np.percentile(smoothed, 80))
    high, low = reference*cfg.quiet_threshold, reference*cfg.quiet_threshold*.75
    metadata.update(reference_energy=reference, threshold=high, release_threshold=low)
    active, start, below_since = False, 0., None
    intervals = []
    if reference > 1e-10:
        for t, value in zip(times, smoothed):
            if not active and value >= high:
                start, active, below_since = float(t), True, None
            elif active:
                if value < low:
                    if below_since is None:
                        below_since = float(t)
                    if t-below_since >= cfg.quiet_release-1e-9:
                        intervals.append([start, float(t)])
                        active, below_since = False, None
                else:
                    below_since = None
        if active:
            intervals.append([start, duration])
    cursor = 0.
    quiet = []
    for start, end in intervals:
        if start > cursor:
            quiet.append([cursor, start])
        cursor = end
    if cursor < duration:
        quiet.append([cursor, duration])
    metadata.update(active_intervals=intervals, quiet_intervals=quiet)
    return metadata


def cut_is_active(activity, seconds):
    return not activity["enabled"] or any(a-1e-9 <= seconds < b-1e-9 for a, b in activity["active_intervals"])


def _plan(candidates: list[int], end: int, fps: int, lo: float, hi: float,
          audio: dict, cap_frames: int | None = None, duration_weight: float = 1.,
          quiet_frames=None, salience: dict[int, float] | None = None) -> tuple[list[int], bool] | None:
    """Find an event-aligned path, relaxing soft shot limits only if needed."""
    points = sorted(set([0, end, *(p for p in candidates if 0 < p < end)]))
    lo_f, hi_f = lo * fps, hi * fps
    med_sal = float(np.median(list(salience.values()))) if salience else 0.0
    for relax in ((True,) if quiet_frames else (False, True)):
        best = [math.inf] * len(points)
        prev = [-1] * len(points)
        best[0] = 0.0
        for j in range(1, len(points)):
            sal_bonus = .35 * (float(salience.get(points[j], med_sal)) - med_sal) if (salience and j < len(points) - 1) else 0.0
            for i in range(j):
                length = points[j] - points[i]
                if length <= 0 or (cap_frames is not None and length > cap_frames):
                    continue
                if not relax and (length < lo_f - 1e-9 or length > hi_f + 1e-9):
                    continue
                seconds = length / fps
                energy = _energy_at(audio, (points[i] + points[j]) / (2 * fps))
                target = hi - (hi - lo) * (0.25 + 0.65 * energy)
                below = max(0.0, lo_f - length) / max(1.0, lo_f)
                above = max(0.0, length - hi_f) / max(1.0, hi_f)
                duration_cost = abs(seconds - target) / max(.1, hi - lo)
                quiet_penalty = 1000. if quiet_frames and points[j] in quiet_frames else 0.
                cost = best[i] + duration_weight*duration_cost - sal_bonus + (10.0 * (below + above) if relax else 0.0) + .005 + quiet_penalty
                if cost < best[j]:
                    best[j], prev[j] = cost, i
        if prev[-1] < 0:
            continue
        result = []
        at = len(points) - 1
        while at > 0:
            result.append(points[at])
            at = prev[at]
        result.append(0)
        result.reverse()
        limits_relaxed = any((b - a) < lo_f - 1e-9 or (b - a) > hi_f + 1e-9
                             for a, b in zip(result, result[1:]))
        return result, limits_relaxed
    return None


def choose_boundaries(audio: dict, cfg: Config, max_duration: float | None = None) -> tuple[list[int], list[str]]:
    """Choose absolute output frame boundaries while retaining detected events.

    Intermediate boundaries are selected beats whenever beat tracking passed
    its reliability gate, otherwise onsets, otherwise a regular frame grid.
    The final endpoint always remains the complete analyzed interval. ``max_duration``
    is a hard per-slot cap used while regrouping detected events.
    """
    cfg.validate()
    fps = int(audio.get("fps", cfg.fps))
    total = int(audio["total_frames"])
    if fps <= 0 or total < 1:
        raise ValueError("audio analysis must contain positive fps and total_frames")
    end = total
    cap_frames = None
    if max_duration is not None:
        if not math.isfinite(max_duration) or max_duration <= 0:
            raise ValueError("max_duration must be positive and finite")
        cap_frames = int(math.floor(max_duration * fps + 1e-9))
        if cap_frames < 1:
            raise ValueError("max_duration must be at least one output frame")
    lo, hi = cfg.shot_limits
    fallback = audio.get("fallback", "regular")
    warnings = list(audio.get("warnings", []))
    if fallback == "none":
        event_times = audio.get("beat_times", [])
    elif fallback == "onsets":
        event_times = audio.get("onset_times", [])
    else:
        event_times = []
    # Quantize every selected event once and deduplicate near-identical events.
    event_frames = sorted(set(int(round(float(t) * fps)) for t in event_times
                              if math.isfinite(float(t)) and float(t) >= 0))
    if event_frames:
        candidates = [f for f in event_frames if f < end]
    else:
        preferred = max(1, int(round(((lo + hi) / 2) * fps)))
        step = min(preferred, cap_frames) if cap_frames is not None else preferred
        # Spread regular boundaries evenly so the endpoint does not leave a
        # one-frame tail when a nearby balanced grid is available.
        count = max(1, int(math.ceil(end / step)))
        candidates = [int(round(i * end / count)) for i in range(1, count)]
    # kick_focus lines cuts up to one instrument's kick rather than every beat,
    # so keep the full beat grid aside in case the kicks alone cannot cover.
    all_beat_candidates = None
    if fallback == "none" and getattr(cfg, "kick_focus", False):
        labels = audio.get("beat_labels", [])
        kick_frames = {int(round(float(t) * fps)) for t, lab in zip(event_times, labels) if lab}
        if kick_frames:
            all_beat_candidates = candidates
            candidates = [f for f in candidates if f in kick_frames]
        else:
            warnings.append("kick_focus found no kick beats; using all detected beats.")
    activity = cut_activity(audio, cfg)
    suppressed = {frame for frame in candidates if not cut_is_active(activity, frame/fps)}
    eligible = [frame for frame in candidates if frame not in suppressed]
    activity.update(suppressed_event_frames=sorted(suppressed), eligible_event_frames=eligible,
                    forced_cut_frames=[], relaxed=False)
    audio["cut_activity"] = activity
    salience: dict[int, float] = {}
    if event_times:
        b_strs = audio.get("beat_strengths", [])
        b_bass = audio.get("beat_bass", [])
        b_interp = audio.get("interpolated_beats", [])
        e_times = np.asarray(audio.get("energy_times", []), dtype=float)
        e_vals = np.asarray(audio.get("energy", []), dtype=float)
        for idx, t in enumerate(event_times):
            if not math.isfinite(float(t)) or float(t) < 0:
                continue
            f = int(round(float(t) * fps))
            if f >= end:
                continue
            if idx < len(b_interp) and b_interp[idx]:
                sal = .05
            else:
                s_val = float(b_strs[idx]) if idx < len(b_strs) else .5
                bass_val = float(b_bass[idx]) if idx < len(b_bass) else s_val
                if len(e_times) and len(e_vals) == len(e_times):
                    pre = e_vals[(e_times >= float(t) - .5) & (e_times < float(t) - .04)]
                    post = e_vals[(e_times >= float(t)) & (e_times <= float(t) + .5)]
                    pre_m = float(np.mean(pre)) if len(pre) else _energy_at(audio, max(0.0, float(t) - .25))
                    post_m = float(np.mean(post)) if len(post) else _energy_at(audio, float(t) + .25)
                    rms_step = max(0.0, post_m - pre_m)
                else:
                    rms_step = 0.0
                sal = (s_val + bass_val + rms_step) / 3.0
            salience[f] = max(salience.get(f, 0.0), sal)
    planned = _plan(eligible, end, fps, lo, hi, audio, cap_frames=cap_frames,
                    duration_weight=cfg.weights["duration"], salience=salience)
    if planned is None and suppressed:
        # Quiet holds are relaxed only to fit real continuous footage, with a
        # large per-cut penalty so the minimum necessary extra cuts are used.
        planned = _plan(candidates, end, fps, lo, hi, audio, cap_frames=cap_frames,
                        duration_weight=cfg.weights["duration"], quiet_frames=suppressed, salience=salience)
        if planned is not None:
            forced = [f for f in planned[0][1:-1] if f in suppressed]
            activity.update(forced_cut_frames=forced, relaxed=bool(forced))
            if forced:
                warnings.append(f"Quiet-section hold relaxed for {len(forced)} cut(s): available continuous footage is too short. Add longer clips to avoid these cuts.")
    if planned is None and all_beat_candidates:
        # The kicks alone cannot cover the interval; fall back to every beat.
        # Suppression is remeasured on the full grid: reusing the
        # kick-restricted set would leave quiet holds unrecorded on non-kick
        # beats, failing downstream validation.
        suppressed_all = {f for f in all_beat_candidates if not cut_is_active(activity, f / fps)}
        planned = _plan(all_beat_candidates, end, fps, lo, hi, audio, cap_frames=cap_frames,
                        duration_weight=cfg.weights["duration"], quiet_frames=suppressed_all,
                        salience=salience)
        if planned is not None:
            forced = [f for f in planned[0][1:-1] if f in suppressed_all]
            activity.update(suppressed_event_frames=sorted(suppressed_all),
                            eligible_event_frames=[f for f in all_beat_candidates if f not in suppressed_all],
                            forced_cut_frames=forced, relaxed=bool(forced))
            warnings.append("kick_focus: too few kick beats to cover the interval; boundaries fell back to all detected beats.")
    if planned is None:
        cap_seconds = cap_frames / fps if cap_frames is not None else hi
        raise MediaError(f"selected beat grid cannot cover all {end / fps:.3f}s of footage with slots at most {cap_seconds:.3f}s; use an explicit onset or regular fallback")
    boundaries, relaxed = planned
    if relaxed:
        if suppressed and not activity["relaxed"]:
            warnings.append("Shot-duration preferences relaxed to hold quiet sections without transitions; active-section cuts remain on the selected rhythm.")
        else:
            warnings.append("Detected event spacing could not satisfy configured shot limits; limits were relaxed to preserve the selected rhythm and duration.")
    if len(boundaries) < 2:
        boundaries = [0, end]
        warnings.append("No usable interior boundary fit; used a single duration-bounded segment.")
    return boundaries, list(dict.fromkeys(warnings))

