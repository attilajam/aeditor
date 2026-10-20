import wave

import numpy as np
import pytest

from autoedit.audio import _fill_grid, _kick_beat_labels, _madmom_state, _octave_check, analyze_audio, choose_boundaries
from autoedit.config import Config
from autoedit.probe import MediaError


def write_wav(path, samples, sr=22_050):
    pcm = np.clip(samples * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sr)
        out.writeframes(pcm.tobytes())


def clicks(duration=10, sr=22_050, positions=None):
    x = np.zeros(int(duration * sr), dtype=np.float32)
    positions = positions if positions is not None else np.arange(.25, duration - .1, .5)
    # Short, broadband click with enough level to survive PCM conversion.
    pulse = np.hanning(101).astype(np.float32) * .9
    for t in positions:
        start = int(t * sr)
        x[start:start + len(pulse)] += pulse
    return x


def mixed_strength_clicks(duration=10, sr=22_050):
    """Full beats at 1s and quieter in-between transients at .5s."""
    x = np.zeros(int(duration * sr), dtype=np.float32)
    pulse = np.hanning(101).astype(np.float32)
    for beat in np.arange(.25, duration - .1, 1.0):
        start = int(beat * sr)
        x[start:start + len(pulse)] += .9 * pulse
    for transient in np.arange(.75, duration - .1, 1.0):
        start = int(transient * sr)
        # PCM-representable but well below the loud downbeat transients.
        x[start:start + len(pulse)] += .001 * pulse
    return x


def test_silence_marks_regular_fallback_and_has_no_fabricated_beats(tmp_path):
    path = tmp_path / "silence.wav"
    write_wav(path, np.zeros(4 * 22_050, dtype=np.float32))
    result = analyze_audio(path, Config(duration=3.5))
    assert result["fallback"] == "regular"
    assert result["beat_times"] == []
    assert result["reliability"]["score"] == 0
    assert result["total_frames"] == 105
    assert result["warnings"]


def test_click_track_has_supported_beats_and_boundaries_follow_them(tmp_path):
    path = tmp_path / "clicks.wav"
    write_wav(path, clicks())
    cfg = Config(duration=8, pacing="fast", min_shot=0.5, max_shot=1.0)
    result = analyze_audio(path, cfg)
    assert result["fallback"] == "none"
    assert len(result["beat_times"]) >= 5
    known = np.arange(.25, 8., .5)
    # Detector window/hop delay plus transient shape: explicit 75 ms tolerance.
    assert max(min(abs(t-event) for event in known) for t in result["beat_times"]) <= .075
    phase = result["beat_times"][0]
    assert all(abs((float(t) - phase) - round((float(t) - phase) * 2) / 2) < .10
               for t in result["beat_times"])
    boundaries, _ = choose_boundaries(result, cfg)
    beat_frames = {round(t * cfg.fps) for t in result["beat_times"]}
    assert boundaries[0] == 0 and boundaries[-1] == result["total_frames"]
    assert all(min(abs(frame - beat) for beat in beat_frames) <= 1
               for frame in boundaries[1:-1])


def test_audio_start_decodes_offset_interval_and_reports_relative_events(tmp_path):
    path = tmp_path / "offset.wav"
    # The selected interval starts at source second two; clicks are at 2.25, ...
    write_wav(path, clicks(duration=10, positions=np.arange(2.25, 8.1, .5)))
    result = analyze_audio(path, Config(audio_start=2.0, duration=5.0))
    assert result["audio_start"] == 2.0
    assert result["duration"] == 5.0
    assert result["beat_times"]
    assert min(result["onset_times"]) < .5
    assert abs(min(result["onset_times"]) - .25) < .12


def test_boundary_planner_deduplicates_quantized_events_and_avoids_tiny_tail():
    cfg = Config(fps=30, pacing="medium", min_shot=1.0, max_shot=2.0)
    audio = {"fps": 30, "total_frames": 90, "fallback": "none",
             "beat_times": [1.0, 1.001, 2.0, 2.9], "onset_times": [],
             "energy_times": [0, 3], "energy": [0.5, 0.5], "warnings": []}
    boundaries, warnings = choose_boundaries(audio, cfg)
    assert boundaries[0] == 0 and boundaries[-1] == 90
    assert len(boundaries) == len(set(boundaries))
    assert all(30 <= b - a <= 60 for a, b in zip(boundaries, boundaries[1:]))
    assert not warnings


def test_explicit_duration_beyond_remaining_audio_is_clear_error(tmp_path):
    path = tmp_path / "short.wav"
    write_wav(path, clicks(duration=2))
    with pytest.raises(MediaError, match="exceeds available audio"):
        analyze_audio(path, Config(audio_start=1.0, duration=2.0))


def test_max_duration_caps_each_slot_without_shortening_analyzed_interval():
    cfg = Config(fps=30, pacing="medium", min_shot=1.0, max_shot=2.0)
    audio = {"fps": 30, "total_frames": 120, "fallback": "regular",
             "beat_times": [], "onset_times": [], "energy_times": [0, 4],
             "energy": [0.2, 0.2], "warnings": []}
    boundaries, warnings = choose_boundaries(audio, cfg, max_duration=1.0)
    assert boundaries[0] == 0 and boundaries[-1] == 120
    assert all(0 < b - a <= 30 for a, b in zip(boundaries, boundaries[1:]))
    assert boundaries == [0, 30, 60, 90, 120]
    assert not warnings


def test_max_duration_fails_when_beat_grid_cannot_cover_entire_interval():
    cfg = Config(fps=30, pacing="medium", min_shot=0.5, max_shot=1.0)
    audio = {"fps": 30, "total_frames": 120, "fallback": "none",
             "beat_times": [1.0, 2.0, 3.0], "onset_times": [],
             "energy_times": [0, 4], "energy": [0.5, 0.5], "warnings": []}
    with pytest.raises(MediaError, match="cannot cover all 4.000s"):
        choose_boundaries(audio, cfg, max_duration=0.75)


def test_default_eof_duration_floors_partial_frame_and_warns(tmp_path):
    path = tmp_path / "fractional.wav"
    write_wav(path, np.zeros(int(3.51 * 22_050), dtype=np.float32))
    result = analyze_audio(path, Config(fps=30, cache_dir=str(tmp_path / "cache")))
    assert result["total_frames"] == 105
    assert result["duration"] == 3.5
    assert any("floored" in warning for warning in result["warnings"])


def test_higher_sensitivity_detects_quieter_transients_and_uses_distinct_cache_key(tmp_path):
    path = tmp_path / "mixed strength clicks.wav"
    write_wav(path, mixed_strength_clicks())
    cache = str(tmp_path / "shared cache")
    baseline = analyze_audio(path, Config(duration=9, beat_sensitivity=1.0, cache_dir=cache))
    sensitive = analyze_audio(path, Config(duration=9, beat_sensitivity=2.0, cache_dir=cache))

    assert baseline["analysis"]["sensitivity"] == 1.0
    assert baseline["analysis"]["hop_length"] == 512
    assert sensitive["analysis"]["sensitivity"] == 2.0
    assert sensitive["analysis"]["hop_length"] < baseline["analysis"]["hop_length"]
    assert sensitive["analysis"]["onset_detection"]["delta"] < baseline["analysis"]["onset_detection"]["delta"]
    # Denser beat grid: lower DBN threshold and looser transition lambda.
    assert sensitive["analysis"]["beat_threshold"] < baseline["analysis"]["beat_threshold"]
    assert sensitive["analysis"]["transition_lambda"] < baseline["analysis"]["transition_lambda"]
    assert len(sensitive["onset_times"]) > len(baseline["onset_times"])
    quiet_times = np.arange(.75, 9.0, 1.0)
    detected_quiet = [t for t in sensitive["onset_times"]
                      if min(abs(t - q) for q in quiet_times) <= .06]
    assert len(detected_quiet) >= 3
    assert sensitive["fallback"] == "none"
    assert len(sensitive["beat_times"]) < len(sensitive["onset_times"])
    boundaries, _ = choose_boundaries(sensitive, Config(duration=9, beat_sensitivity=2.0, cache_dir=cache))
    beat_frames = {round(t * sensitive["fps"]) for t in sensitive["beat_times"]}
    assert all(min(abs(frame - beat) for beat in beat_frames) <= 1 for frame in boundaries[1:-1])
    cache_files = list((tmp_path / "shared cache" / "audio").glob("*.json"))
    assert len(cache_files) == 2


def test_sensitive_analysis_keeps_silence_on_regular_fallback(tmp_path):
    path = tmp_path / "quiet silence.wav"
    write_wav(path, np.zeros(5 * 22_050, dtype=np.float32))
    result = analyze_audio(path, Config(duration=4, beat_sensitivity=3.0,
                                        cache_dir=str(tmp_path / "silent cache")))
    assert result["analysis"]["sensitivity"] == 3.0
    assert result["fallback"] == "regular"
    assert result["beat_times"] == []


def test_drifting_kicks_snap_beats_to_transient_starts(tmp_path):
    if _madmom_state() is False:
        pytest.skip("madmom beat tracker unavailable")
    sr = 22_050
    true, t, i = [], 0.25, 0
    while t < 9.5:
        true.append(t)
        t += 0.46 * (1 + 0.08 * np.sin(i * 0.7)) * (1.03 if i % 2 else 0.97)
        i += 1
    x = np.zeros(int(10 * sr), dtype=np.float32)
    n = int(0.09 * sr)
    tt = np.arange(n) / sr
    kick = (0.9 * np.minimum(tt / 0.012, 1) * np.exp(-tt * 28) *
            np.sin(2 * np.pi * 55 * tt)).astype(np.float32)
    for t0 in true:
        s = int(t0 * sr)
        x[s:s + n] += kick
    hn = int(0.02 * sr)
    ht = np.arange(hn) / sr
    hat = (0.35 * np.exp(-ht * 160) * np.sin(2 * np.pi * 6000 * ht)).astype(np.float32)
    for t0 in np.array(true[:-1]) + np.diff(true) / 2:
        s = int(t0 * sr)
        x[s:s + hn] += hat
    path = tmp_path / "drifting kicks.wav"
    write_wav(path, x)
    result = analyze_audio(path, Config(duration=9.5, cache_dir=str(tmp_path / "cache")))
    assert result["fallback"] == "none"
    errors = []
    for t0 in true:
        nearest = min(result["beat_times"], key=lambda beat: abs(beat - t0))
        if abs(nearest - t0) < 0.25:
            errors.append(abs(nearest - t0))
    assert len(errors) == len(true)
    assert sum(errors) / len(errors) <= 0.010
    assert max(errors) <= 0.030

    # Loud off-beat and near-beat hats must not blind or steal the kick snap.
    x_hats = np.zeros(int(10 * sr), dtype=np.float32)
    quiet_kick = (0.35 / 0.9) * kick
    loud_hat = (0.95 * np.exp(-ht * 80) * np.sin(2 * np.pi * 6000 * ht)).astype(np.float32)
    for t0 in true:
        s = int(t0 * sr)
        x_hats[s:s + n] += quiet_kick
    for t0 in np.array(true[:-1]) + np.diff(true) / 2:
        s = int(t0 * sr)
        x_hats[s:s + hn] += loud_hat
    for t0 in true[::2]:
        s = int((t0 + 0.038) * sr)
        if s + hn <= len(x_hats):
            x_hats[s:s + hn] += loud_hat
    hat_path = tmp_path / "loud hats.wav"
    write_wav(hat_path, x_hats)
    hat_res = analyze_audio(hat_path, Config(duration=9.5, cache_dir=str(tmp_path / "cache")))
    assert hat_res["fallback"] == "none"
    hat_errors = [min(abs(b - t0) for b in hat_res["beat_times"]) for t0 in true]
    assert sum(hat_errors) / len(hat_errors) <= 0.010
    assert max(hat_errors) <= 0.030


def test_half_beat_locked_drifting_track_recovers_kick_phase(tmp_path):
    if _madmom_state() is False:
        pytest.skip("madmom beat tracker unavailable")
    sr = 22_050
    true, t = [], 0.25
    while t < 9.5:
        true.append(t)
        t += 0.36 + 0.28 * (t / 9.5)
    x = np.zeros(int(10 * sr), dtype=np.float32)
    n = int(0.09 * sr)
    tt = np.arange(n) / sr
    kick = (0.45 * np.minimum(tt / 0.012, 1) * np.exp(-tt * 28) *
            np.sin(2 * np.pi * 55 * tt)).astype(np.float32)
    for t0 in true:
        s = int(t0 * sr)
        x[s:s + n] += kick
    sn = int(0.05 * sr)
    st = np.arange(sn) / sr
    offbeat = (0.65 * np.exp(-st * 60) * np.sin(2 * np.pi * 2000 * st)).astype(np.float32)
    for t0 in np.array(true[:-1]) + np.diff(true) / 2:
        s = int(t0 * sr)
        x[s:s + sn] += offbeat
    path = tmp_path / "half beat drift.wav"
    write_wav(path, x)
    result = analyze_audio(path, Config(duration=9.5, cache_dir=str(tmp_path / "cache")))
    assert result["fallback"] == "none"
    errors = [min(abs(b - t0) for b in result["beat_times"]) for t0 in true]
    assert sum(errors) / len(errors) <= 0.010
    assert max(errors) <= 0.030


def test_tempo_switch_and_breakdown_hold_and_octave_check(tmp_path):
    if _madmom_state() is False:
        pytest.skip("madmom beat tracker unavailable")
    sr = 22_050
    n = int(0.09 * sr)
    tt = np.arange(n) / sr
    kick = (0.9 * np.minimum(tt / 0.012, 1) * np.exp(-tt * 28) *
            np.sin(2 * np.pi * 55 * tt)).astype(np.float32)
    hn = int(0.02 * sr)
    ht = np.arange(hn) / sr
    hat = (0.35 * np.exp(-ht * 160) * np.sin(2 * np.pi * 6000 * ht)).astype(np.float32)

    # 1. Tempo switch: 100 BPM -> 128 BPM mid-track stays phase-locked.
    t1 = np.arange(0.25, 4.8, 0.60)
    t2 = np.arange(5.05, 9.8, 60.0 / 128.0)
    true_ts = np.concatenate([t1, t2])
    x_ts = np.zeros(int(10 * sr), dtype=np.float32)
    for t0 in true_ts:
        s = int(t0 * sr)
        x_ts[s:s + n] += kick
        x_ts[s:s + hn] += hat
    ts_path = tmp_path / "tempo switch.wav"
    write_wav(ts_path, x_ts)
    ts_res = analyze_audio(ts_path, Config(duration=9.8, cache_dir=str(tmp_path / "cache")))
    assert ts_res["fallback"] == "none"
    post_errs = [min(abs(b - t0) for b in ts_res["beat_times"]) for t0 in t2]
    assert sum(post_errs) / len(post_errs) <= 0.010
    assert max(post_errs) <= 0.030

    # _fill_grid fills gaps with local neighbor intervals, not global step.
    holes = np.delete(true_ts, [3, 12])
    filled, interp, gaps = _fill_grid(holes, 9.8, float(np.median(np.diff(holes))),
                                      onsets=true_ts.tolist(), return_meta=True)
    assert max(min(abs(b - t0) for b in filled) for t0 in true_ts) < 0.005
    assert len(interp) == 2 and not gaps

    # 2. Breakdown synth: no interior beats or boundary markers inside breakdown.
    b1 = np.arange(0.25, 3.5, 0.5)
    b2 = np.arange(7.25, 9.8, 0.5)
    x_bd = np.zeros(int(10 * sr), dtype=np.float32)
    for t0 in np.concatenate([b1, b2]):
        s = int(t0 * sr)
        x_bd[s:s + n] += kick
    pad_s, pad_e = int(3.4 * sr), int(7.1 * sr)
    pad_t = np.arange(pad_e - pad_s) / sr
    env_pad = np.sin(np.pi * np.arange(pad_e - pad_s) / (pad_e - pad_s)) ** 2
    x_bd[pad_s:pad_e] += (0.08 * env_pad * np.sin(2 * np.pi * 220 * pad_t)).astype(np.float32)
    bd_path = tmp_path / "breakdown.wav"
    write_wav(bd_path, x_bd)
    bd_cfg = Config(duration=9.8, min_shot=0.8, max_shot=2.0, cache_dir=str(tmp_path / "cache"))
    bd_res = analyze_audio(bd_path, bd_cfg)
    assert not [b for b in bd_res["beat_times"] if 3.4 < b < 7.1]
    boundaries, _ = choose_boundaries(bd_res, bd_cfg)
    assert not [f for f in boundaries[1:-1] if 3.4 * bd_cfg.fps < f < 7.1 * bd_cfg.fps]

    # 3. Octave check: sparse section halves to 0.5x; dense section doubles to 2x.
    sparse = _octave_check(np.arange(0.4, 4.0, 0.4), np.arange(0.4, 4.0, 0.8).tolist(), lambda _: 0.5)
    assert np.allclose(sparse, np.arange(0.4, 4.0, 0.8))
    dense = _octave_check(np.arange(0.45, 4.5, 0.9), np.arange(0.45, 4.5, 0.45).tolist(), lambda _: 0.5)
    assert np.allclose(dense, np.arange(0.45, 4.5, 0.45))


def test_alternating_kick_and_snare_beats_get_instrument_labels(tmp_path):
    sr = 22_050
    period = 60.0 / 115.0
    kicks = np.arange(.30, 8.0, 2 * period)
    snares = np.arange(.30 + period, 8.0, 2 * period)
    x = np.zeros(int(9.0 * sr), dtype=np.float32)
    n, tt = int(.09 * sr), np.arange(int(.09 * sr)) / sr
    kick = (.9 * np.minimum(tt / .012, 1) * np.exp(-tt * 28) *
            np.sin(2 * np.pi * 55 * tt)).astype(np.float32)
    for t0 in kicks:
        s = int(t0 * sr)
        x[s:s + n] += kick
    sn = int(.06 * sr)
    st = np.arange(sn) / sr
    rng = np.random.default_rng(7)
    mag = np.fft.rfft(rng.standard_normal(sn).astype(np.float32))
    band = np.fft.rfftfreq(sn, 1 / sr)
    mag[(band < 1500) | (band > 8000)] = 0
    crack = np.fft.irfft(mag, sn).astype(np.float32)
    crack /= np.abs(crack).max()
    snare = (.9 * np.exp(-st * 45) * crack).astype(np.float32)
    for t0 in snares:
        s = int(t0 * sr)
        x[s:s + sn] += snare
    path = tmp_path / "kick snare.wav"
    write_wav(path, np.clip(x, -.99, .99))

    beats = np.sort(np.concatenate([kicks, snares]))
    expected = [any(abs(float(t) - k) < 1e-9 for k in kicks) for t in beats]
    labels, ratios = _kick_beat_labels(beats, x)
    assert labels == expected
    kick_ratios = [r for r, lab in zip(ratios, labels) if lab]
    other_ratios = [r for r, lab in zip(ratios, labels) if not lab]
    assert min(kick_ratios) > max(other_ratios)

    # The same labelling must reach the cached analysis, aligned per beat.
    result = analyze_audio(path, Config(duration=9.0, cache_dir=str(tmp_path / "cache")))
    assert len(result["beat_labels"]) == len(result["beat_times"])
    assert result["fallback"] == "none"
    checked = 0
    for t, label in zip(result["beat_times"], result["beat_labels"]):
        if min(abs(t - k) for k in kicks) <= .10:
            assert label
            checked += 1
        elif min(abs(t - s) for s in snares) <= .10:
            assert not label
            checked += 1
    assert checked >= 8

    # End to end: with kick_focus the planner may only cut on labelled kicks.
    kick_cfg = Config(duration=9.0, fps=30, pacing="medium", min_shot=1.0, max_shot=2.0,
                      quiet_threshold=0.0, kick_focus=True, cache_dir=str(tmp_path / "cache"))
    boundaries, _ = choose_boundaries(result, kick_cfg)
    kick_frames = {round(t * 30) for t, lab in zip(result["beat_times"], result["beat_labels"]) if lab}
    assert boundaries[-1] == result["total_frames"]
    assert all(frame in kick_frames for frame in boundaries[1:-1])


def test_kick_focus_cuts_on_kick_beats_and_falls_back_when_kicks_too_sparse():
    def grid(beats, labels, total_frames=240):
        return {"fps": 30, "total_frames": total_frames, "fallback": "none",
                "beat_times": beats, "beat_labels": labels,
                "beat_strengths": [.2 if lab else .9 for lab in labels],
                "beat_bass": [.5] * len(beats), "interpolated_beats": [False] * len(beats),
                "onset_times": [], "energy_times": [0, 8], "energy": [.5, .5], "warnings": []}

    plain_cfg = Config(fps=30, pacing="medium", min_shot=1.0, max_shot=2.0)
    kick_cfg = Config(fps=30, pacing="medium", min_shot=1.0, max_shot=2.0, kick_focus=True)
    beats = [round(float(t), 3) for t in np.arange(.5, 8.0, .5)]
    dense_kicks = (1.0, 2.5, 4.0, 5.5, 7.0)
    labels = [any(abs(t - k) < 1e-9 for k in dense_kicks) for t in beats]
    kick_frames = {round(t * 30) for t, lab in zip(beats, labels) if lab}

    # Kicks every 1.5 s between half-second beats: the kick grid alone covers 8 s.
    boundaries, warnings = choose_boundaries(grid(beats, labels), kick_cfg)
    assert boundaries[0] == 0 and boundaries[-1] == 240
    assert all(frame in kick_frames for frame in boundaries[1:-1])
    assert not warnings

    # Without kick_focus the louder off-beats win, so the two runs differ.
    plain, _ = choose_boundaries(grid(beats, labels), plain_cfg)
    assert any(frame not in kick_frames for frame in plain[1:-1])

    # Kicks stop at 3 s, so the kick grid cannot cover 8 s under a 2 s cap: the
    # planner retries with every beat and names the fallback in a warning.
    sparse_labels = [any(abs(t - k) < 1e-9 for k in (1.0, 2.0, 3.0)) for t in beats]
    fallback_boundaries, fallback_warnings = choose_boundaries(grid(beats, sparse_labels), kick_cfg, max_duration=2.0)
    assert any("kick" in warning for warning in fallback_warnings)
    assert any(frame not in kick_frames for frame in fallback_boundaries[1:-1])


def test_kick_focus_ignored_when_beat_tracking_fell_back():
    cfg = Config(fps=30, pacing="medium", min_shot=1.0, max_shot=2.0, kick_focus=True)
    audio = {"fps": 30, "total_frames": 90, "fallback": "onsets",
             "beat_times": [1.0, 2.0], "beat_labels": [True, True], "onset_times": [1.0, 2.0, 2.9],
             "energy_times": [0, 3], "energy": [.5, .5], "warnings": []}
    boundaries, warnings = choose_boundaries(audio, cfg)
    assert boundaries[0] == 0 and boundaries[-1] == 90
    assert not warnings


def test_salience_weighted_planner_selects_loud_drop_on_off_duration_beat(tmp_path):
    if _madmom_state() is False:
        pytest.skip("madmom beat tracker unavailable")
    sr = 22_050
    x = np.zeros(int(4.5 * sr), dtype=np.float32)
    n = int(0.09 * sr)
    tt = np.arange(n) / sr
    kick = (0.95 * np.minimum(tt / 0.012, 1) * np.exp(-tt * 28) *
            np.sin(2 * np.pi * 55 * tt)).astype(np.float32)
    pulse = np.hanning(201).astype(np.float32)
    for t0 in np.arange(0.35, 4.2, 0.35):
        s = int(t0 * sr)
        if t0 < 2.40:
            x[s:s + len(pulse)] += 0.25 * pulse
        elif abs(t0 - 2.45) < 0.01:
            x[s:s + n] += kick
            x[s:s + len(pulse)] += 0.95 * pulse
        else:
            x[s:s + n] += 0.5 * kick
            x[s:s + len(pulse)] += 0.40 * pulse
    drop_s = int(2.45 * sr)
    t_post = np.arange(len(x) - drop_s) / sr
    x[drop_s:] += (0.35 * np.sin(2 * np.pi * 110 * t_post)).astype(np.float32)
    x = np.clip(x, -0.99, 0.99)
    path = tmp_path / "drop.wav"
    write_wav(path, x)
    cfg = Config(duration=4.0, fps=30, pacing="medium", min_shot=1.4, max_shot=2.6,
                 quiet_threshold=0.0, cache_dir=str(tmp_path / "cache"))
    res = analyze_audio(path, cfg)
    boundaries, _ = choose_boundaries(res, cfg)
    drop_frame = round(2.45 * cfg.fps)
    assert any(abs(b - drop_frame) <= 1 for b in boundaries[1:-1])





