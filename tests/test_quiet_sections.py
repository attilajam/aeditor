"""Energy-aware event gating tests for quiet intros and breakdowns."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import wave

import numpy as np

from autoedit.audio import analyze_audio, choose_boundaries, cut_activity
from autoedit.config import Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from generate_test_media import generate_media


FPS = 30


def _energy_audio(duration=10.0, fps=FPS, quiet_ranges=((0.0, 4.0),), *, quiet=.1, loud=1.0,
                  event_step=.5, event_start=.5):
    times = np.arange(0.0, duration + .025, .05)
    energy = np.full(len(times), loud, dtype=float)
    for start, end in quiet_ranges:
        energy[(times >= start) & (times < end)] = quiet
    event_times = np.arange(event_start, duration, event_step, dtype=float).tolist()
    return {
        "fps": fps,
        "total_frames": round(duration * fps),
        "duration": duration,
        "fallback": "none",
        "beat_times": event_times,
        "onset_times": event_times,
        "energy_times": times.tolist(),
        "energy": energy.tolist(),
        "warnings": [],
    }


def _config(**kwargs):
    options = {"fps": FPS, "pacing": "fast", "beat_sensitivity": 2.0,
               "quiet_threshold": .6, "quiet_window": .4, "quiet_release": .3,
               "cache_dir": ".autoedit-cache-test"}
    options.update(kwargs)
    return Config(**options).validate()


def _cuts_in(audio, frames, start, end):
    return [frame for frame in frames[1:-1] if start <= frame / audio["fps"] < end]


def test_dense_beats_are_held_through_quiet_intro_then_resume_after_drop():
    audio = _energy_audio(quiet_ranges=((0, 4),))
    boundaries, warnings = choose_boundaries(audio, _config(), max_duration=10)
    metadata = audio["cut_activity"]
    assert metadata["enabled"] is True
    assert metadata["reference_percentile"] == 80
    assert metadata["active_intervals"] and metadata["quiet_intervals"]
    assert not _cuts_in(audio, boundaries, .1, 4.0)
    assert boundaries[1] / FPS >= 4.0
    assert len(_cuts_in(audio, boundaries, 4.0, 10.0)) >= 5
    assert metadata["suppressed_event_frames"]
    assert metadata["eligible_event_frames"]


def test_breakdown_has_no_interior_cuts_until_release_window_is_active():
    audio = _energy_audio(quiet_ranges=((2, 6),))
    boundaries, _ = choose_boundaries(audio, _config(), max_duration=10)
    metadata = audio["cut_activity"]
    assert not _cuts_in(audio, boundaries, 3.0, 5.8)
    assert _cuts_in(audio, boundaries, .5, 2.0)
    assert _cuts_in(audio, boundaries, 6.5, 10.0)
    assert metadata["quiet_intervals"]


def test_quiet_gate_is_relative_to_each_tracks_energy_scale():
    quiet_scale = _energy_audio(quiet_ranges=(), quiet=.1, loud=.1)
    full_scale = _energy_audio(quiet_ranges=(), quiet=1.0, loud=1.0)
    quiet_boundaries, _ = choose_boundaries(quiet_scale, _config(), max_duration=10)
    full_boundaries, _ = choose_boundaries(full_scale, _config(), max_duration=10)
    assert quiet_scale["cut_activity"]["enabled"]
    assert full_scale["cut_activity"]["enabled"]
    assert quiet_boundaries == full_boundaries
    assert len(quiet_boundaries) > 5


def test_isolated_transients_do_not_open_a_quiet_intro_gate():
    audio = _energy_audio(quiet_ranges=((0, 6),))
    # Brief high samples stand in for isolated peaks. A trailing 0.4-second
    # power window should keep their relative average below the active gate.
    times = np.asarray(audio["energy_times"])
    values = np.asarray(audio["energy"])
    for peak in (.5, 1.5, 2.5, 3.5, 4.5, 5.5):
        values[np.argmin(np.abs(times - peak))] = 1.0
    audio["energy"] = values.tolist()
    boundaries, _ = choose_boundaries(audio, _config(), max_duration=10)
    activity = audio["cut_activity"]
    assert not _cuts_in(audio, boundaries, .1, 6.0)
    assert activity["suppressed_event_frames"]


def test_maximum_slot_cap_forces_only_the_needed_quiet_cuts_and_reports_them():
    audio = _energy_audio(quiet_ranges=((0, 4),))
    boundaries, warnings = choose_boundaries(audio, _config(), max_duration=1.5)
    activity = audio["cut_activity"]
    assert activity["relaxed"] is True
    assert activity["forced_cut_frames"]
    assert len(activity["forced_cut_frames"]) <= 2
    assert set(activity["forced_cut_frames"]).issubset(set(boundaries[1:-1]))
    assert all((b - a) / FPS <= 1.5 + 1 / FPS for a, b in zip(boundaries, boundaries[1:]))
    assert any("quiet" in warning.lower() or "forced" in warning.lower() for warning in warnings)


def test_threshold_zero_disables_gate_and_restores_dense_event_cuts():
    audio = _energy_audio(quiet_ranges=((0, 4),))
    cfg = _config(quiet_threshold=0.0)
    boundaries, _ = choose_boundaries(audio, cfg, max_duration=10)
    activity = audio["cut_activity"]
    assert activity["enabled"] is False
    assert activity["suppressed_event_frames"] == []
    assert _cuts_in(audio, boundaries, .5, 4.0)


def _write_quiet_intro_wav(path, duration=8.0, quiet_until=3.0, sample_rate=22_050):
    count = round(duration * sample_rate)
    times = np.arange(count, dtype=np.float32) / sample_rate
    amplitude = np.where(times < quiet_until, .2, .45)
    signal = amplitude * np.sin(2 * np.pi * 110 * times)
    pulse = np.hanning(101).astype(np.float32)
    for beat in np.arange(.25, duration - .1, .5):
        start = round(beat * sample_rate)
        gain = .2 if beat < quiet_until else .7
        stop = min(count, start + len(pulse))
        signal[start:stop] += gain * pulse[:stop-start]
    pcm = np.clip(signal, -1, 1)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(np.round(pcm * 32767).astype("<i2").tobytes())


def test_real_quiet_intro_audio_keeps_beat_grid_but_suppresses_intro_cuts(tmp_path):
    path = tmp_path / "quiet intro then loud beat.wav"
    _write_quiet_intro_wav(path)
    cfg = Config(duration=8.0, fps=30, pacing="fast", beat_sensitivity=2.0,
                 quiet_threshold=.6, quiet_window=.4, quiet_release=.3,
                 cache_dir=str(tmp_path / "quiet audio cache"))
    audio = analyze_audio(path, cfg)
    original_beats = list(audio["beat_times"])
    assert original_beats
    boundaries, _ = choose_boundaries(audio, cfg, max_duration=8.0)
    activity = audio["cut_activity"]
    assert audio["beat_times"] == original_beats
    assert activity["enabled"]
    assert boundaries[1] / FPS >= 3.0
    assert any(abs(boundaries[1] / FPS - beat) <= 1 / FPS for beat in original_beats)
    assert any(frame / FPS < 3.0 for frame in activity["suppressed_event_frames"])


def test_real_cli_holds_quiet_intro_then_uses_distinct_footage(tmp_path):
    fixture_root = tmp_path / "quiet section fixture"
    manifest = generate_media(fixture_root, duration=6.0)
    audio_path = fixture_root / "quiet intro music.wav"
    _write_quiet_intro_wav(audio_path, duration=6.0, quiet_until=2.0)
    clips = fixture_root / "distinct long clips"
    clips.mkdir()
    classes = {"similar-a", "similar-b", "different", "motion-left", "motion"}
    source_specs = [v for v in manifest["videos"] if v["composition"] in classes]
    assert len(source_specs) == 5
    for index, source in enumerate(source_specs):
        shutil.copy2(source["path"], clips / f"distinct source {index}.mp4")

    output = fixture_root / "quiet intro edit.mp4"
    env = os.environ.copy()
    source_path = str(ROOT / "src")
    env["PYTHONPATH"] = source_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    result = subprocess.run(
        [sys.executable, "-m", "autoedit.cli", "create", "--audio", str(audio_path),
         "--clips", str(clips), "--output", str(output), "--duration", "4", "--fps", "24",
         "--pacing", "fast", "--beat-sensitivity", "2", "--cache-dir", str(fixture_root / "cache")],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        timeout=120, check=False)
    assert result.returncode == 0, f"CLI failed ({result.returncode})\n{result.stdout}\n{result.stderr}"

    from autoedit.probe import probe

    plan = json.loads(output.with_suffix(".plan.json").read_text())
    activity = plan["audio"]["cut_activity"]
    interior = plan["cut_frames"][1:-1]
    assert plan["configuration"]["reuse"] == "forbid"
    assert plan["configuration"]["beat_sensitivity"] == 2
    assert activity["enabled"]
    assert activity["suppressed_event_frames"]
    assert any(frame / plan["output"]["fps"] < 2.0 for frame in activity["suppressed_event_frames"])
    assert interior and interior[0] / plan["output"]["fps"] >= 2.0
    assert len({segment["clip"] for segment in plan["segments"]}) == len(plan["segments"])
    info = probe(output)
    assert (info["video"]["width"], info["video"]["height"]) == (240, 160)
    validation = json.loads(output.with_suffix(".validation.json").read_text())
    assert validation["valid"], validation["errors"]
