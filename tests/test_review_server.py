"""Integration checks for the local beat-review HTTP server."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from generate_test_media import generate_media
from autoedit.review import ReviewSession, create_server
from autoedit.render import decoded_frame_count


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def review_plan(tmp_path_factory):
    """Build one small, real plan shared by the HTTP and render checks."""
    root = tmp_path_factory.mktemp("review media")
    media = generate_media(root / "fixtures", duration=4.0)
    source = Path(next(v["path"] for v in media["videos"] if v["composition"] == "similar-a"))
    clips = root / "clips with spaces"
    clips.mkdir()
    shutil.copy2(source, clips / "source.mp4")
    output = root / "source edit.mp4"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, "-m", "autoedit.cli", "create",
               "--audio", media["audio"]["rhythmic"], "--clips", str(clips),
               "--output", str(output), "--size", "160x160", "--fps", "12",
               "--duration", "3", "--audio-start", "0.25", "--pacing", "fast", "--reuse", "allow",
               "--seed", "17", "--cache-dir", str(root / "cache")]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"fixture plan build failed:\n{result.stdout}\n{result.stderr}"
    plan = output.with_suffix(".plan.json")
    assert plan.is_file()
    return root, plan, output, media


@pytest.fixture
def live_server(review_plan, tmp_path):
    _, plan, _, _ = review_plan
    session = ReviewSession(plan, session_path=tmp_path / "review-session.json")
    server = create_server(session)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield session, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        worker = session.worker
        if worker is not None:
            worker.join(timeout=90)
            assert not worker.is_alive(), "review render worker did not stop before fixture teardown"
        server.server_close()
        thread.join(timeout=5)


def _request(base, path, method="GET", body=None, token=None, origin=None, headers=None):
    data = None if body is None else json.dumps(body).encode()
    request_headers = dict(headers or {})
    if data is not None:
        request_headers["Content-Type"] = "application/json"
    if token is not None:
        request_headers["X-Autoedit-Token"] = token
    if origin is not None:
        request_headers["Origin"] = origin
    request = urllib.request.Request(base + path, data=data, method=method, headers=request_headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers, exc.read()


def _state(base):
    status, _, data = _request(base, "/api/state")
    assert status == 200
    return json.loads(data)


def _post(base, path, state, **extra):
    return _request(base, path, "POST", {"revision": state["revision"], **extra},
                    token=state["csrf_token"])


def _wait_job(base, job_id, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = _state(base)
        job = state.get("job")
        if job and job.get("id") == job_id and job.get("status") in {"completed", "failed"}:
            return state, job
        time.sleep(.1)
    pytest.fail(f"review job {job_id} did not finish within {timeout}s")


def test_state_and_offline_assets_are_served(live_server):
    _, base = live_server
    state = _state(base)
    assert {"title", "plan_name", "fps", "total_frames", "duration", "markers",
            "no_cut_ranges", "audio_url", "video_url", "revision", "csrf_token"} <= state.keys()
    assert state["markers"]
    status, headers, html = _request(base, "/")
    assert status == 200
    assert "text/html" in headers.get("Content-Type", "")
    assert html
    references = re.findall(rb'(?:src|href)=["\']([^"\']+)', html)
    local_assets = [ref.decode("utf-8") for ref in references if ref.startswith(b"/")]
    assert not any(ref.startswith(("http://", "https://")) for ref in
                   (ref.decode("utf-8") for ref in references)), "review UI must not require remote assets"
    for path in local_assets:
        status, _, data = _request(base, path)
        assert status == 200, path
        assert data


def test_media_endpoints_support_byte_ranges(live_server):
    _, base = live_server
    state = _state(base)
    for path in (state["audio_url"], state["video_url"]):
        status, headers, data = _request(base, path, headers={"Range": "bytes=0-127"})
        assert status == 206
        assert headers.get("Content-Range", "").startswith("bytes 0-127/")
        assert len(data) == 128


def test_mutations_require_token_and_same_origin(live_server):
    _, base = live_server
    state = _state(base)
    body = {"revision": state["revision"], "markers": state["markers"],
            "no_cut_ranges": state["no_cut_ranges"]}
    status, _, _ = _request(base, "/api/session", "POST", body)
    assert status == 403
    status, _, _ = _request(base, "/api/session", "POST", body,
                            token=state["csrf_token"], origin="https://attacker.invalid")
    assert status == 403


def test_download_path_traversal_is_rejected(live_server):
    _, base = live_server
    for path in ("/download/..%2f..%2fREADME.md", "/download/%2e%2e%2fREADME.md"):
        status, _, _ = _request(base, path)
        assert status == 404


def test_nudges_disable_and_no_cut_ranges_persist_across_restart(live_server, review_plan, tmp_path):
    _, base = live_server
    session, _ = live_server
    state = _state(base)
    markers = [dict(marker) for marker in state["markers"]]
    marker = markers[0]
    original_frame = marker["frame"]
    marker["frame"] = original_frame + 2
    marker["enabled"] = False
    no_cut_ranges = [[3, 8]]
    status, _, payload = _post(base, "/api/session", state,
                               markers=markers, no_cut_ranges=no_cut_ranges)
    assert status == 200
    saved = json.loads(payload)
    assert saved["markers"][0]["frame"] == original_frame + 2
    assert saved["markers"][0]["enabled"] is False
    assert saved["no_cut_ranges"] == no_cut_ranges
    assert session.session_path.exists()

    restarted = ReviewSession(review_plan[1], session_path=session.session_path)
    state2 = restarted.state()
    assert state2["markers"][0]["frame"] == original_frame + 2
    assert state2["markers"][0]["enabled"] is False
    assert state2["no_cut_ranges"] == no_cut_ranges


def test_keep_only_kind_save_reconstructs_kind_filtered_cuts(live_server):
    # Mirrors the UI's "keep only kicks/beats" action: enable exactly one kind.
    _, base = live_server
    state = _state(base)
    kinds = {marker["kind"] for marker in state["markers"]}
    target = "kick" if "kick" in kinds else "beat"
    assert target in kinds, f"no detected markers to filter, kinds={sorted(kinds)}"
    markers = [dict(marker, enabled=(marker["kind"] == target)) for marker in state["markers"]]
    status, _, payload = _post(base, "/api/session", state,
                               markers=markers, no_cut_ranges=state["no_cut_ranges"])
    assert status == 200, payload[:500]
    saved = json.loads(payload)
    assert all(marker["enabled"] == (marker["kind"] == target) for marker in saved["markers"])
    ranges = state["no_cut_ranges"]
    expected = [0, *sorted(marker["frame"] for marker in saved["markers"]
                           if marker["enabled"] and not any(a <= marker["frame"] < b for a, b in ranges)),
                saved["cut_frames"][-1]]
    assert saved["cut_frames"] == expected


def test_stale_revision_and_invalid_marker_are_rejected(live_server):
    _, base = live_server
    state = _state(base)
    bad = {"revision": state["revision"] - 1, "markers": state["markers"],
           "no_cut_ranges": state["no_cut_ranges"]}
    status, _, _ = _request(base, "/api/session", "POST", bad, token=state["csrf_token"])
    assert status == 409
    markers = [dict(marker) for marker in state["markers"]]
    markers[0]["frame"] = -1
    status, _, _ = _post(base, "/api/session", state, markers=markers,
                         no_cut_ranges=state["no_cut_ranges"])
    assert status in {400, 422}


def test_reset_restores_detected_frame_schedule(live_server):
    _, base = live_server
    initial = _state(base)
    changed_markers = [dict(marker) for marker in initial["markers"]]
    changed_markers[0]["frame"] += 1
    status, _, data = _post(base, "/api/session", initial,
                            markers=changed_markers, no_cut_ranges=initial["no_cut_ranges"])
    assert status == 200
    changed = json.loads(data)
    status, _, data = _post(base, "/api/reset", changed)
    assert status == 200
    reset = json.loads(data)
    assert [(marker["frame"], marker["enabled"]) for marker in reset["markers"]] == [
        (marker["original_frame"], marker["original_frame"] in initial["cut_frames"])
        for marker in reset["markers"]]
    assert reset["no_cut_ranges"] == []


def test_changed_and_missing_sources_fail_cleanly(review_plan, tmp_path):
    _, plan_path, _, _ = review_plan
    original = json.loads(plan_path.read_text())
    audio_path = Path(original["audio"]["identity"]["path"])
    changed_audio = tmp_path / "changed.wav"
    shutil.copy2(audio_path, changed_audio)
    changed_audio.write_bytes(changed_audio.read_bytes() + b"changed")
    changed = json.loads(json.dumps(original))
    changed["audio"]["identity"]["path"] = str(changed_audio)
    changed_path = tmp_path / "changed.plan.json"
    changed_path.write_text(json.dumps(changed))
    with pytest.raises(Exception, match="changed|identity|source"):
        ReviewSession(changed_path)

    missing = json.loads(json.dumps(original))
    missing["sources"][0]["identity"]["path"] = str(tmp_path / "missing.mp4")
    missing_path = tmp_path / "missing.plan.json"
    missing_path.write_text(json.dumps(missing))
    with pytest.raises(Exception, match="missing|source|identity"):
        ReviewSession(missing_path)


def test_preview_and_export_render_fixed_frame_schedule(live_server, review_plan):
    _, base = live_server
    _, _, source_output, _ = review_plan
    source_plan_bytes = review_plan[1].read_bytes()
    state = _state(base)
    original_cuts = state["cut_frames"]
    markers = [dict(marker) for marker in state["markers"]]
    # Move a selected beat by three frames; the exported plan must retain that
    # edited integer-frame schedule and preserve the fixed output frame count.
    moved = next(marker for marker in markers if marker["enabled"])
    moved["frame"] += 3
    status, _, data = _post(base, "/api/session", state, markers=markers,
                            no_cut_ranges=state["no_cut_ranges"])
    assert status == 200
    state = json.loads(data)
    assert state["cut_frames"] != original_cuts
    status, _, data = _post(base, "/api/preview", state)
    assert status == 202
    preview_id = json.loads(data)["job"]["id"]
    preview_state, preview_job = _wait_job(base, preview_id)
    assert preview_job["status"] == "completed", preview_job
    preview_url = preview_job.get("video_url")
    assert preview_url
    status, _, data = _request(base, preview_url)
    assert status == 200 and data[4:8] == b"ftyp"

    status, _, data = _post(base, "/api/export", preview_state, filename="reviewed.mp4", overwrite=False)
    assert status == 202
    export_id = json.loads(data)["job"]["id"]
    exported_state, export_job = _wait_job(base, export_id)
    assert export_job["status"] == "completed", export_job
    export_url = export_job.get("download_url")
    assert export_url
    status, _, video = _request(base, export_url)
    assert status == 200
    output_path = review_plan[0] / "reviewed.mp4"
    output_path.write_bytes(video)
    assert decoded_frame_count(output_path) == state["total_frames"]
    plan_url = export_job.get("plan_url")
    assert plan_url
    status, _, rendered_plan_bytes = _request(base, plan_url)
    assert status == 200
    rendered_plan = json.loads(rendered_plan_bytes)
    assert rendered_plan["output"]["total_frames"] == state["total_frames"]
    assert rendered_plan["cut_frames"] == state["cut_frames"]
    assert review_plan[1].read_bytes() == source_plan_bytes
    assert source_output.exists()

    status, _, _ = _post(base, "/api/export", exported_state, filename="reviewed.mp4", overwrite=False)
    assert status in {400, 409}
    status, _, body = _post(base, "/api/export", exported_state, filename="reviewed.mp4", overwrite=True)
    assert status == 202, body.decode(errors="replace")
    overwrite_id = json.loads(body)["job"]["id"]
    _, overwrite_job = _wait_job(base, overwrite_id)
    assert overwrite_job["status"] == "completed", overwrite_job


def test_only_one_job_can_run_at_a_time(live_server):
    _, base = live_server
    state = _state(base)
    status, _, data = _post(base, "/api/preview", state)
    assert status == 202
    job_id = json.loads(data)["job"]["id"]
    state = _state(base)
    status, _, _ = _post(base, "/api/preview", state)
    assert status in {409, 423}
    _, job = _wait_job(base, job_id)
    assert job["status"] == "completed", job


def test_stale_draft_reopens_on_the_plans_own_cuts(review_plan, tmp_path):
    _, plan_path, _, _ = review_plan
    session_path = tmp_path / "stale-session.json"
    original = {"version": 1, "base_plan": {"sha256": "0" * 64}, "revision": 22,
                "saved_at": "2026-01-01T00:00:00+00:00", "markers": [], "no_cut_ranges": []}
    session_path.write_text(json.dumps(original))
    session = ReviewSession(plan_path, session_path=session_path)
    # A draft saved for a re-created plan must not stop the editor from opening.
    assert session.ignored_draft == session_path
    state = session.state()
    assert state["revision"] == 0
    assert state["markers"] == session.initial
    assert session_path.read_text() == json.dumps(original)


def test_render_failure_is_reported_and_preserves_source_plan(review_plan, tmp_path):
    _, plan_path, _, _ = review_plan
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    source_path = Path(plan["sources"][0]["identity"]["path"])
    source_bytes = source_path.read_bytes()
    session = ReviewSession(plan_path, session_path=tmp_path / "failure-session.json")
    server = create_server(session)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        state = _state(f"http://127.0.0.1:{server.server_port}")
        source_path.write_bytes(source_bytes + b"deliberately changed after session load")
        status, _, data = _post(f"http://127.0.0.1:{server.server_port}", "/api/preview", state)
        assert status == 202
        job_id = json.loads(data)["job"]["id"]
        failed_state, job = _wait_job(f"http://127.0.0.1:{server.server_port}", job_id)
        assert job["status"] == "failed"
        assert job.get("error") or job.get("message")
        assert failed_state["job"]["status"] == "failed"
        assert plan_path.read_bytes() == plan_bytes
    finally:
        source_path.write_bytes(source_bytes)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

