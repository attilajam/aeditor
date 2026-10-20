"""Offline localhost beat review, persistent drafts, and background media jobs.

Only this session's explicitly registered media can be served. No general file
browser or arbitrary command execution is exposed to the web interface.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import secrets
import threading
from urllib.parse import unquote, urlsplit
import wave
import webbrowser

import numpy as np

from .audio import _decode
from .cache import write_json
from .probe import MediaError, fingerprint, require_tools
from .render import load_plan, render_plan, verify_sources


SESSION_VERSION = 1
WEB_ROOT = Path(__file__).with_name("web")


class ReviewError(MediaError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stem(path):
    return path.name.removesuffix(".plan.json") if path.name.endswith(".plan.json") else path.stem


def initial_markers(plan):
    """Stable IDs distinguish measured beats from regular/manual cut positions."""
    if plan.get("user_timing", {}).get("adjustments") is not None:
        return deepcopy(plan["user_timing"]["adjustments"])
    fps, total = plan["output"]["fps"], plan["output"]["total_frames"]
    kept = set(plan["cut_frames"][1:-1])
    events, kind = plan["audio"].get("beat_times", []), "beat"
    if not events and plan["audio"].get("fallback") == "onsets":
        events, kind = plan["audio"].get("onset_times", []), "onset"
    markers, occupied = [], set()
    labels = plan["audio"].get("beat_labels", []) if kind == "beat" else []
    for i, time in enumerate(events):
        frame = round(time*fps)
        if not 0 < frame < total or frame in occupied:
            continue
        occupied.add(frame)
        event_kind = "kick" if kind == "beat" and i < len(labels) and labels[i] else kind
        markers.append({"id": f"{event_kind}-{i}", "kind": event_kind, "detected_time": float(time),
                        "original_frame": frame, "frame": frame, "enabled": frame in kept})
    for frame in sorted(kept-occupied):
        markers.append({"id": f"automatic-{frame}", "kind": "automatic", "detected_time": None,
                        "original_frame": frame, "frame": frame, "enabled": True})
    return sorted(markers, key=lambda marker: (marker["frame"], marker["id"]))


def effective_cuts(markers, ranges, total):
    return [0, *sorted({m["frame"] for m in markers if m["enabled"] and
                       not any(a <= m["frame"] < b for a, b in ranges)}), total]


class ReviewSession:
    def __init__(self, plan_path, session_path=None, video_path=None):
        require_tools()
        self.plan_path = Path(plan_path).expanduser().resolve()
        self.base = load_plan(self.plan_path)
        verify_sources(self.base)
        self.base_identity = fingerprint(self.plan_path)
        self.stem = _stem(self.plan_path)
        self.session_path = (Path(session_path).expanduser().resolve() if session_path else
                             self.plan_path.with_name(self.stem+".review.json"))
        if self.session_path == self.plan_path:
            raise MediaError("review session must not replace the source plan")
        self.work = self.session_path.parent / ".autoedit-review" / (self.session_path.stem+"-"+self.base_identity["sha256"][:10])
        self.work.mkdir(parents=True, exist_ok=True)
        self.original_video = (Path(video_path).expanduser().resolve() if video_path else
                               self.plan_path.with_name(self.stem+".mp4"))
        self.lock = threading.RLock()
        self.token = secrets.token_urlsafe(32)
        self.initial = initial_markers(self.base)
        self.canonical = {m["id"]: m for m in self.initial}
        self.initial_ranges = deepcopy(self.base.get("user_timing", {}).get("no_cut_ranges", []))
        self.markers, self.no_cut_ranges = deepcopy(self.initial), deepcopy(self.initial_ranges)
        self.revision, self.saved_at, self.job, self.exports = 0, _now(), None, []
        self.preview_revision, self.video_key = None, None
        self.media = {}
        self.downloads = {}
        self.worker = None
        self.ignored_draft = None
        if self.original_video.is_file():
            self.media["original.mp4"] = self.original_video
            self.video_key = "original.mp4"
            self.preview_revision = 0
        if self.session_path.exists():
            self._restore()
        self._audio_preview()
        if not self.session_path.exists():
            self._persist()

    def _audio_preview(self):
        """Waveform and playback use one decoded PCM interval, avoiding VBR drift."""
        settings = {"identity": self.base["audio"]["identity"],
                    "start": self.base["audio"]["audio_start"], "duration": self.base["output"]["duration"]}
        key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
        audio_path, peaks_path = self.work / f"audio-{key}.wav", self.work / f"peaks-{key}.json"
        if not audio_path.is_file() or not peaks_path.is_file():
            samples = _decode(settings["identity"]["path"], settings["start"], settings["duration"])
            if not len(samples):
                raise MediaError("no decoded soundtrack for review playback")
            with wave.open(str(audio_path), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(22050)
                stream.writeframes((np.clip(samples, -1, 1)*32767).astype("<i2").tobytes())
            # Approximately 200 peaks/second supplies detail when zooming in.
            block = max(1, int(len(samples)/max(1, min(100_000, round(settings["duration"]*200)))))
            padded = np.pad(np.abs(samples), (0, (-len(samples)) % block))
            write_json(peaks_path, padded.reshape(-1, block).max(axis=1).astype(float).tolist())
        self.peaks = json.loads(peaks_path.read_text())
        self.media["audio.wav"] = audio_path

    def _restore(self):
        try:
            saved = json.loads(self.session_path.read_text())
            if saved.get("version") != SESSION_VERSION or saved.get("base_plan", {}).get("sha256") != self.base_identity["sha256"]:
                # The plan beside this draft was re-created, so its saved cut frames no
                # longer map onto the new timeline; reopen on the plan's own detected
                # cuts instead of refusing to start. The draft file is left untouched.
                self.ignored_draft = self.session_path
                return
            self.markers, self.no_cut_ranges = self._normalize(saved)
            if type(saved["revision"]) is not int or saved["revision"] < 0:
                raise ValueError("invalid saved revision")
            self.revision, self.saved_at = saved["revision"], saved["saved_at"]
            # Register only known filenames under this session's private workdir.
            preview = saved.get("preview")
            if preview and isinstance(preview.get("file"), str) and re.fullmatch(r"preview-[a-f0-9]+\.mp4", preview["file"]):
                path = self.work / preview["file"]
                if path.is_file():
                    self.media[path.name] = path
                    self.video_key, self.preview_revision = path.name, preview["revision"]
            elif self.video_key and effective_cuts(self.markers, self.no_cut_ranges, self.base["output"]["total_frames"]) == self.base["cut_frames"]:
                self.preview_revision = self.revision
            for export in saved.get("exports", []):
                if not isinstance(export.get("filename"), str) or Path(export["filename"]).name != export["filename"]:
                    continue
                path = self.plan_path.parent / export["filename"]
                if path.is_file() and path.suffix.lower() == ".mp4" and path != self.original_video:
                    self.exports.append(self._register_export(path, export.get("revision", 0)))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise MediaError(f"invalid review draft {self.session_path}: {exc}") from exc

    def _normalize(self, payload):
        markers = payload.get("markers")
        ranges = payload.get("no_cut_ranges", [])
        total = self.base["output"]["total_frames"]
        if not isinstance(markers, list) or len(markers) > 10_000:
            raise ReviewError("markers must be a list with at most 10000 entries")
        result, ids, positions = [], set(), set()
        for item in markers:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str) or item["id"] in ids:
                raise ReviewError("markers must have unique string IDs")
            ident, frame, enabled = item["id"], item.get("frame"), item.get("enabled")
            if type(frame) is not int or not 0 < frame < total or type(enabled) is not bool:
                raise ReviewError(f"marker {ident}: frame must be an integer inside the timeline and enabled a boolean")
            if ident in self.canonical:
                marker = deepcopy(self.canonical[ident])
            elif re.fullmatch(r"manual-[A-Za-z0-9_-]{1,80}", ident):
                original = item.get("original_frame", frame)
                if type(original) is not int or not 0 < original < total:
                    raise ReviewError("invalid manual marker origin")
                marker = {"id": ident, "kind": "manual", "detected_time": None, "original_frame": original}
            else:
                raise ReviewError(f"unknown marker ID: {ident}")
            marker.update(frame=frame, enabled=enabled)
            if enabled and frame in positions:
                raise ReviewError(f"two enabled cuts occupy frame {frame}; move or disable one")
            if enabled:
                positions.add(frame)
            result.append(marker)
            ids.add(ident)
        if not set(self.canonical).issubset(ids):
            raise ReviewError("detected markers must be retained; disable unwanted markers instead of deleting them")
        if not isinstance(ranges, list) or len(ranges) > 1000:
            raise ReviewError("no_cut_ranges must be a list with at most 1000 entries")
        normalized_ranges = []
        for bounds in ranges:
            if not isinstance(bounds, list) or len(bounds) != 2 or any(type(f) is not int for f in bounds):
                raise ReviewError("no-cut ranges require two integer frame positions")
            a, b = bounds
            if not 0 <= a < b <= total:
                raise ReviewError("no-cut range must start before its end inside the timeline")
            normalized_ranges.append([a, b])
        return sorted(result, key=lambda m: (m["frame"], m["id"])), sorted(normalized_ranges)

    def _persist(self):
        preview = None
        if self.video_key and self.video_key != "original.mp4":
            preview = {"file": self.video_key, "revision": self.preview_revision}
        write_json(self.session_path, {"version": SESSION_VERSION, "base_plan": self.base_identity,
                   "revision": self.revision, "saved_at": self.saved_at, "markers": self.markers,
                   "no_cut_ranges": self.no_cut_ranges, "preview": preview,
                   "exports": [{"filename": e["filename"], "revision": e["revision"]} for e in self.exports]})

    def _require_revision(self, payload):
        if type(payload.get("revision")) is not int or payload["revision"] != self.revision:
            raise ReviewError("This draft changed in another tab. Reload before editing it.", 409)

    def state(self):
        with self.lock:
            out = self.base["output"]
            return deepcopy({"title": self.stem, "plan_name": self.plan_path.name,
                "fps": out["fps"], "total_frames": out["total_frames"], "duration": out["duration"], "output": out,
                "markers": self.markers, "no_cut_ranges": self.no_cut_ranges,
                "cut_frames": effective_cuts(self.markers, self.no_cut_ranges, out["total_frames"]),
                "quiet_intervals": self.base["audio"].get("cut_activity", {}).get("quiet_intervals", []),
                "onset_times": self.base["audio"].get("onset_times", []), "peaks": self.peaks,
                "audio_url": "/media/audio.wav", "video_url": "/media/"+self.video_key if self.video_key else None,
                "preview_revision": self.preview_revision, "revision": self.revision,
                "saved_at": self.saved_at, "job": self.job, "exports": self.exports, "csrf_token": self.token,
                "session_path": str(self.session_path), "export_filename": self.stem+".reviewed.mp4"})

    def save(self, payload):
        with self.lock:
            self._require_revision(payload)
            markers, ranges = self._normalize(payload)
            if markers != self.markers or ranges != self.no_cut_ranges:
                self.markers, self.no_cut_ranges = markers, ranges
                self.revision += 1
                self.saved_at = _now()
                self._persist()
            return self.state()

    def reset(self, payload):
        return self.save({**payload, "markers": deepcopy(self.initial), "no_cut_ranges": deepcopy(self.initial_ranges)})

    def _register_export(self, output, revision):
        links = {"filename": output.name, "revision": revision, "output": str(output)}
        for kind, path in {"download": output, "plan": output.with_suffix(".plan.json"),
                           "validation": output.with_suffix(".validation.json"), "report": output.with_suffix(".report.html")}.items():
            if path.is_file():
                key = hashlib.sha256(str(path).encode()).hexdigest()[:20]+path.suffix
                self.downloads[key] = path
                links[kind+"_url"] = "/download/"+key
        return links

    def _export_path(self, payload):
        name = payload.get("filename") or self.stem+".reviewed.mp4"
        if not isinstance(name, str) or len(name) > 160 or Path(name).name != name or "\\" in name or not name.lower().endswith(".mp4") or name.startswith(".") or any(ord(c) < 32 for c in name):
            raise ReviewError("export filename must be a plain .mp4 filename in the plan directory")
        output = self.plan_path.parent / name
        protected = {self.original_video, self.plan_path, self.session_path}
        protected.update(Path(s["identity"]["path"]).resolve() for s in self.base["sources"])
        protected.add(Path(self.base["audio"]["identity"]["path"]).resolve())
        destinations = [output, *(output.with_suffix(s) for s in (".plan.json", ".report.html", ".validation.json"))]
        if any(path.resolve() in protected for path in destinations):
            raise ReviewError("choose a new export filename; the original edit, plan and inputs are protected")
        if not payload.get("overwrite", False) and any(p.exists() for p in destinations):
            raise ReviewError(f"destination already exists: {output.name}; choose a new name or enable replacement", 409)
        return output

    def start_job(self, kind, payload):
        with self.lock:
            self._require_revision(payload)
            if kind not in {"preview", "export"}:
                raise ReviewError("unknown render job")
            if self.job and self.job["status"] in {"queued", "running"}:
                raise ReviewError("a render is already running; wait for it to finish", 409)
            output = self._export_path(payload) if kind == "export" else None
            snapshot = {"markers": deepcopy(self.markers), "no_cut_ranges": deepcopy(self.no_cut_ranges),
                        "revision": self.revision, "overwrite": bool(payload.get("overwrite", False))}
            self.job = {"id": secrets.token_hex(8), "kind": kind, "status": "queued", "message": "Preparing selection",
                        "revision": self.revision}
            self.worker = threading.Thread(target=self._run_job, args=(snapshot, output), daemon=True,
                                           name="autoedit-review-render")
            self.worker.start()
            return {"job": deepcopy(self.job)}

    def _run_job(self, snapshot, output):
        from . import __version__
        from .report import write_report
        from .review_planner import build_review_plan
        try:
            with self.lock:
                self.job.update(status="running", message="Matching source clips to your cut choices")
            verify_sources(self.base)
            digest = hashlib.sha256(json.dumps({"base": self.base_identity["sha256"], "version": __version__,
                "markers": snapshot["markers"], "ranges": snapshot["no_cut_ranges"]}, sort_keys=True).encode()).hexdigest()
            selection_file = self.work / ("selection-"+digest[:20]+".plan.json")
            if selection_file.is_file():
                plan = load_plan(selection_file)
            else:
                cuts = effective_cuts(snapshot["markers"], snapshot["no_cut_ranges"], self.base["output"]["total_frames"])
                plan = build_review_plan(self.base, cuts, adjustments=snapshot["markers"], no_cut_ranges=snapshot["no_cut_ranges"])
                write_json(selection_file, plan)
            preview = output is None
            if preview:
                plan = deepcopy(plan)
                width, height = plan["output"]["width"], plan["output"]["height"]
                factor = min(1., 640/max(width, height))
                width, height = max(16, round(width*factor/2)*2), max(16, round(height*factor/2)*2)
                plan["preview_of"] = deepcopy(plan["output"])
                plan["configuration"].update(size=f"{width}x{height}", preset="ultrafast", crf=22)
                plan["output"].update(width=width, height=height)
                output = self.work / ("preview-"+digest[:20]+".mp4")
            with self.lock:
                self.job["message"] = "Rendering and validating "+("preview" if preview else "final export")
            # A cached preview was already fully validated when it was published.
            if preview and output.is_file() and output.with_suffix(".validation.json").is_file():
                validation = json.loads(output.with_suffix(".validation.json").read_text())
            else:
                validation = render_plan(plan, output, overwrite=preview or snapshot["overwrite"])
                write_json(output.with_suffix(".plan.json"), plan)
                write_report(plan, output, validation, output.with_suffix(".report.html"))
            with self.lock:
                self.media[output.name] = output
                links = {"video_url": "/media/"+output.name}
                if preview:
                    self.video_key, self.preview_revision = output.name, snapshot["revision"]
                else:
                    export = self._register_export(output, snapshot["revision"])
                    self.exports = [e for e in self.exports if e["filename"] != output.name]+[export]
                    links.update(export)
                self.job.update({"status": "completed", "message": "Preview ready" if preview else "Export ready",
                                 "output": str(output), **links})
                self._persist()
        except Exception as exc:
            with self.lock:
                self.job.update(status="failed", message="Could not create this edit", error=str(exc))


class ReviewHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


def create_server(session, host="127.0.0.1", port=0):
    if host not in {"127.0.0.1", "localhost"}:
        raise MediaError("the review server only binds to localhost")
    server = ReviewHTTPServer((host, port), ReviewHandler)
    server.session = session
    return server


class ReviewHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError, TimeoutError):
            # Browsers cancel range requests when the playhead changes.
            pass

    @property
    def session(self):
        return self.server.session

    def log_message(self, format, *args):
        # Polling and media seeks are deliberately quiet; job failures are in state.
        pass

    def _headers(self, status, content_type, length, extra=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()

    def _json(self, status, payload):
        raw = json.dumps(payload, allow_nan=False).encode()
        self._headers(status, "application/json; charset=utf-8", len(raw))
        if self.command != "HEAD":
            self.wfile.write(raw)

    def _valid_host(self):
        port = self.server.server_port
        return self.headers.get("Host") in {f"127.0.0.1:{port}", f"localhost:{port}"}

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        try:
            if not self._valid_host():
                raise ReviewError("invalid local host", 403)
            path = unquote(urlsplit(self.path).path)
            if path == "/api/state":
                self._json(200, self.session.state())
                return
            if path in {"/", "/index.html"}:
                return self._file(WEB_ROOT / "index.html")
            if path.startswith("/assets/"):
                candidate = (WEB_ROOT / path.removeprefix("/assets/")).resolve()
                if not candidate.is_relative_to(WEB_ROOT.resolve()):
                    raise ReviewError("not found", 404)
                return self._file(candidate)
            if path.startswith("/media/"):
                with self.session.lock:
                    candidate = self.session.media.get(path.removeprefix("/media/"))
                if candidate:
                    return self._file(candidate)
            if path.startswith("/download/"):
                with self.session.lock:
                    candidate = self.session.downloads.get(path.removeprefix("/download/"))
                if candidate:
                    return self._file(candidate, download=True)
            raise ReviewError("not found", 404)
        except (ReviewError, OSError) as exc:
            self._json(getattr(exc, "status", 404), {"error": str(exc)})

    def _file(self, path, download=False):
        if not path.is_file():
            raise ReviewError("not found", 404)
        size = path.stat().st_size
        start, end, status = 0, size-1, 200
        headers = {"Accept-Ranges": "bytes"}
        value = self.headers.get("Range")
        if value:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", value)
            if not match or not any(match.groups()):
                self._headers(416, "text/plain", 0, {"Content-Range": f"bytes */{size}"})
                return
            a, b = match.groups()
            if a:
                start, end = int(a), min(size-1, int(b)) if b else size-1
            else:
                start, end = max(0, size-int(b)), size-1
            if start > end or start >= size or start < 0:
                self._headers(416, "text/plain", 0, {"Content-Range": f"bytes */{size}"})
                return
            status = 206
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        if download:
            # RFC 5987 keeps unicode filenames out of raw HTTP headers.
            from urllib.parse import quote
            headers["Content-Disposition"] = "attachment; filename*=UTF-8''"+quote(path.name)
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix == ".js":
            content_type = "text/javascript; charset=utf-8"
        self._headers(status, content_type, max(0, end-start+1), headers)
        if self.command == "HEAD":
            return
        try:
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = end-start+1
                while remaining > 0:
                    block = stream.read(min(256*1024, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        try:
            if not self._valid_host():
                raise ReviewError("invalid local host", 403)
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}:
                raise ReviewError("only the local editor may change this session", 403)
            if not secrets.compare_digest(self.headers.get("X-Autoedit-Token", ""), self.session.token):
                raise ReviewError("missing or invalid local editor token", 403)
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 2_000_000:
                raise ReviewError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ReviewError("request must be a JSON object")
            path = urlsplit(self.path).path
            if path == "/api/session":
                self._json(200, self.session.save(payload))
            elif path == "/api/reset":
                self._json(200, self.session.reset(payload))
            elif path in {"/api/preview", "/api/export"}:
                self._json(202, self.session.start_job(path.rsplit("/", 1)[1], payload))
            else:
                raise ReviewError("not found", 404)
        except (MediaError, ValueError, TypeError, KeyError) as exc:
            # Drain/close failed requests, so unread bytes cannot become another request.
            self.close_connection = True
            self._json(getattr(exc, "status", 400), {"error": str(exc)})


def serve_review(plan_path, *, session_path=None, video_path=None, port=8765, open_browser=True):
    session = ReviewSession(plan_path, session_path, video_path)
    try:
        server = create_server(session, port=port)
    except OSError as exc:
        raise MediaError(f"cannot start review server on port {port}: {exc}; choose another --port") from exc
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Beat editor: {url}\nDraft: {session.session_path}\nPress Ctrl+C to stop.", flush=True)
    if session.ignored_draft:
        print(f"Note: {session.ignored_draft} was saved for a different plan; starting from its detected cuts.", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        print("\nReview draft saved.", flush=True)
    finally:
        server.server_close()
    return 0
