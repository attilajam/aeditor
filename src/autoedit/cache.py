"""Atomic content-addressed cache; settings and implementation version are part of keys."""
from pathlib import Path
import hashlib
import json
import os
import tempfile

CACHE_VERSION = 1


def key_for(identity, settings, kind):
    payload = {"version": CACHE_VERSION, "source": identity["sha256"], "settings": settings, "kind": kind}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(data, out, indent=2, allow_nan=False)
            out.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_cache(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
