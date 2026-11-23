"""Response cache keyed by the exact request, so CI can replay a run without API calls.

Any change to the prompt, model, settings or documents changes the key, which
forces a live call. In --offline mode a missing entry is an error, never a guess.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable

MODES = ("live", "offline", "refresh")


class OfflineMiss(RuntimeError):
    """Raised in offline mode when a request has no cached response."""


def request_key(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, directory: Path, mode: str = "live"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.directory = Path(directory)
        self.mode = mode
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}

    def _lock_for(self, key: str) -> threading.Lock:
        with self._lock:
            lock = self._key_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._key_locks[key] = lock
            return lock

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def _write_atomic(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def fetch(
        self,
        payload: dict[str, Any],
        call: Callable[[dict[str, Any]], dict[str, Any]],
        validate: Callable[[dict[str, Any]], bool] | None = None,
    ) -> dict[str, Any]:
        key = request_key(payload)
        path = self._path(key)
        with self._lock_for(key):
            if self.mode != "refresh" and path.exists():
                cached = json.loads(path.read_text(encoding="utf-8"))["response"]
                if validate is None or validate(cached):
                    with self._lock:
                        self.hits += 1
                    return cached
                if self.mode == "offline":
                    raise OfflineMiss(
                        f"cached response for request {key[:12]} failed validation "
                        f"({payload.get('provider')}, {payload.get('model')})"
                    )
            if self.mode == "offline":
                raise OfflineMiss(f"no cached response for request {key[:12]} ({payload.get('provider')}, {payload.get('model')})")
            with self._lock:
                self.misses += 1
            response = call(payload)
            if validate is None or validate(response):
                record = {"request": payload, "response": response}
                self._write_atomic(path, json.dumps(record, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
            return response
