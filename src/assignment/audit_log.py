"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import uuid
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-independent observer; it records but never blocks requests."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}

    @staticmethod
    def _key(user_id: str, request_id: str | None) -> str:
        return request_id or f"{user_id}:{uuid.uuid4().hex}"

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        key = self._key(user_id, request_id)
        started = time.perf_counter()
        self._open[key] = started
        record = {
            "request_id": key,
            "user_id": user_id,
            "input": text or "",
            "started_at": utc_now_iso(),
            "output": None,
            "blocked": None,
            "layer": None,
            "latency_ms": None,
        }
        self.logs.append(record)
        return key

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        key = request_id
        record = next(
            (item for item in reversed(self.logs)
             if item.get("request_id") == key and item.get("user_id") == user_id),
            None,
        ) if key else None

        if record is None:
            key = self._key(user_id, request_id)
            record = {
                "request_id": key,
                "user_id": user_id,
                "input": None,
                "started_at": utc_now_iso(),
            }
            self.logs.append(record)

        started = self._open.pop(key, None)
        record.update({
            "output": text or "",
            "blocked": bool(blocked),
            "layer": layer,
            "completed_at": utc_now_iso(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3)
                if started is not None else None,
        })
        return record

    def export_json(self, filepath: str | None = None):
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
