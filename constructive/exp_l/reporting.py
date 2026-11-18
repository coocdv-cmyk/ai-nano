"""Artifact writing, immutable snapshots, additive manifests, quick isolation."""

from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .determinism import ROOT, fingerprint

RESULTS = ROOT / "constructive" / "exp_l" / "results"
HISTORY = RESULTS / "history"


def now_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")


def write_json(path: Path, obj: Any) -> Path:
    """Atomic write so a crashed run never leaves a half-written artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, default=str)
            fh.write("\n")
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def artifact_name(stage: str, kind: str, quick: bool) -> str:
    return f"{stage}-{kind}-quick.json" if quick else f"{stage}-{kind}.json"


def snapshot(stage: str, block: dict[str, Any], quick: bool = False,
             stamp: str | None = None) -> Path:
    """Write an immutable, timestamped copy of a complete stage block."""
    stamp = stamp or now_stamp()
    name = f"{stage}-quick-{stamp}.json" if quick else f"{stage}-{stamp}.json"
    return write_json(HISTORY / name, block)


def update_latest(stage: str, block: dict[str, Any], quick: bool = False) -> Path:
    latest_path = RESULTS / ("latest-quick.json" if quick else "latest.json")
    latest: dict[str, Any]
    if latest_path.exists():
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
    else:
        latest = {"schema": "exp-l-results-v1", "stages": {}}
    latest.setdefault("stages", {})[stage] = block
    latest["updated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    latest["fingerprint"] = fingerprint(latest["stages"])
    return write_json(latest_path, latest)


def update_integrity(entry: dict[str, Any], quick: bool = False) -> Path:
    """Append/replace a stage entry in the ADDITIVE EXP-L integrity manifest."""
    path = RESULTS / ("integrity-quick.json" if quick else "integrity.json")
    manifest: dict[str, Any]
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "schema": "exp-l-additive-integrity-v1",
            "note": "Additive manifest. Never replaces EXP-I/EXP-J/EXP-K manifests.",
            "stages": {},
        }
    stage = entry.get("stage", "unknown")
    manifest["stages"][stage] = entry
    manifest["fingerprint"] = fingerprint(manifest["stages"])
    return write_json(path, manifest)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
