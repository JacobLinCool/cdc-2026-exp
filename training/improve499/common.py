"""Paths, fingerprints, and atomic records for the bounded 499 experiment."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "training/scaling"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def plan():
    return json.loads((FOLDER / "experiment-plan.json").read_text())


def fingerprints(paths):
    return {relative(p): digest(p) for p in sorted(set(paths))}


def verify(hashes):
    for name, expected in hashes.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"frozen file changed: {name}")


def freeze(path, hashes, **metadata):
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["hashes"] != hashes:
            raise ValueError(f"frozen inputs changed: {path}")
        return previous
    result = dict(created_utc=timestamp(), hashes=hashes, **metadata)
    save(path, result)
    return result


def engine_sources():
    return [
        *ROOT.glob("engine/src/**/*.rs"),
        ROOT / "engine/Cargo.toml",
        ROOT / "engine/Cargo.lock",
    ]
