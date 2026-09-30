"""Atomic manifests, hashes, and restart guards shared by all stages."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def stable_hash(value: Any) -> str:
    payload = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, payload: Any, replace_attempts: int = 12) -> None:
    """Write JSON safely, tolerating transient Windows antivirus/indexer locks."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.writing"
    )
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    last_error: PermissionError | None = None
    for attempt in range(max(int(replace_attempts), 1)):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as error:
            last_error = error
            time.sleep(min(0.05 * (2**attempt), 0.80))
    # A program may have opened the destination without delete sharing.  A
    # direct overwrite can still succeed and is preferable to losing status.
    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(_jsonable(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.unlink(missing_ok=True)
        return
    except PermissionError:
        pass
    raise PermissionError(
        f"Could not update {path} after {replace_attempts} retries. "
        "Close any editor/Excel preview holding this JSON file and rerun; "
        f"the complete recovery payload remains at {temporary}."
    ) from last_error


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def completion_matches(path: Path, signature: str, required: tuple[Path, ...] = ()) -> bool:
    path = Path(path)
    if not path.exists() or any(not Path(item).exists() for item in required):
        return False
    try:
        return read_json(path).get("signature") == signature
    except (OSError, ValueError, TypeError):
        return False


def write_completion(path: Path, signature: str, extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {"signature": signature, "completed_at_utc": utc_now()}
    if extra:
        payload.update(extra)
    write_json_atomic(path, payload)
