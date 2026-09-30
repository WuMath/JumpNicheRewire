"""New entry points may read old data but never write old experiment state."""

from __future__ import annotations
from dataclasses import asdict, is_dataclass
from pathlib import Path
import hashlib
import json
import os
from configs.paths import PATHS, RECONSTRUCT_ROOT


def is_within(path, root):
    """Path.is_relative_to equivalent for the Stereopy Python 3.8 runtime."""
    try:
        Path(path).relative_to(Path(root))
        return True
    except ValueError:
        return False


def writable(path):
    target = Path(path).resolve()
    if not any(is_within(target, Path(root).resolve())
               for root in PATHS.allowed_write_roots):
        raise PermissionError(f"Output outside configured write roots: {target}. Edit configs/paths.py explicitly.")
    # Never permit writing original code, even if an overly broad root is set.
    old = PATHS.source_project.resolve()
    if is_within(target, old) and not is_within(target, RECONSTRUCT_ROOT.resolve()):
        raise PermissionError(f"Original project is protected: {target}")
    return target


def jsonable(value):
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def write_json(path, value):
    path = writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.writing')
    temporary.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)
    return path


def digest(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def implementation_signature():
    # Paths/entry action are excluded. Effective parameter values are hashed
    # separately, so switching check/run/evaluate doesn't invalidate results.
    files = list((RECONSTRUCT_ROOT / 'src').rglob('*.py'))
    files += list((RECONSTRUCT_ROOT / 'configs/common').glob('*.py'))
    return digest({p.relative_to(RECONSTRUCT_ROOT).as_posix(): file_hash(p) for p in sorted(files)})
