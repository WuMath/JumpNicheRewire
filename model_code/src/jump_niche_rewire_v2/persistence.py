"""Immutable compact target checkpoints and cooperative resource checks.

The future subprocess supervisor must additionally enforce hard RSS/time limits
during native linear algebra calls. These callbacks alone are NOT a hard limit.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import os
import shutil
import time
import uuid
import numpy as np
import psutil
from .core import prediction_parts


class ResourceGuard:
    def __init__(self, config):
        self.config = config
        self.started = time.monotonic()
        self.last_disk_check = -np.inf
        self.peak_rss_bytes = 0
        self.cumulative_bytes = 0

    def __call__(self, reserve_bytes=0):
        cfg = self.config
        rss = psutil.Process().memory_info().rss
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
        if rss > cfg.maximum_worker_rss_gib*1024**3: raise RuntimeError('resource_limit: RSS')
        if psutil.virtual_memory().available < cfg.minimum_available_ram_gib*1024**3:
            raise RuntimeError('resource_limit: available RAM')
        if time.monotonic()-self.started > cfg.maximum_task_hours*3600: raise RuntimeError('resource_limit: time')
        if reserve_bytes or time.monotonic()-self.last_disk_check > 60:
            # Disjoint configured roots; exclude duplicate/overlapping trees.
            roots = sorted({Path(p).resolve() for p in cfg.budget_roots}, key=lambda p: len(p.parts))
            unique = []
            for root in roots:
                if not any(root == parent or parent in root.parents for parent in unique): unique.append(root)
            total = 0
            for root in unique:
                for folder, _, names in os.walk(root):
                    for name in names:
                        try: total += (Path(folder)/name).stat().st_size
                        except FileNotFoundError: pass  # atomic checkpoint publication can race scanning
            self.cumulative_bytes = total
            self.last_disk_check = time.monotonic()
        if self.cumulative_bytes+reserve_bytes > cfg.maximum_new_gib*1024**3:
            raise RuntimeError('resource_limit: cumulative storage')
        if shutil.disk_usage(Path(cfg.output_root).anchor).free-reserve_bytes < cfg.minimum_free_disk_gib*1024**3:
            raise RuntimeError('resource_limit: disk free')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def write_target_checkpoint(path, design, fitted, identity, numerical_config, guard=None):
    """Call immediately after computation and BEFORE Parquet/table export.

    One target per file, full coefficients/design retained, including failures.
    No pickle; atomic rename; existing completed snapshots cannot be replaced.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(schema='jnr_decomposition_v2_target_1', identity=identity,
                  identity_digest=digest(identity), numerical_config=asdict(numerical_config),
                  design_audit=design.audit, converged=fitted.converged.tolist(),
                  iterations=fitted.iterations, lipschitz=fitted.lipschitz)
    base, local, communication = prediction_parts(design, fitted.backbone, fitted.spatial, fitted.lr)
    arrays = dict(x=design.x, y=design.y, h=design.h, z=design.z, matrix=design.matrix,
                  projected_y=design.projected_y, column_scales=design.column_scales,
                  inactive_columns=design.inactive_columns, nuisance=design.nuisance,
                  nuisance_pinv=design.nuisance_pinv, l1=design.l1, ridge=design.ridge,
                  theta=fitted.theta, backbone=fitted.backbone, spatial=fitted.spatial, lr=fitted.lr,
                  nuisance_coefficients=fitted.nuisance_coefficients, residual=fitted.residual,
                  tolerance=fitted.tolerance, base_prediction=base, spatial_prediction=local,
                  lr_prediction=communication, objective=np.array([r['objective'] for r in fitted.history]),
                  proximal_mapping_inf=np.array([r['proximal_mapping_inf'] for r in fitted.history]))
    if not all(np.isfinite(a).all() for a in arrays.values()): raise ValueError('Non-finite checkpoint arrays')
    # Hash every array, not merely the input signature. This detects damaged or
    # edited numeric values on resume, even if the surrounding zip still opens.
    record['array_sha256'] = {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in arrays.items()}
    record['array_shapes'] = {k: list(v.shape) for k, v in arrays.items()}
    if guard: guard(reserve_bytes=2*sum(a.nbytes for a in arrays.values())+65536)
    lock = path.with_suffix('.lock')
    # Exclusive per-target lock prevents concurrent writers replacing each other.
    with lock.open('x', encoding='utf8') as stream:
        stream.write(json.dumps(dict(pid=os.getpid(), created=psutil.Process().create_time())))
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.writing')
    try:
        if path.exists(): raise FileExistsError(f'Immutable checkpoint already exists: {path}')
        with temporary.open('xb') as stream:
            np.savez_compressed(stream, metadata=np.array(json.dumps(record, ensure_ascii=False)), **arrays)
            stream.flush(); os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        # An interrupted partial is retained for audit; do not broadly clean dirs.
        lock.unlink()
    return path


def load_target_checkpoint(path, identity, numerical_config):
    with np.load(Path(path), allow_pickle=False) as archive:
        record = json.loads(str(archive['metadata']))
        if record['schema'] != 'jnr_decomposition_v2_target_1': raise ValueError('Unknown checkpoint schema')
        if record['identity_digest'] != digest(identity) or record['identity'] != identity:
            raise ValueError('Checkpoint input/implementation/target identity differs')
        if record['numerical_config'] != asdict(numerical_config): raise ValueError('Checkpoint parameters differ')
        expected = set(record['array_sha256'])
        if set(archive.files) != expected | {'metadata'}: raise ValueError('Checkpoint members differ')
        arrays = {name: archive[name].copy() for name in expected}
    for name, array in arrays.items():
        if (not np.isfinite(array).all() or list(array.shape) != record['array_shapes'][name] or
            hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() != record['array_sha256'][name]):
            raise ValueError(f'Checkpoint numeric integrity failed: {name}')
    return record, arrays
