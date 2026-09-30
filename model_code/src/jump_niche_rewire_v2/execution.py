"""Per-target execution and complete exports. Public run gate remains locked.

This module does not load labels or choose candidates. The future registered
workflow supplies already verified Prepared inputs. Unit fixtures are allowed.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import os
import uuid
import numpy as np
from .adapter import target_design, sha
from .core import fit
from .persistence import digest, write_target_checkpoint, load_target_checkpoint


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.writing')
    with temporary.open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def implementation_signature():
    paths = sorted(Path(__file__).parent.glob('*.py'))
    return digest({p.name: sha(p) for p in paths})


def prepared_signature(prepared):
    arrays = {name: getattr(prepared, name) for name in ('standardized', 'covariates', 'spatial_basis', 'exposure')}
    arrays.update(graph_data=prepared.graph.data, graph_indices=prepared.graph.indices, graph_indptr=prepared.graph.indptr)
    values = {k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest())
              for k, v in arrays.items()}
    return digest(dict(arrays=values, cells=prepared.cells, genes=prepared.genes, tfs=prepared.tfs,
                       targets=prepared.targets, lr_ids=prepared.lr_ids, lr_members=prepared.lr_members,
                       covariate_names=prepared.covariate_names, basis_names=prepared.basis_names,
                       audit=prepared.audit))


def target_path(root, index, target):
    return Path(root)/'numerical_targets'/f'{index:05d}_{hashlib.sha256(target.encode()).hexdigest()[:16]}.npz'


def checkpoint_identity(contract, target):
    return dict(contract_sha256=digest(contract), target=target,
                implementation_hash=contract['implementation_hash'], input_signature=contract['input_signature'])


def _validate_checkpoint(record, arrays, target, tfs):
    if record['design_audit']['target'] != target or record['design_audit']['tfs'] != tfs:
        raise ValueError('Checkpoint target/TF mapping differs')
    if arrays['backbone'].shape != (len(tfs), 1): raise ValueError('Backbone shape differs')
    finite = np.isfinite(arrays['residual']) & np.isfinite(arrays['tolerance'])
    actual = finite & (arrays['residual'] <= arrays['tolerance'])
    if actual.tolist() != record['converged']: raise ValueError('Convergence flag inconsistent with residual')
    return bool(actual.all())


def export_global(prepared, numerical, root, contract, guard=None):
    """One small TF block per target, streamed to the FULL GRN Parquet.

    Re-export can run from intact checkpoints after failed file output. It
    never invokes fit and does not build a list of millions of row dictionaries.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = Path(root)
    output = root/'global_grn.parquet'
    marker = root/'export_manifest.json'
    if marker.exists():
        old = json.loads(marker.read_text(encoding='utf8'))
        if old['contract_sha256'] != digest(contract) or not output.exists() or sha(output) != old['sha256']:
            raise ValueError('Export cache identity/integrity failed')
        return old
    temporary = output.with_name(output.name+'.'+uuid.uuid4().hex+'.writing')
    schema = pa.schema([('tf', pa.string()), ('target', pa.string()), ('score', pa.float64()),
                        ('backbone', pa.float64()), ('target_converged', pa.bool_())])
    status, count = [], 0
    with pq.ParquetWriter(temporary, schema, compression='zstd') as writer:
        for index, target in enumerate(prepared.targets):
            if guard: guard(reserve_bytes=65536)
            record, arrays = load_target_checkpoint(target_path(root, index, target), checkpoint_identity(contract, target), numerical)
            accepted = _validate_checkpoint(record, arrays, target, prepared.tfs)
            backbone = arrays['backbone'][:, 0]
            writer.write_table(pa.Table.from_pydict(dict(tf=prepared.tfs, target=[target]*len(prepared.tfs),
                score=np.abs(backbone), backbone=backbone, target_converged=[accepted]*len(prepared.tfs)), schema=schema))
            count += len(backbone)
            status.append(dict(target=target, converged=accepted, iterations=record['iterations'],
                proximal_mapping_inf=float(arrays['residual'][0]), tolerance=float(arrays['tolerance'][0])))
    if count != len(prepared.tfs)*len(prepared.targets): raise ValueError('Incomplete candidate export')
    # A previously published orphan export is retained, not silently replaced.
    if output.exists():
        if sha(output) != sha(temporary): raise ValueError('Untracked previous export differs; preserve and audit')
        temporary.unlink()  # only our redundant, fully validated temporary
    else:
        temporary.replace(output)
    manifest = dict(contract_sha256=digest(contract), sha256=sha(output), rows=count,
        global_readout='absolute_mean_backbone', full_candidate_universe=True,
        convergence_rate=sum(r['converged'] for r in status)/len(status), targets=status,
        accepted=all(r['converged'] for r in status))
    atomic_json(marker, manifest)
    return manifest


def fit_prepared(prepared, numerical, adapter_config, output_root, job_identity, guard=None):
    """Preserve all target checkpoints, resume without refitting saved targets.

    Admission to a scientific run belongs to the still-locked public workflow;
    this low-level executor only enforces numerical/input/code reproducibility.
    """
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    # Lists round-trip through JSON; tuple fields in dataclasses are normalized.
    contract = json.loads(json.dumps(dict(schema='jnr_v2_execution_1', job=job_identity,
        numerical=asdict(numerical), adapter=asdict(adapter_config),
        input_signature=prepared_signature(prepared), implementation_hash=implementation_signature(),
        cells=prepared.cells, genes=prepared.genes, tfs=prepared.tfs, targets=prepared.targets,
        global_readout='absolute_mean_backbone', input_audit=prepared.audit)))
    contract_file = root/'execution_contract.json'
    if contract_file.exists():
        if json.loads(contract_file.read_text(encoding='utf8')) != contract:
            raise ValueError('Input/config/implementation changed; preserve this output and start a new version')
    else:
        atomic_json(contract_file, contract)
    fitted_count, reused_count, convergence_flags = 0, 0, []
    for index, target in enumerate(prepared.targets):
        if guard: guard()
        path = target_path(root, index, target)
        identity = checkpoint_identity(contract, target)
        if path.exists():
            record, arrays = load_target_checkpoint(path, identity, numerical)
            accepted = _validate_checkpoint(record, arrays, target, prepared.tfs)
            reused_count += 1
            del arrays
        else:
            design = target_design(prepared, target, numerical)
            result = fit(design, numerical, resource_callback=guard)
            write_target_checkpoint(path, design, result, identity, numerical, guard)
            accepted = bool(result.converged.all())
            fitted_count += 1
            del design, result
        convergence_flags.append(accepted)
        atomic_json(root/'target_progress.json', dict(target=target, target_index=index,
            targets_total=len(prepared.targets), target_converged=accepted,
            fitted_this_attempt=fitted_count, reused_this_attempt=reused_count))
    exported = export_global(prepared, numerical, root, contract, guard)
    if exported['accepted'] != all(convergence_flags): raise ValueError('Export acceptance disagrees with checkpoints')
    return dict(state='completed' if exported['accepted'] else 'nonconverged',
        fitted_targets=fitted_count, reused_targets=reused_count, rows=exported['rows'],
        convergence_rate=exported['convergence_rate'], prediction_sha256=exported['sha256'],
        contract_sha256=digest(contract))
