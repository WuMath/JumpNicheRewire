"""Validation -> freeze -> test gate for this independent benchmark only."""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
from .common import stable_hash, sha256_file, write_json
from .prepare_scmultisim import simulator_protocol


def normalized_model(value):
    value = json.loads(json.dumps(value, default=str))
    # These fields are assigned per dataset by the existing worker, not tuned.
    for section, names in {
        "data": ("input_dir",),
        "output": ("output_dir", "workbook_name", "method_name", "save_parquet"),
        "runtime": ("random_seed", "checkpoint_data_signature"),
        "rewiring": ("top_grn_rows",),
    }.items():
        for name in names:
            value[section].pop(name, None)
    return value


def protocol(config):
    from jumpniche_app.configuration import effective_model
    src = Path(__file__).resolve().parents[1]
    code = {str(p.relative_to(src)): sha256_file(p) for p in sorted((src / "jump_niche_rewire").glob("*.py"))}
    return dict(model=normalized_model(asdict(effective_model(config))),
                simulator=simulator_protocol(config.preprocessing),
                seed_plan={name: list(getattr(config.preprocessing, name + "_seeds"))
                           for name in ("pilot", "validation", "test")},
                converter_sha256=sha256_file(Path(__file__).with_name("prepare_scmultisim.py")), code=code)


def lock_path(config):
    return config.run.output_root / "00_protocol/scmultisim_frozen.json"


def freeze(config):
    if config.preprocessing.split != "validation":
        raise PermissionError("Freeze only after validation, never pilot/test")
    value = protocol(config)
    audit = []
    for dataset_id in config.inputs.dataset_ids:
        root = config.run.output_root / "02_runs/public_simulator" / dataset_id
        status = json.loads((root / "run_status.json").read_text(encoding="utf-8"))
        if status.get("state") != "completed" or status.get("convergence_passed") is not True:
            raise ValueError(f"Validation did not pass convergence: {dataset_id}")
        effective = json.loads((root / "effective_config.json").read_text(encoding="utf-8"))
        if stable_hash(normalized_model(effective)) != stable_hash(value["model"]):
            raise ValueError(f"Validation used different model settings: {dataset_id}")
        audit.append(dict(dataset_id=dataset_id, status_sha256=sha256_file(root / "run_status.json")))
    path = lock_path(config)
    if path.exists():
        old = json.loads(path.read_text())
        if old["signature"] != stable_hash(value):
            raise FileExistsError("A different frozen protocol exists. Use a new experiment root")
        return path
    write_json(path, dict(signature=stable_hash(value), protocol=value, validation_runs=audit,
                         note="GENIE3/GRNBoost2/LIANA must separately freeze their own protocols"))
    return path


def require_frozen(config):
    if not config.run.unlock_test:
        raise PermissionError("test is locked: freeze on validation, then set RunParameters.unlock_test=True")
    path = lock_path(config)
    if not path.is_file():
        raise FileNotFoundError(f"Missing validation freeze: {path}")
    stored = json.loads(path.read_text())
    if stored["signature"] != stable_hash(protocol(config)):
        raise PermissionError("Model/simulator/code changed after freeze. Do not tune against test")
