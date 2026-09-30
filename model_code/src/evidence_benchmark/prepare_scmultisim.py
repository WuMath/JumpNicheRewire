"""Official R simulation -> strict model contract. Never select features by labels."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import subprocess
import pandas as pd
import numpy as np
from .common import (sha256_file, stable_hash, write_json, write_empty_priors,
                     write_baseline_input, finalize_dataset, MODEL_FILES)

EXPORT_FILES = ("expression.csv", "coordinates.csv", "metadata.csv", "candidate_tfs.csv",
                "candidate_lr.csv", "grn_positive_edges.csv", "cci_positive_edges.csv",
                "export_complete.json", "R_sessionInfo.txt", "official_result.rds")


def simulator_protocol(data):
    values = asdict(data)
    for key in ("rscript", "raw_root", "output_dir", "split", "pilot_seeds", "validation_seeds",
                "test_seeds", "maximum_truth_rows"):
        values.pop(key)
    values["adapter_sha256"] = sha256_file(Path(__file__).with_name("scmultisim_export.R"))
    return values


def _required(frame, columns, label):
    if not set(columns).issubset(frame) or frame[list(columns)].isna().any().any():
        raise ValueError(f"{label}: missing fields/values: {columns}")
    if frame.duplicated(list(columns)).any():
        raise ValueError(f"{label}: duplicate keys: {columns}")


def convert_export(raw, destination, dataset_id, data):
    """Labels are only used to build evaluation tables after model inputs are fixed."""
    raw, destination = Path(raw), Path(destination)
    info = json.loads((raw / "export_complete.json").read_text())
    if info["version"] != data.expected_package_version or info["direction"] != "ct2_sender_ct1_receiver":
        raise ValueError("Unsupported exporter version/direction; re-audit official R API first")
    expression = pd.read_csv(raw / "expression.csv", index_col=0)
    expression.index = expression.index.astype(str)
    expression.columns = expression.columns.astype(str)
    if expression.empty or expression.index.has_duplicates or expression.columns.has_duplicates:
        raise ValueError("Empty/duplicate expression axes")
    if expression.shape != (data.cells, data.genes):
        raise ValueError(f"Expected cells x genes {(data.cells, data.genes)}, got {expression.shape}")
    values = expression.to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Counts must be finite and nonnegative")
    coords = pd.read_csv(raw / "coordinates.csv", dtype={"cell_id": str})
    meta = pd.read_csv(raw / "metadata.csv", dtype=str)
    for frame in (coords, meta):
        _required(frame, ("cell_id",), "cell IDs")
        if set(frame.cell_id) != set(expression.index):
            raise ValueError("Cell-ID mismatch; never silently discard unmatched cells")
    coords = coords.set_index("cell_id").loc[expression.index, ["x", "y"]]
    if not np.isfinite(coords.to_numpy(float)).all():
        raise ValueError("Invalid coordinates")
    meta = meta.set_index("cell_id").loc[expression.index]
    meta.index.name = "cell_id"
    meta = meta.reset_index()
    _required(meta, ("cell_id", "cell_type", "region", "sample_id", "condition"), "metadata")
    tfs = pd.read_csv(raw / "candidate_tfs.csv", dtype=str)
    lr = pd.read_csv(raw / "candidate_lr.csv", dtype=str)
    _required(tfs, ("tf",), "TF catalog")
    _required(lr, ("ligand", "receptor"), "LR catalog")
    genes = set(expression.columns)
    if tfs.empty or lr.empty or (set(tfs.tf) | set(lr.ligand) | set(lr.receptor)) - genes:
        raise ValueError("Candidate TF/LR IDs absent from counts")
    destination.mkdir(parents=True, exist_ok=True)
    truth_dir = destination / "truth"
    truth_dir.mkdir(exist_ok=True)
    expression.to_csv(destination / "expression.csv")
    coords.to_csv(destination / "coordinates.csv", index_label="cell_id")
    meta.to_csv(destination / "metadata.csv", index=False)
    tfs.to_csv(destination / "tf_list.csv", index=False)
    lr = lr.assign(lr_id=lr.ligand + "__" + lr.receptor,
                   signal_type="synthetic_cross_cell_pair", rho=1.0, confidence=1.0)
    lr.to_csv(destination / "lr_pairs.csv", index=False)
    write_empty_priors(destination)
    write_baseline_input(destination)
    # Dense candidate tables are bounded. Labels never enter prior_tf_gene/LR->TF.
    types = sorted(meta.cell_type.unique())
    n_grn = len(tfs) * len(expression.columns)
    n_cci = len(types)**2 * len(lr)
    if max(n_grn, n_cci) > data.maximum_truth_rows:
        raise MemoryError("Truth universe exceeds configured row budget; do not truncate")
    grn_keys = ["tf", "target"]
    cci_keys = ["sender_cell_type", "receiver_cell_type", "ligand", "receptor"]
    grn = pd.MultiIndex.from_product([tfs.tf, expression.columns], names=grn_keys).to_frame(index=False)
    grn = grn[grn.tf.ne(grn.target)]
    cci = pd.MultiIndex.from_product([types, types], names=cci_keys[:2]).to_frame(index=False)
    cci = cci.merge(lr[["ligand", "receptor"]], how="cross")
    for name, universe, keys, positives in (
        ("truth_grn.csv", grn, grn_keys, "grn_positive_edges.csv"),
        ("truth_scmultisim_cci.csv", cci, cci_keys, "cci_positive_edges.csv"),
    ):
        edges = pd.read_csv(raw / positives, dtype={key: str for key in keys})
        _required(edges, keys, positives)
        effects = pd.to_numeric(edges.effect, errors="raise")
        if not np.isfinite(effects).all():
            raise ValueError("Nonfinite simulator effects")
        edges = edges.assign(effect=effects)
        if name == "truth_grn.csv":
            edges = edges[edges.tf.ne(edges.target)]
        check = edges.merge(universe, on=keys, how="left", indicator=True)
        if check._merge.ne("both").any():
            raise ValueError(f"Truth outside declared candidate universe: {positives}")
        labeled = universe.merge(edges[keys + ["effect"]], on=keys, how="left", validate="one_to_one")
        labeled["effect"] = labeled.effect.fillna(0)
        labeled["label"] = labeled.effect.ne(0).astype(int)
        labeled.to_csv(truth_dir / name, index=False)
    write_json(destination / "preparation_audit.json", dict(
        model_input_uses_truth=False, candidate_catalog="declared before simulation; all candidates retained",
        expression_scale="raw_counts", cells=len(expression), genes=expression.shape[1],
        tf_candidates=len(tfs), lr_candidates=len(lr),
        zero_library_cells=int((values.sum(axis=1) == 0).sum()),
        zero_expression_genes=int((values.sum(axis=0) == 0).sum()),
        cci_direction=info["direction"], protocol=simulator_protocol(data)))
    manifest = finalize_dataset(destination, dataset_id=dataset_id, evidence_track="public_simulator",
        source="scMultiSim " + info["version"], evaluation_scope="scmultisim_grn_cci",
        truth_kind="exact", species="synthetic", tissue="simulated_spatial",
        truth_files=("truth_grn.csv", "truth_scmultisim_cci.csv"),
        notes="Fixed public GRN across seeds; CCI is type-level synthetic LR support, not measured biological LR or a full LR-TF-gene truth.")
    return json.loads(manifest.read_text(encoding="utf-8"))


def preprocess(config, action):
    from jumpniche_app.safety import writable
    data = config.preprocessing
    ids = data.dataset_ids()
    if action == "check":
        print(f"[Rscript] {data.rscript} / exists={data.rscript.is_file()}")
        print(f"[version required] scMultiSim {data.expected_package_version}")
        print(f"[split] {data.split}: {ids}")
        print(f"[raw] {data.raw_root / data.split}")
        print(f"[prepared] {data.output_dir / data.split}")
        return None
    if action not in {"simulate", "prepare"}:
        raise ValueError("scmultisim preprocess_action: check / simulate / prepare")
    if data.split == "test":
        from .scmultisim_protocol import require_frozen
        require_frozen(config)
    if not 1 <= data.active_lr_min <= data.active_lr_max <= data.lr_count:
        raise ValueError("Invalid active LR range")
    if data.cci_effect <= 0:
        raise ValueError("This adapter supports positive CCI effects; zero-effect controls need a separate audited protocol")
    records = []
    for dataset_id, seed in zip(ids, data.seeds()):
        raw = writable(data.raw_root / data.split / dataset_id)
        destination = writable(data.output_dir / data.split / dataset_id)
        payload = dict(simulator_protocol(data), seed=seed, dataset_id=dataset_id,
                       destination=str(raw), expected_package_version=data.expected_package_version)
        marker = raw / "simulation_receipt.json"
        if action == "simulate":
            if marker.exists():
                receipt = json.loads(marker.read_text())
                if receipt["signature"] != stable_hash(payload) or any(
                    not (raw / f).is_file() or sha256_file(raw / f) != receipt["files"][f] for f in EXPORT_FILES):
                    raise ValueError(f"Changed/incomplete simulation cache: {raw}. Use a new version directory.")
                print(f"[reuse official simulation] {dataset_id}")
                continue
            if raw.exists() and any(raw.iterdir()):
                raise FileExistsError(f"Incomplete/untracked simulation: {raw}; preserve it and use a new version directory")
            raw.mkdir(parents=True)
            config_path = raw / "simulation_config.json"
            write_json(config_path, payload)
            with (raw / "simulation.log").open("w", encoding="utf-8") as log:
                print(f"[official R simulation] {dataset_id}; log={log.name}", flush=True)
                subprocess.run([str(data.rscript), "--vanilla", str(Path(__file__).with_name("scmultisim_export.R")),
                                str(config_path)], stdout=log, stderr=subprocess.STDOUT, check=True)
            write_json(marker, dict(signature=stable_hash(payload), files={f: sha256_file(raw / f) for f in EXPORT_FILES}))
            continue
        if not marker.is_file():
            raise FileNotFoundError(f"First run preprocess_action='simulate': {marker}")
        receipt = json.loads(marker.read_text())
        if receipt["signature"] != stable_hash(payload):
            raise ValueError("Simulation configuration changed; use a new experiment version")
        hashes = {f: sha256_file(raw / f) for f in EXPORT_FILES}
        if hashes != receipt["files"]:
            raise ValueError("Raw export hash mismatch")
        signature = stable_hash(dict(raw=hashes, converter=sha256_file(Path(__file__))))
        prepared_marker = destination / "prepare_receipt.json"
        if prepared_marker.is_file():
            old = json.loads(prepared_marker.read_text())
            if old["signature"] != signature or any(sha256_file(destination / f) != h for f,h in old["files"].items()):
                raise ValueError(f"Prepared cache changed: {destination}; use a new version")
            manifest = json.loads((destination / "evidence_manifest.json").read_text(encoding="utf-8"))
        else:
            if destination.exists() and any(destination.iterdir()):
                raise FileExistsError(f"Untracked prepared output: {destination}")
            manifest = convert_export(raw, destination, dataset_id, data)
            write_json(prepared_marker, dict(signature=signature, files=manifest["files"]))
        manifest.update(split=data.split, scenario=data.scenario)
        records.append({k:v for k,v in manifest.items() if k not in {"files", "notes"}})
    if action == "prepare":
        path = data.output_dir / data.split / "dataset_registry.csv"
        pd.DataFrame(records).to_csv(path, index=False)
        return path
    return data.raw_root / data.split
