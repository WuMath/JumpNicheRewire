"""Stage 1 utilities with manual downloads and simulation kept separate.

No function in this module downloads a data file. Manual source archives live
under ``data/formal_benchmark/manual_downloads``. Generated model-ready data
live under ``data/formal_benchmark/simulated``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

from .config import (
    DATA_CONFIG,
    DataStageConfig,
    PROJECT_ROOT,
    ScenarioSpec,
    validate_benchmark_profile,
)
from .state import completion_matches, stable_hash, utc_now, write_completion, write_json_atomic


SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jump_niche_rewire.simulation import SimulationConfig, generate_simulated_project


ZENODO_FILES: dict[str, dict[str, object]] = {
    "main_tree1.zip": {"bytes": 941_000_000, "md5": "f7bc45378f3c3d54b3e818b8fc8edaa3", "role": "scMultiSim main input/tree replicate 1"},
    "main_tree3.zip": {"bytes": 944_700_000, "md5": "319780bc7e8e428199e5d149443016fd", "role": "scMultiSim main input/tree replicate 3"},
    "main_tree5.zip": {"bytes": 938_900_000, "md5": "3065b58dadb97b21ecdd953c716baf9b", "role": "scMultiSim main input/tree replicate 5"},
    "spatial_3000cells_2000genes.zip": {"bytes": 132_200_000, "md5": "1430e905e8309e58a5a76ed3923bd4de", "role": "scMultiSim spatial benchmark, 3000 cells/2000 genes"},
    "spatial_5000cells_500genes.zip": {"bytes": 41_300_000, "md5": "5ee5acd69545c1a30fbf560643166d02", "role": "scMultiSim spatial benchmark, 5000 cells/500 genes"},
    "main_grn_output.zip": {"bytes": 2_500_000, "md5": "a9d13e951553098733b81fe7c4bb197f", "role": "published method outputs; reference only, not model input"},
}


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_manual_files(config: DataStageConfig) -> tuple[str, ...]:
    selected = list(config.manual_download_files)
    if config.include_reference_output_archive:
        selected.append("main_grn_output.zip")
    unknown = sorted(set(selected) - set(ZENODO_FILES))
    if unknown:
        raise ValueError(f"Unknown manual download file(s): {unknown}")
    return tuple(dict.fromkeys(selected))


def prepare_manual_downloads(config: DataStageConfig = DATA_CONFIG) -> Path:
    """Write browser links and verify manually downloaded files; never download."""

    root = config.data_root
    download_dir = Path(config.manual_download_root or (root / "manual_downloads"))
    prepared_external = root / "prepared_external"
    simulated_dir = root / "simulated"
    baseline_jobs_dir = root / "baseline_jobs"
    manifest_dir = root / "manifests"
    for directory in (download_dir, prepared_external, simulated_dir, baseline_jobs_dir, manifest_dir):
        directory.mkdir(parents=True, exist_ok=True)
    selected = set(_selected_manual_files(config))
    catalog_rows = []
    status_rows = []
    for name, metadata in ZENODO_FILES.items():
        url = f"https://zenodo.org/records/{config.zenodo_record_id}/files/{name}?download=1"
        catalog_rows.append({
            "selected": name in selected,
            "file": name,
            "expected_bytes_approx": metadata["bytes"],
            "expected_md5": metadata["md5"],
            "role": metadata["role"],
            "manual_download_url": url,
            "save_to": str(download_dir / name),
        })
        if name not in selected:
            continue
        path = download_dir / name
        if not path.exists():
            state, observed = "missing", None
        else:
            observed = _md5sum(path)
            state = "verified" if observed.lower() == str(metadata["md5"]).lower() else "checksum_mismatch"
        status_rows.append({
            "file": name,
            "state": state,
            "path": str(path),
            "expected_md5": metadata["md5"],
            "observed_md5": observed,
            "checked_at_utc": utc_now(),
        })
    catalog = pd.DataFrame(catalog_rows)
    status = pd.DataFrame(status_rows)
    _atomic_csv(catalog, manifest_dir / "manual_download_catalog.csv")
    status_path = manifest_dir / "manual_download_status.csv"
    _atomic_csv(status, status_path)
    selected_catalog = catalog[catalog["selected"]]
    lines = [
        "# Benchmark 原始数据手动下载",
        "",
        "本项目不会自动下载这些文件。请在浏览器中逐个打开链接，并把文件原名保存到：",
        "",
        f"`{download_dir}`",
        "",
        "| 文件 | 用途 | 手动下载链接 |",
        "|---|---|---|",
    ]
    for _, row in selected_catalog.iterrows():
        lines.append(f"| `{row['file']}` | {row['role']} | [Zenodo 下载]({row['manual_download_url']}) |")
    lines.extend([
        "",
        "下载完成后再次运行 `python .\\run_benchmark_stage1_manual_download.py`，",
        "程序只计算 MD5 并更新 `manifests/manual_download_status.csv`，不会联网下载。",
        "",
        "原始 ZIP 不是 JumpNicheRewire 的直接输入。解压并完成格式转换后的数据集应放到",
        f"`{prepared_external}`，每个数据集必须具有与模拟数据相同的输入契约和 `dataset_manifest.json`。",
        "`main_grn_output.zip` 是论文已发表结果，不得作为模型输入。",
    ])
    (root / "MANUAL_DOWNLOAD.md").write_text("\n".join(lines), encoding="utf-8")
    mismatches = status[status["state"] == "checksum_mismatch"]
    if not mismatches.empty:
        raise IOError(f"Manual download checksum mismatch: {mismatches['file'].tolist()}")
    missing = status[status["state"] == "missing"]
    if config.require_all_selected_manual_downloads and not missing.empty:
        raise FileNotFoundError(f"Missing manual downloads: {missing['file'].tolist()}; see {root / 'MANUAL_DOWNLOAD.md'}")
    if not status.empty and (status["state"] == "verified").all():
        signature = stable_hash(status[["file", "observed_md5"]].to_dict("records"))
        write_completion(root / ".manual_downloads_complete.json", signature, {"files": len(status)})
    print(f"[manual download plan] {root / 'MANUAL_DOWNLOAD.md'}")
    print(f"[manual download status] {status_path}")
    return status_path


def _validate_split_seeds(config: DataStageConfig) -> list[tuple[str, int]]:
    groups = {"train": tuple(config.split.train_seeds), "validation": tuple(config.split.validation_seeds), "test": tuple(config.split.test_seeds)}
    all_values = [seed for values in groups.values() for seed in values]
    if len(all_values) != len(set(all_values)):
        raise ValueError("train/validation/test seeds must be mutually disjoint")
    return [(split_name, seed) for split_name, values in groups.items() for seed in values]


def _manifest_profile(item: dict[str, object]) -> str | None:
    """Read an explicit profile, with a narrow migration rule for old pilots."""

    explicit = item.get("benchmark_profile")
    if explicit is not None:
        return validate_benchmark_profile(str(explicit))
    scenario = item.get("scenario", "")
    scenario_name = scenario.get("name", "") if isinstance(scenario, dict) else str(scenario)
    if str(scenario_name).startswith("pilot_"):
        return "pilot"
    if item.get("data_origin") == "simulated":
        return "final"
    # Legacy external manifests stay unassigned until the user explicitly
    # declares whether they belong to a pilot or final benchmark cohort.
    return None


def _apply_observation_noise(dataset_dir: Path, scenario: ScenarioSpec, seed: int) -> None:
    random = np.random.default_rng(seed + 91_337)
    expression_path = dataset_dir / "expression.csv"
    expression = pd.read_csv(expression_path, index_col=0)
    if scenario.dropout_rate > 0:
        values = expression.to_numpy(dtype=float)
        values[(values > 0) & (random.random(values.shape) < scenario.dropout_rate)] = 0.0
        expression.iloc[:, :] = values
        expression.to_csv(expression_path)
    for name, evidence_columns in (("prior_tf_gene.csv", ("motif", "atac", "grn")), ("prior_lr_tf.csv", ("score",))):
        path = dataset_dir / name
        frame = pd.read_csv(path)
        if scenario.prior_coverage < 1.0:
            keep = random.random(len(frame)) < scenario.prior_coverage
            for column in evidence_columns:
                if column in frame:
                    frame.loc[~keep, column] = np.nan
            frame.to_csv(path, index=False)


def _make_sparse_truth(dataset_dir: Path, scenario: ScenarioSpec) -> dict[str, float]:
    truth_dir = dataset_dir / "truth"
    prevalences: dict[str, float] = {}
    for name, score_column in (("truth_grn.csv", "signed_truth"), ("truth_lr_gene.csv", "signed_truth")):
        path = truth_dir / name
        frame = pd.read_csv(path)
        values = pd.to_numeric(frame[score_column], errors="coerce").fillna(0.0).abs()
        frame["label"] = (values >= scenario.truth_effect_threshold).astype(int)
        frame.to_csv(path, index=False)
        prevalences[name.removeprefix("truth_").removesuffix(".csv")] = float(frame["label"].mean())
    for name in ("truth_receptors.csv", "truth_cci.csv"):
        frame = pd.read_csv(truth_dir / name)
        prevalences[name.removeprefix("truth_").removesuffix(".csv")] = float(frame["label"].mean()) if len(frame) else 0.0
    causal = pd.read_csv(truth_dir / "truth_causal.csv")
    denominator = scenario.region_count * scenario.lr_count * scenario.tf_count * scenario.target_count
    prevalences["causal_edges_per_candidate"] = float(len(causal) / max(denominator, 1))
    return prevalences


def _write_baseline_inputs(dataset_dir: Path, write_h5ad: bool) -> Path:
    output = dataset_dir / "baseline_input"
    output.mkdir(parents=True, exist_ok=True)
    expression = pd.read_csv(dataset_dir / "expression.csv", index_col=0)
    coordinates = pd.read_csv(dataset_dir / "coordinates.csv", index_col=0)
    metadata = pd.read_csv(dataset_dir / "metadata.csv")
    expression.to_csv(output / "expression_cells_by_genes.csv")
    expression.T.to_csv(output / "expression_genes_by_cells.csv")
    coordinates.to_csv(output / "coordinates.csv")
    metadata.to_csv(output / "metadata.csv", index=False)
    for name in ("tf_list.csv", "lr_pairs.csv", "prior_tf_gene.csv", "prior_lr_tf.csv"):
        shutil.copy2(dataset_dir / name, output / name)
    np.save(output / "counts.npy", expression.to_numpy(dtype=np.float32))
    np.save(output / "coordinates.npy", coordinates.loc[expression.index].to_numpy(dtype=np.float32))
    pd.DataFrame({"gene": expression.columns.astype(str)}).to_csv(output / "genes.csv", index=False)
    pd.DataFrame({"cell_id": expression.index.astype(str)}).to_csv(output / "cells.csv", index=False)
    if write_h5ad:
        try:
            import anndata as ad
            ad.AnnData(X=expression.to_numpy(dtype=np.float32), obs=metadata.set_index("cell_id").reindex(expression.index), var=pd.DataFrame(index=expression.columns.astype(str)), obsm={"spatial": coordinates.loc[expression.index].to_numpy(dtype=np.float32)}).write_h5ad(output / "input.h5ad")
        except ImportError:
            pass
    return output


def _generate_one(config: DataStageConfig, scenario: ScenarioSpec, split_name: str, seed: int) -> dict[str, object]:
    dataset_id = f"{scenario.name}__{split_name}__s{seed}"
    dataset_dir = config.data_root / "simulated" / dataset_id
    profile = validate_benchmark_profile(config.profile)
    signature = stable_hash({"benchmark_profile": profile, "scenario": scenario, "split": split_name, "seed": seed, "generator": "simulation.py@v2"})
    marker = dataset_dir / ".complete.json"
    manifest_path = dataset_dir / "dataset_manifest.json"
    required = (manifest_path, dataset_dir / "truth/truth_grn.csv", dataset_dir / "baseline_input/counts.npy")
    if not config.overwrite and completion_matches(marker, signature, required):
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    if dataset_dir.exists():
        expected_parent = (config.data_root / "simulated").resolve()
        resolved = dataset_dir.resolve()
        if resolved.parent != expected_parent or resolved == expected_parent:
            raise RuntimeError(f"Refusing to replace unexpected dataset path: {resolved}")
        shutil.rmtree(resolved)
    generate_simulated_project(SimulationConfig(
        output_dir=dataset_dir, random_seed=seed, region_count=scenario.region_count,
        cells_per_region=scenario.cells_per_region, tf_count=scenario.tf_count,
        target_count=scenario.target_count, lr_count=scenario.lr_count,
        noise_gene_count=scenario.noise_gene_count, coordinate_spread=scenario.coordinate_spread,
        region_separation=scenario.region_separation, count_dispersion=scenario.count_dispersion,
        base_edge_density=scenario.base_edge_density, rewiring_edge_density=scenario.rewiring_edge_density,
        causal_confounding=scenario.causal_confounding, confounder_strength=scenario.confounder_strength,
        include_activity_evidence=scenario.include_activity_evidence,
        include_perturbation_evidence=scenario.include_perturbation_evidence,
    ))
    _apply_observation_noise(dataset_dir, scenario, seed)
    prevalence = _make_sparse_truth(dataset_dir, scenario)
    grn_prevalence = prevalence["grn"]
    if not config.minimum_truth_prevalence <= grn_prevalence <= config.maximum_truth_prevalence:
        raise ValueError(f"{dataset_id}: GRN prevalence {grn_prevalence:.4f} is outside [{config.minimum_truth_prevalence}, {config.maximum_truth_prevalence}]")
    baseline_dir = _write_baseline_inputs(dataset_dir, config.write_h5ad_when_available)
    manifest = {
        "dataset_id": dataset_id, "data_origin": "simulated", "split": split_name, "seed": seed,
        "benchmark_profile": profile,
        "scenario": scenario, "cells": scenario.region_count * scenario.cells_per_region,
        "genes": scenario.tf_count + 2 * scenario.lr_count + scenario.target_count + scenario.noise_gene_count,
        "truth_prevalence": prevalence, "dataset_dir": dataset_dir, "truth_dir": dataset_dir / "truth",
        "baseline_input_dir": baseline_dir, "signature": signature, "created_at_utc": utc_now(),
    }
    write_json_atomic(manifest_path, manifest)
    write_completion(marker, signature, {"dataset_id": dataset_id})
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def finalize_stage1(config: DataStageConfig = DATA_CONFIG) -> Path:
    """Freeze all already prepared datasets into one registry; generate nothing."""

    manifest_paths = sorted((config.data_root / "simulated").glob("*/dataset_manifest.json"))
    manifest_paths += sorted((config.data_root / "prepared_external").glob("*/dataset_manifest.json"))
    if not manifest_paths:
        raise FileNotFoundError(f"No model-ready dataset manifests under {config.data_root}. Run run_benchmark_stage1_simulation.py first.")
    manifests = []
    for path in manifest_paths:
        item = json.loads(path.read_text(encoding="utf-8"))
        # The manifest can be moved between computers or reorganized locally.
        # Its current parent, not an embedded historical absolute path, is the
        # authoritative dataset location.
        item["dataset_dir"] = str(path.parent.resolve())
        item["truth_dir"] = str((path.parent / "truth").resolve())
        item["baseline_input_dir"] = str((path.parent / "baseline_input").resolve())
        manifests.append(item)
    profile = validate_benchmark_profile(config.profile)
    manifests = [item for item in manifests if _manifest_profile(item) == profile]
    if not manifests:
        raise FileNotFoundError(f"No model-ready {profile!r} manifests were found under {config.data_root}")
    dataset_ids = [str(item["dataset_id"]) for item in manifests]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("Duplicate dataset_id values were found while finalizing stage 1")
    overview = pd.DataFrame([{
        "dataset_id": item["dataset_id"], "benchmark_profile": profile,
        "data_origin": item.get("data_origin", "external"),
        "split": item["split"],
        "scenario": item["scenario"]["name"] if isinstance(item.get("scenario"), dict) else item.get("scenario", "external"),
        "seed": item["seed"], "cells": item["cells"], "genes": item["genes"],
        "grn_prevalence": item.get("truth_prevalence", {}).get("grn", np.nan),
        "dataset_dir": item["dataset_dir"], "signature": item["signature"],
    } for item in manifests]).sort_values(["data_origin", "scenario", "split", "seed"])
    registry_path = config.registry_path
    _atomic_csv(overview, registry_path)
    jobs_dir = config.data_root / "baseline_jobs" / profile
    jobs_dir.mkdir(parents=True, exist_ok=True)
    for method in ("CeSpGRN", "SpaGRN", "iMIRACLE", "CLARIFY"):
        jobs = overview[["dataset_id", "benchmark_profile", "data_origin", "split", "scenario", "seed", "signature"]].copy()
        jobs["input_dir"] = overview["dataset_dir"].map(lambda value: str(Path(str(value)) / "baseline_input"))
        jobs["raw_output_dir"] = overview["dataset_id"].map(lambda dataset_id: str(config.profile_benchmark_root / "02_runs" / "baselines_raw" / method / dataset_id))
        jobs["unified_output"] = overview["dataset_id"].map(lambda dataset_id: str(config.profile_benchmark_root / "02_runs" / "baselines" / method / dataset_id / f"{method}_results.xlsx"))
        _atomic_csv(jobs, jobs_dir / f"{method}_jobs.csv")
    signature = stable_hash({"config": config, "datasets": overview["signature"].tolist()})
    write_completion(registry_path.parent / ".data_complete.json", signature, {"profile": profile, "datasets": len(overview)})
    print(f"[stage 1 finalized] {registry_path} ({len(overview)} datasets)")
    return registry_path


def run_simulations(config: DataStageConfig = DATA_CONFIG) -> Path:
    """Generate only simulated datasets, then freeze the model-ready registry."""

    seeds = _validate_split_seeds(config)
    for scenario in config.scenarios():
        for split_name, seed in seeds:
            _generate_one(config, scenario, split_name, seed)
    return finalize_stage1(config)


def run_stage1(config: DataStageConfig = DATA_CONFIG) -> Path:
    """Backward-compatible finalizer; it neither downloads nor simulates."""

    return finalize_stage1(config)



