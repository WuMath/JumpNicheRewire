"""Run the official SERGIO code and convert only its GRN task to our contract."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .common import finalize_dataset, rebuild_registry, write_baseline_input, write_empty_priors
from .config import SERGIO_CONFIG, SergioConfig


def _find_file(dataset_dir: Path, preferred: str, contains: str) -> Path:
    direct = dataset_dir / preferred
    if direct.exists():
        return direct
    matches = sorted(path for path in dataset_dir.iterdir() if contains.lower() in path.name.lower())
    if len(matches) != 1:
        raise FileNotFoundError(f"Cannot uniquely find {contains!r} in {dataset_dir}: {matches}")
    return matches[0]


def _parse_interactions(path: Path, number_genes: int) -> tuple[pd.DataFrame, list[int]]:
    edges: list[tuple[int, int, float]] = []
    regulators: set[int] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            values = [value.strip() for value in row]
            target, count = int(float(values[0])), int(float(values[1]))
            sources = [int(float(value)) for value in values[2 : 2 + count]]
            strengths = [float(value) for value in values[2 + count : 2 + 2 * count]]
            for source, strength in zip(sources, strengths):
                edges.append((source, target, strength))
                regulators.add(source)
    edge_lookup = {(source, target): strength for source, target, strength in edges}
    rows = []
    for source in sorted(regulators):
        # Evaluate the full directed TF-to-gene universe.  A SERGIO regulator
        # may itself be regulated at an upstream layer, so excluding all TFs
        # from the target universe would silently delete true TF-to-TF edges.
        for target in range(number_genes):
            if source == target:
                continue
            strength = edge_lookup.get((source, target), 0.0)
            rows.append({
                "tf": f"G{source}", "target": f"G{target}",
                "label": int((source, target) in edge_lookup),
                "true_score": abs(strength), "signed_truth": strength,
            })
    return pd.DataFrame(rows), sorted(regulators)


def _coordinates(number_bins: int, cells_per_bin: int, sd: float, seed: int) -> tuple[np.ndarray, list[str]]:
    random = np.random.default_rng(seed)
    angles = np.linspace(0.0, 2.0 * np.pi, number_bins, endpoint=False)
    centers = np.column_stack([np.cos(angles), np.sin(angles)]) * 5.0
    values = np.vstack([center + random.normal(0.0, sd, size=(cells_per_bin, 2)) for center in centers])
    labels = [f"SERGIO_bin_{index + 1}" for index in range(number_bins) for _ in range(cells_per_bin)]
    return values, labels


def _run_one(config: SergioConfig) -> Path:
    dataset_dir = config.official_repo / config.official_dataset_relative
    interaction_path = _find_file(dataset_dir, config.interaction_filename, "Interaction")
    regulator_path = _find_file(dataset_dir, config.regulator_filename, "Regs")
    truth, regulators = _parse_interactions(interaction_path, config.number_genes)
    if not config.run_official_simulator:
        raise ValueError("run_official_simulator=False requires an explicit imported expression adapter; no local substitute is used.")
    if str(config.official_repo) not in sys.path:
        sys.path.insert(0, str(config.official_repo))
    try:
        from SERGIO.sergio import sergio
    except Exception as error:
        raise ImportError(
            "Cannot import official SERGIO. Use its source tree and a separate environment with numpy<2; "
            "do not install the unrelated PyPI package named 'sergio'."
        ) from error
    np.random.seed(config.random_seed)
    simulator = sergio(
        number_genes=config.number_genes, number_bins=config.number_bins,
        number_sc=config.cells_per_bin, noise_params=config.noise_parameter,
        decays=config.decay, sampling_state=config.sampling_state,
        noise_type=config.noise_type,
    )
    simulator.build_graph(
        input_file_taregts=str(interaction_path), input_file_regs=str(regulator_path),
        shared_coop_state=config.shared_coop_state,
    )
    simulator.simulate()
    expression_3d = simulator.getExpressions()
    if config.add_technical_noise:
        _, library_adjusted = simulator.lib_size_effect(expression_3d, mean=4.6, scale=0.4)
        indicator = simulator.dropout_indicator(
            library_adjusted, shape=config.dropout_shape, percentile=config.dropout_percentile
        )
        expression_3d = simulator.convert_to_UMIcounts(np.multiply(indicator, library_adjusted))
    expression = np.concatenate(expression_3d, axis=1).T
    expected_cells = config.number_bins * config.cells_per_bin
    if expression.shape != (expected_cells, config.number_genes):
        raise ValueError(f"Unexpected SERGIO expression shape {expression.shape}")
    root = config.output_dir
    truth_dir = root / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)
    cells = [f"SERGIO_C{index + 1}" for index in range(expected_cells)]
    genes = [f"G{index}" for index in range(config.number_genes)]
    coordinates, bins = _coordinates(config.number_bins, config.cells_per_bin, config.coordinate_sd, config.random_seed)
    pd.DataFrame(expression, index=cells, columns=genes).rename_axis("cell_id").to_csv(root / "expression.csv")
    pd.DataFrame(coordinates, index=cells, columns=["x", "y"]).rename_axis("cell_id").to_csv(root / "coordinates.csv")
    pd.DataFrame({
        "cell_id": cells, "sample_id": "SERGIO_1", "cell_type": bins,
        "region": bins, "condition": "simulated",
    }).to_csv(root / "metadata.csv", index=False)
    pd.DataFrame({"tf": [f"G{value}" for value in regulators]}).to_csv(root / "tf_list.csv", index=False)
    non_tf = [gene for gene in range(config.number_genes) if gene not in set(regulators)]
    if len(non_tf) < 2:
        raise ValueError("SERGIO adapter needs two non-TF genes for an inert LR placeholder")
    pd.DataFrame([{
        "lr_id": "SERGIO_NULL_LR", "ligand": f"G{non_tf[0]}", "receptor": f"G{non_tf[1]}",
        "signal_type": "null_placeholder", "rho": 1.0, "confidence": np.nan,
    }]).to_csv(root / "lr_pairs.csv", index=False)
    write_empty_priors(root)
    truth.to_csv(truth_dir / "truth_grn.csv", index=False)
    write_baseline_input(root)
    manifest = finalize_dataset(
        root, dataset_id=f"sergio_grn_100g_seed_{config.random_seed}", evidence_track="public_simulator",
        source="SERGIO official source and GNW-derived bundled network",
        evaluation_scope="grn_only", truth_kind="exact", species="synthetic",
        tissue="not_applicable", truth_files=("truth_grn.csv",),
        notes="Coordinates are adapter layout only. Evaluate GRN recovery; do not evaluate spatial CCI or causal claims.",
    )
    rebuild_registry()
    print(f"[prepared SERGIO] {manifest}")
    return manifest


def run_sergio(config: SergioConfig = SERGIO_CONFIG) -> Path:
    seeds = config.replicate_seeds or (config.random_seed,)
    for seed in seeds:
        _run_one(replace(
            config,
            random_seed=int(seed),
            replicate_seeds=(),
            output_dir=config.output_dir / f"seed_{seed}",
        ))
    registry = rebuild_registry()
    print(f"[prepared SERGIO replicates] {len(seeds)}; registry={registry}")
    return registry



