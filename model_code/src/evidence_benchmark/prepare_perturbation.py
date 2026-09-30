"""Prepare a blinded control-only model input and real CRISPR truth."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import sparse, stats

from .common import bh_fdr, finalize_dataset, read_table, rebuild_registry, write_baseline_input, write_empty_priors
from .config import PERTURBATION_CONFIG, PerturbationConfig


def _pick(columns: pd.Index, candidates: tuple[str, ...], label: str, required: bool = True) -> str | None:
    lookup = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    if required:
        raise ValueError(f"Cannot identify {label}; columns={columns.tolist()}")
    return None


def _clean_perturbation(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"^(sgrna[-_:]?|grna[-_:]?|sg[-_:]?)", "", text, flags=re.I)
    text = re.sub(r"[-_:]?(ko|kd|crispri|crispr)$", "", text, flags=re.I)
    # Official GEO names abbreviate four targets; use the series' 18-target
    # list for biological identity, not string resemblance to a gene symbol.
    target_aliases = {"dpp5": "Dpp6", "ndufaf": "Ndufaf2", "oligo2": "Olig2", "rbfox": "Rbfox3"}
    return target_aliases.get(text.lower(), text)


def _is_control(value: str, patterns: tuple[str, ...]) -> bool:
    lowered = value.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


def _dense(matrix: object) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def _screen_genes_backed(backed, selected_indices: list[int], config: PerturbationConfig) -> list[int]:
    """Screen genes in bounded-memory cell batches from a backed H5AD."""

    gene_count = backed.n_vars
    detected = np.zeros(gene_count, dtype=np.int64)
    log_sum = np.zeros(gene_count, dtype=np.float64)
    log_square_sum = np.zeros(gene_count, dtype=np.float64)
    for start in range(0, len(selected_indices), config.gene_screening_batch_cells):
        batch_indices = selected_indices[start : start + config.gene_screening_batch_cells]
        batch = _dense(backed[batch_indices, :].X).astype(np.float64, copy=False)
        detected += np.count_nonzero(batch > 0, axis=0)
        logged = np.log1p(batch)
        log_sum += logged.sum(axis=0)
        log_square_sum += np.square(logged).sum(axis=0)
    count = float(len(selected_indices))
    detection = detected / count
    variance = np.maximum(log_square_sum / count - np.square(log_sum / count), 0.0)
    var_names = backed.var_names.astype(str).to_numpy()
    order = np.lexsort((var_names, -variance))
    return [
        int(index) for index in order
        if detection[index] >= config.minimum_detection_fraction
    ][: config.maximum_genes]


def _control_only_screen_indices(selected_indices: list[int], control_mask: np.ndarray) -> list[int]:
    indices = [index for index in selected_indices if control_mask[index]]
    if not indices:
        raise ValueError("No safe-control cells available for blinded gene screening")
    return indices


def _pseudobulk_truth(
    values: np.ndarray,
    genes: list[str],
    perturbations: np.ndarray,
    samples: np.ndarray,
    controls: np.ndarray,
    tfs: set[str],
    config: PerturbationConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for perturbation in sorted(set(perturbations[~controls])):
        tf = _clean_perturbation(perturbation)
        if tf not in tfs:
            continue
        effects: list[np.ndarray] = []
        used_samples: list[str] = []
        for sample in sorted(set(samples)):
            treated_idx = (perturbations == perturbation) & (samples == sample)
            control_idx = controls & (samples == sample)
            if treated_idx.sum() < config.minimum_cells_per_group or control_idx.sum() < config.minimum_cells_per_group:
                continue
            treated_counts = values[treated_idx].sum(axis=0)
            control_counts = values[control_idx].sum(axis=0)
            treated_cpm = treated_counts / max(float(treated_counts.sum()), 1.0) * 1e6
            control_cpm = control_counts / max(float(control_counts.sum()), 1.0) * 1e6
            effects.append(np.log2(1.0 + treated_cpm) - np.log2(1.0 + control_cpm))
            used_samples.append(str(sample))
        if not effects:
            continue
        effect_matrix = np.vstack(effects)
        mean_effect = effect_matrix.mean(axis=0)
        if len(effects) >= 2:
            pvalues = stats.ttest_1samp(effect_matrix, popmean=0.0, axis=0, nan_policy="omit").pvalue
        else:
            pvalues = np.full(len(genes), np.nan)
        fdr = bh_fdr(np.asarray(pvalues, dtype=float))
        for index, target in enumerate(genes):
            if target == tf:
                continue
            label = bool(abs(mean_effect[index]) >= config.log2fc_threshold and np.isfinite(fdr[index]) and fdr[index] <= config.fdr_threshold)
            rows.append({
                "tf": tf, "target": target, "label": int(label),
                "perturbation_effect_log2fc": float(mean_effect[index]),
                # These are knockouts: negative perturbation effect implies positive TF regulation.
                "signed_truth": float(-np.sign(mean_effect[index])) if label else 0.0,
                "true_score": float(abs(mean_effect[index])), "pvalue": float(pvalues[index]),
                "fdr": float(fdr[index]), "biological_replicates": len(effects),
                "samples": ";".join(used_samples), "truth_source": "sample-aware_pseudobulk_KO",
            })
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError(
            "No TF perturbation had enough treated/control cells in matched biological samples. "
            "Inspect perturbation names, sample IDs, and minimum_cells_per_group."
        )
    return result


def run_perturbation(config: PerturbationConfig = PERTURBATION_CONFIG):
    try:
        import anndata as ad
    except ImportError as error:
        raise ImportError("Install anndata in the preprocessing environment") from error
    if not config.standardized_h5ad.exists():
        raise FileNotFoundError(
            f"Missing {config.standardized_h5ad}. Convert the official GSE274447 Stereo-seq output to an H5AD "
            f"whose obs contains perturbation/sample/cell_type and whose obsm[{config.spatial_key!r}] contains real XY coordinates."
        )
    backed = ad.read_h5ad(config.standardized_h5ad, backed="r")
    perturb_col = _pick(backed.obs.columns, config.perturbation_column_candidates, "perturbation")
    sample_col = _pick(backed.obs.columns, config.sample_column_candidates, "biological sample")
    cell_type_col = _pick(backed.obs.columns, config.cell_type_column_candidates, "cell type", required=False)
    perturb_raw = backed.obs[perturb_col].astype(str).to_numpy()
    cleaned_raw = np.asarray([_clean_perturbation(value) for value in perturb_raw], dtype=object)
    control_mask = np.asarray([_is_control(value, config.control_patterns) for value in cleaned_raw])
    tf_reference = read_table(config.tf_reference)
    tf_column = "source_genesymbol" if "source_genesymbol" in tf_reference else "source"
    known_tfs = set(tf_reference[tf_column].astype(str))
    tf_lookup = {value.lower(): value for value in known_tfs}
    cleaned = np.asarray([tf_lookup.get(str(value).lower(), value) for value in cleaned_raw], dtype=object)
    relevant = control_mask | np.asarray([value in known_tfs for value in cleaned])
    random = np.random.default_rng(config.random_seed)
    selected_indices: list[int] = []
    for group in sorted(set(cleaned[relevant])):
        indices = np.flatnonzero(relevant & (cleaned == group))
        if len(indices) > config.maximum_cells_per_group:
            indices = np.sort(random.choice(indices, config.maximum_cells_per_group, replace=False))
        selected_indices.extend(indices.tolist())
    selected_indices = sorted(selected_indices)
    if not selected_indices:
        raise ValueError("No control or TF-perturbed cells selected")
    # Feature selection is part of model fitting: treated cells must not
    # influence which genes enter the control-only GRN or its evaluation grid.
    try:
        screening_indices = _control_only_screen_indices(selected_indices, control_mask)
    except ValueError:
        backed.file.close()
        raise
    gene_indices = _screen_genes_backed(backed, screening_indices, config)
    if not gene_indices:
        backed.file.close()
        raise ValueError("No genes passed perturbation detection screening")
    data = backed[selected_indices, gene_indices].to_memory()
    backed.file.close()
    values = _dense(data.X).astype(float)
    genes = data.var_names.astype(str).tolist()
    obs = data.obs.copy()
    perturbations_raw = obs[perturb_col].astype(str).to_numpy()
    perturbations_clean = np.asarray([_clean_perturbation(value) for value in perturbations_raw], dtype=object)
    perturbations = np.asarray(
        [tf_lookup.get(str(value).lower(), value) for value in perturbations_clean], dtype=object
    )
    controls = np.asarray([_is_control(value, config.control_patterns) for value in perturbations])
    samples = obs[sample_col].astype(str).to_numpy()
    truth = _pseudobulk_truth(values, genes, perturbations, samples, controls, known_tfs, config)
    if config.spatial_key not in data.obsm:
        raise ValueError(f"H5AD lacks real coordinates in obsm[{config.spatial_key!r}]")
    coordinates_all = np.asarray(data.obsm[config.spatial_key])[:, :2]
    # Critical blinding rule: only safe/control cells enter the model.
    control_values = values[controls]
    control_obs = obs.iloc[np.flatnonzero(controls)].copy()
    control_coordinates = coordinates_all[controls]
    cells = control_obs.index.astype(str).tolist()
    root = config.output_dir
    truth_dir = root / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(control_values, index=cells, columns=genes).rename_axis("cell_id").to_csv(root / "expression.csv")
    pd.DataFrame(control_coordinates, index=cells, columns=["x", "y"]).rename_axis("cell_id").to_csv(root / "coordinates.csv")
    cell_types = control_obs[cell_type_col].astype(str).to_numpy() if cell_type_col else np.repeat("NA", len(cells))
    pd.DataFrame({
        "cell_id": cells, "sample_id": control_obs[sample_col].astype(str).to_numpy(),
        "cell_type": cell_types, "region": cell_types, "condition": "safe_control",
    }).to_csv(root / "metadata.csv", index=False)
    perturbed_tfs = sorted(set(truth.tf) & set(genes))
    pd.DataFrame({"tf": perturbed_tfs}).to_csv(root / "tf_list.csv", index=False)
    lr_reference = read_table(config.lr_reference)
    ligand_col = (
        "ligand" if "ligand" in lr_reference
        else ("source_genesymbol" if "source_genesymbol" in lr_reference else "source")
    )
    receptor_col = (
        "receptor" if "receptor" in lr_reference
        else ("target_genesymbol" if "target_genesymbol" in lr_reference else "target")
    )
    lr = lr_reference.assign(ligand=lr_reference[ligand_col].astype(str), receptor=lr_reference[receptor_col].astype(str))
    lr = lr[lr.ligand.isin(genes) & lr.receptor.isin(genes) & lr.ligand.ne(lr.receptor)].drop_duplicates(["ligand", "receptor"]).head(200)
    lr["lr_id"] = lr.ligand + "__" + lr.receptor
    lr["signal_type"] = "unknown"
    lr["rho"] = 1.0
    lr["confidence"] = np.nan
    lr[["lr_id", "ligand", "receptor", "signal_type", "rho", "confidence"]].to_csv(root / "lr_pairs.csv", index=False)
    write_empty_priors(root)
    truth.to_csv(truth_dir / "truth_perturbation_grn.csv", index=False)
    pd.DataFrame([{
        "model_cells": int(controls.sum()), "truth_cells": int((~controls).sum()),
        "biological_samples": len(set(samples)), "truth_tfs": truth.tf.nunique(),
        "significant_edges": int(truth.label.sum()),
        "blinding": "model_and_gene_screen_used_safe_controls_only",
    }]).to_csv(truth_dir / "perturbation_audit.csv", index=False)
    write_baseline_input(root)
    manifest = finalize_dataset(
        root, dataset_id="spatial_perturbseq_mouse_brain_controls", evidence_track="perturbation",
        source="GSE274447 Spatial Perturb-Seq", evaluation_scope="perturbation_grn",
        truth_kind="interventional", species="mouse", tissue="brain_hippocampus",
        truth_files=("truth_perturbation_grn.csv", "perturbation_audit.csv"),
        notes="Model input contains safe-control cells only; knockout cells are used only for sample-aware held-out evaluation.",
    )
    registry = rebuild_registry()
    print(f"[prepared perturbation] {manifest}; registry={registry}")
    return manifest



