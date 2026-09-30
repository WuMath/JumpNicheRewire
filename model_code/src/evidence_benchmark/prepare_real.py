"""Build leakage-controlled pseudo-truth for real spatial datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    copy_or_link, finalize_dataset, read_table, rebuild_registry, split_sources,
    write_baseline_input, write_empty_priors,
)
from .config import REAL_CONFIG, RealDatasetSpec, RealPseudoTruthConfig


def _first_present_column(frame: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    """Resolve a biological identifier column across raw and standardized tables."""
    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(
        f"{label} column is missing. Expected one of {list(candidates)}; "
        f"available columns are {list(frame.columns)}"
    )


def _expressed_universe(expression: pd.DataFrame, minimum_fraction: float, maximum_targets: int) -> list[str]:
    values = expression.to_numpy(dtype=float)
    detection = (values > 0).mean(axis=0)
    variance = np.var(np.log1p(values), axis=0)
    frame = pd.DataFrame({"gene": expression.columns.astype(str), "detection": detection, "variance": variance})
    frame = frame[frame["detection"] >= minimum_fraction]
    return frame.sort_values(["variance", "gene"], ascending=[False, True], kind="mergesort").head(maximum_targets)["gene"].tolist()


def _high_confidence_reference(frame: pd.DataFrame, config: RealPseudoTruthConfig) -> pd.DataFrame:
    frame = frame.copy()
    sources = (
        frame["expert_sources"] if "expert_sources" in frame
        else (frame["sources"] if "sources" in frame else pd.Series("", index=frame.index))
    )
    if "curation_effort" in frame:
        curation = frame["curation_effort"]
    elif "curation_effort_numeric" in frame:
        curation = frame["curation_effort_numeric"]
    else:
        curation = pd.Series(0, index=frame.index)
    frame["independent_sources"] = sources.map(split_sources)
    frame["curation_effort_numeric"] = pd.to_numeric(
        curation, errors="coerce"
    ).fillna(0)
    return frame[
        (frame["independent_sources"] >= config.minimum_independent_sources)
        & (frame["curation_effort_numeric"] >= config.minimum_curation_effort)
    ].copy()


def _sample_unlabeled_grn(
    positives: pd.DataFrame, tfs: list[str], targets: list[str], ratio: int, random: np.random.Generator,
) -> pd.DataFrame:
    positive_keys = set(zip(positives.tf, positives.target))
    rows: list[dict[str, object]] = []
    targets_array = np.asarray(targets, dtype=object)
    for tf, group in positives.groupby("tf"):
        candidates = [target for target in targets_array if (tf, str(target)) not in positive_keys and str(target) != tf]
        take = min(len(candidates), ratio * len(group))
        if take:
            for target in random.choice(candidates, take, replace=False):
                rows.append({
                    "tf": tf, "target": str(target), "label": 0, "true_score": 0.0,
                    "signed_truth": 0.0, "candidate_class": "unlabeled_not_confirmed_negative",
                    "independent_sources": 0, "curation_effort": 0,
                })
    return pd.DataFrame(rows)


def _build_grn_truth(
    expression: pd.DataFrame, tf_list: list[str], config: RealPseudoTruthConfig, random: np.random.Generator,
) -> pd.DataFrame:
    reference = _high_confidence_reference(read_table(config.tf_reference), config)
    source = _first_present_column(reference, ("source_genesymbol", "tf", "source"), "TF")
    target = _first_present_column(reference, ("target_genesymbol", "target"), "TF target")
    target_universe = _expressed_universe(expression, config.minimum_detection_fraction, config.maximum_target_universe)
    reference = reference.assign(tf=reference[source].astype(str), target=reference[target].astype(str))
    reference = reference[reference.tf.isin(tf_list) & reference.target.isin(target_universe) & reference.tf.ne(reference.target)]
    stimulation = (
        reference["consensus_stimulation"].astype(str).str.lower().isin(("true", "1", "yes"))
        if "consensus_stimulation" in reference else pd.Series(False, index=reference.index)
    )
    inhibition = (
        reference["consensus_inhibition"].astype(str).str.lower().isin(("true", "1", "yes"))
        if "consensus_inhibition" in reference else pd.Series(False, index=reference.index)
    )
    sign = np.where(stimulation, 1.0, np.where(inhibition, -1.0, np.nan))
    positives = pd.DataFrame({
        "tf": reference.tf, "target": reference.target, "label": 1,
        "true_score": reference["curation_effort_numeric"], "signed_truth": sign,
        "candidate_class": "curated_positive",
        "independent_sources": reference["independent_sources"],
        "curation_effort": reference["curation_effort_numeric"],
    }).sort_values(["tf", "target"]).drop_duplicates(["tf", "target"])
    if positives.empty:
        raise ValueError("No high-confidence TF-target pseudo-truth overlaps the real dataset")
    unlabeled = _sample_unlabeled_grn(positives, tf_list, target_universe, config.decoys_per_positive, random)
    return pd.concat([positives, unlabeled], ignore_index=True)


def _build_lr_truth(
    expression: pd.DataFrame, config: RealPseudoTruthConfig, random: np.random.Generator,
) -> pd.DataFrame:
    genes = set(expression.columns.astype(str))
    if not config.lr_reference.exists():
        raise FileNotFoundError(
            f"Strict LR catalog is missing: {config.lr_reference}. Run "
            "python .\\preprocess\\download_public_resources.py and then "
            "python .\\preprocess\\prepare_curated_lr.py."
        )
    reference_all = read_table(config.lr_reference)
    reference = _high_confidence_reference(reference_all, config)
    ligand_column = _first_present_column(
        reference, ("source_genesymbol", "ligand", "source"), "ligand"
    )
    receptor_column = _first_present_column(
        reference, ("target_genesymbol", "receptor", "target"), "receptor"
    )
    reference = reference.assign(
        ligand=reference[ligand_column].astype(str), receptor=reference[receptor_column].astype(str)
    )
    reference = reference[
        reference.ligand.isin(genes) & reference.receptor.isin(genes) & reference.ligand.ne(reference.receptor)
    ].sort_values(["curation_effort_numeric", "independent_sources"], ascending=False)
    positives = reference.drop_duplicates(["ligand", "receptor"])[
        ["ligand", "receptor", "independent_sources", "curation_effort_numeric"]
    ].rename(columns={"curation_effort_numeric": "curation_effort"})
    positives["lr_id"] = positives.ligand + "__" + positives.receptor
    positives["label"] = 1
    positives["candidate_class"] = "curated_positive"
    if positives.empty:
        raise ValueError("No high-confidence strict LR pseudo-truth overlaps the real dataset")
    all_ligand_column = _first_present_column(
        reference_all, ("source_genesymbol", "ligand", "source"), "ligand"
    )
    all_receptor_column = _first_present_column(
        reference_all, ("target_genesymbol", "receptor", "target"), "receptor"
    )
    all_curated = set(zip(reference_all[all_ligand_column].astype(str),
                          reference_all[all_receptor_column].astype(str)))
    ligands = positives.ligand.to_numpy(dtype=object)
    receptors = positives.receptor.to_numpy(dtype=object)
    decoys: list[dict[str, object]] = []
    attempts = 0
    target_decoys = config.decoys_per_positive * len(positives)
    while len(decoys) < target_decoys and attempts < max(100, 50 * target_decoys):
        ligand = str(random.choice(ligands))
        receptor = str(random.choice(receptors))
        attempts += 1
        if ligand == receptor or (ligand, receptor) in all_curated:
            continue
        decoys.append({
            "ligand": ligand, "receptor": receptor, "lr_id": ligand + "__" + receptor,
            "label": 0, "candidate_class": "degree_matched_decoy",
            "independent_sources": 0, "curation_effort": 0,
        })
    decoy_frame = pd.DataFrame(decoys).drop_duplicates(["ligand", "receptor"])
    return pd.concat([positives, decoy_frame], ignore_index=True)


def _prepare_one(spec: RealDatasetSpec, config: RealPseudoTruthConfig) -> Path:
    source = spec.input_dir
    expression = pd.read_csv(source / "expression.csv", index_col=0)
    tf_table = pd.read_csv(source / "tf_list.csv")
    tf_column = "tf" if "tf" in tf_table else tf_table.columns[0]
    tfs = [tf for tf in tf_table[tf_column].astype(str).drop_duplicates() if tf in expression.columns]
    random = np.random.default_rng(config.random_seed + sum(map(ord, spec.dataset_id)))
    grn_truth = _build_grn_truth(expression, tfs, config, random)
    lr_truth = _build_lr_truth(expression, config, random)
    root = config.output_root / spec.dataset_id
    truth_dir = root / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)
    for name in ("expression.csv", "coordinates.csv", "metadata.csv", "tf_list.csv"):
        copy_or_link(source / name, root / name)
    # The model sees candidate identities but not pseudo-truth labels/confidence.
    lr_input = lr_truth[["lr_id", "ligand", "receptor"]].drop_duplicates().copy()
    lr_input["signal_type"] = "unknown"
    lr_input["rho"] = 1.0
    lr_input["confidence"] = np.nan
    lr_input.to_csv(root / "lr_pairs.csv", index=False)
    if config.withhold_curated_tf_prior:
        prior_path = source / "prior_tf_gene.csv"
        prior = pd.read_csv(prior_path) if prior_path.exists() else pd.DataFrame()
        if not prior.empty:
            for column in ("grn",):
                if column in prior:
                    prior[column] = np.nan
            evidence_columns = [column for column in ("motif", "atac") if column in prior]
            prior = prior[prior[evidence_columns].notna().any(axis=1)] if evidence_columns else prior.iloc[0:0]
            prior.to_csv(root / "prior_tf_gene.csv", index=False)
            pd.DataFrame(columns=["lr_id", "tf", "score"]).to_csv(root / "prior_lr_tf.csv", index=False)
        else:
            write_empty_priors(root)
    else:
        copy_or_link(source / "prior_tf_gene.csv", root / "prior_tf_gene.csv")
        copy_or_link(source / "prior_lr_tf.csv", root / "prior_lr_tf.csv")
    grn_truth.to_csv(truth_dir / "truth_pseudo_grn.csv", index=False)
    lr_truth.to_csv(truth_dir / "truth_pseudo_lr.csv", index=False)
    pd.DataFrame([{
        "warning": "Zeros are unlabeled or constructed decoys, not experimentally proven negatives.",
        "tf_prior_withheld": config.withhold_curated_tf_prior,
        "grn_positives": int(grn_truth.label.sum()), "lr_positives": int(lr_truth.label.sum()),
    }]).to_csv(truth_dir / "pseudotruth_audit.csv", index=False)
    write_baseline_input(root)
    return finalize_dataset(
        root, dataset_id=f"real_pseudo__{spec.dataset_id}", evidence_track="real_pseudotruth",
        source=str(source), evaluation_scope="real_pseudotruth", truth_kind="positive_unlabeled",
        species=spec.species, tissue=spec.tissue,
        truth_files=("truth_pseudo_grn.csv", "truth_pseudo_lr.csv", "pseudotruth_audit.csv"),
        notes="High-confidence literature positives plus unlabeled/degree-matched controls. Report PU metrics, not gold-standard accuracy.",
    )


def run_real(config: RealPseudoTruthConfig = REAL_CONFIG) -> Path:
    manifests = [_prepare_one(spec, config) for spec in config.datasets]
    registry = rebuild_registry()
    print(f"[prepared real pseudo-truth] {len(manifests)} datasets; registry={registry}")
    return registry



