"""Convert official sCCIgen output and its implanted CCI patterns."""

from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from .common import finalize_dataset, read_table, rebuild_registry, split_sources, write_baseline_input, write_empty_priors
from .config import SCCIGEN_CONFIG, SCCIGenConfig


def _metadata_unit_column(meta: pd.DataFrame) -> str:
    """Find the cell/spot identifier without falling back to a numeric row index."""

    candidates = ("Cell", "Spot", "cell_id", "spot_id", "cell", "spot")
    for candidate in candidates:
        if candidate in meta.columns:
            return candidate
    raise ValueError(
        "sCCIgen metadata needs an explicit Cell/Spot identifier column; "
        f"found columns: {list(meta.columns)}"
    )


def _align_metadata_to_counts(count: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.Index, pd.DataFrame]:
    """Require a one-to-one count/metadata unit mapping for benchmark validity."""

    unit_column = _metadata_unit_column(meta)
    count_units = pd.Index(count.columns.astype(str).str.strip(), name="cell_id")
    metadata_units = meta[unit_column].astype(str).str.strip()
    if count_units.has_duplicates:
        duplicates = count_units[count_units.duplicated()].unique().tolist()[:5]
        raise ValueError(f"sCCIgen count has duplicate cell/spot IDs: {duplicates}")
    if metadata_units.duplicated().any():
        duplicates = metadata_units[metadata_units.duplicated()].unique().tolist()[:5]
        raise ValueError(f"sCCIgen metadata has duplicate cell/spot IDs: {duplicates}")
    count_set = set(count_units)
    metadata_set = set(metadata_units)
    missing_metadata = [unit for unit in count_units if unit not in metadata_set]
    extra_metadata = [unit for unit in metadata_units if unit not in count_set]
    if missing_metadata or extra_metadata:
        raise ValueError(
            "sCCIgen count and metadata are not a one-to-one matched simulation output: "
            f"count_units={len(count_units)}, metadata_units={len(metadata_units)}, "
            f"count_without_metadata={len(missing_metadata)}, "
            f"metadata_without_count={len(extra_metadata)}. "
            "Do not truncate or align these files by row order: that would attach expression "
            "to the wrong coordinates/cell composition. Download a commit-pinned matched set. "
            f"Examples count_without_metadata={missing_metadata[:5]}, "
            f"metadata_without_count={extra_metadata[:5]}."
        )
    aligned = meta.assign(_unit_id=metadata_units).set_index("_unit_id").loc[count_units].copy()
    return count_units, aligned


def _cell_type_labels(meta: pd.DataFrame, pattern: pd.DataFrame) -> pd.Series:
    if "annotation" in meta.columns:
        return meta["annotation"].fillna("unassigned").astype(str)
    pattern_types = pd.unique(
        pd.concat([pattern.get("CellType", pd.Series(dtype=str)), pattern.get("AdjCellType", pd.Series(dtype=str))])
        .dropna().astype(str)
    )
    composition_columns = [column for column in pattern_types if column in meta.columns]
    if not composition_columns:
        return pd.Series("unassigned", index=meta.index, dtype=str)
    composition = meta[composition_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    labels = composition.idxmax(axis=1).astype(str)
    labels.loc[composition.sum(axis=1).le(0)] = "unassigned"
    return labels


def _balanced_truth_subset(frame: pd.DataFrame, maximum_rows: int, seed: int) -> pd.DataFrame:
    """Seeded round-robin sampling prevents one cell-type direction dominating truth."""

    if len(frame) <= maximum_rows:
        return frame.copy()
    random = np.random.default_rng(seed)
    queues: list[pd.DataFrame] = []
    for _, group in frame.groupby(["sender_cell_type", "receiver_cell_type"], sort=True):
        order = random.permutation(len(group))
        queues.append(group.iloc[order].reset_index(drop=True))
    cursors = [0] * len(queues)
    selected: list[pd.Series] = []
    while len(selected) < maximum_rows:
        progressed = False
        for index, queue in enumerate(queues):
            if cursors[index] < len(queue):
                selected.append(queue.iloc[cursors[index]])
                cursors[index] += 1
                progressed = True
                if len(selected) == maximum_rows:
                    break
        if not progressed:
            break
    return pd.DataFrame(selected).reset_index(drop=True)


def _run_official(config: SCCIGenConfig) -> None:
    if not config.official_parameter_file.exists():
        raise FileNotFoundError(config.official_parameter_file)
    parameter = repr(str(config.official_parameter_file).replace("\\", "/"))
    model = "NULL" if config.official_model_fit_file is None else repr(str(config.official_model_fit_file).replace("\\", "/"))
    expression = (
        "library(sCCIgen);"
        f"ParaSimulation(input={parameter}, ModelFitFile={model})"
    )
    subprocess.run([config.rscript_executable, "-e", expression], check=True)


def _clean_reference(config: SCCIGenConfig, genes: set[str], truth_pairs: set[tuple[str, str]]) -> pd.DataFrame:
    reference = read_table(config.lr_reference)
    ligand_col = "source_genesymbol" if "source_genesymbol" in reference else "source"
    receptor_col = "target_genesymbol" if "target_genesymbol" in reference else "target"
    sources = reference["sources"] if "sources" in reference else pd.Series("", index=reference.index)
    effort = reference["curation_effort"] if "curation_effort" in reference else pd.Series(0, index=reference.index)
    reference = reference.assign(
        ligand=reference[ligand_col].astype(str), receptor=reference[receptor_col].astype(str),
        source_count=sources.map(split_sources),
        effort=pd.to_numeric(effort, errors="coerce").fillna(0),
    )
    reference = reference[
        reference["ligand"].isin(genes) & reference["receptor"].isin(genes)
        & reference["ligand"].ne(reference["receptor"])
    ].copy()
    reference["is_truth_pair"] = [pair in truth_pairs for pair in zip(reference["ligand"], reference["receptor"])]
    reference = reference.sort_values(
        ["is_truth_pair", "source_count", "effort"], ascending=False, kind="mergesort"
    ).drop_duplicates(["ligand", "receptor"])
    return reference


def run_sccigen(config: SCCIGenConfig = SCCIGEN_CONFIG) -> Path:
    if config.run_official_r_simulation:
        _run_official(config)
    count = read_table(config.raw_dir / config.count_file)
    meta = read_table(config.raw_dir / config.metadata_file)
    pattern = read_table(config.raw_dir / config.pattern_file)
    gene_column = "GeneName" if "GeneName" in count.columns else count.columns[0]
    count[gene_column] = count[gene_column].astype(str)
    count = count[~count[gene_column].str.startswith("Blank.", na=False)]
    count = count.groupby(gene_column, as_index=True).sum(numeric_only=True)
    cells, meta = _align_metadata_to_counts(count, meta)
    expression = count.loc[:, cells].T
    expression.index.name = "cell_id"
    expression.columns = expression.columns.astype(str)
    genes = set(expression.columns)
    cci = pattern[pattern["Type"].astype(str).eq("ExprAssoGenes")].copy()
    cci = cci.dropna(subset=["CellType", "GeneID", "AdjCellType", "AdjGene"])
    cci = cci[cci["GeneID"].astype(str).isin(genes) & cci["AdjGene"].astype(str).isin(genes)]
    # In the official sCCIgen implementation, expression of AdjGene in
    # AdjCellType changes GeneID in CellType.  Therefore Adj* is the sender
    # side and GeneID/CellType is the receiver side.  These are generic
    # synthetic cross-cell gene pairs, not necessarily biological LR pairs.
    cci = cci.assign(
        sender_cell_type=cci["AdjCellType"].astype(str), receiver_cell_type=cci["CellType"].astype(str),
        ligand=cci["AdjGene"].astype(str), receptor=cci["GeneID"].astype(str),
        true_score=pd.to_numeric(cci["beta"], errors="coerce").abs(),
        signed_truth=pd.to_numeric(cci["beta"], errors="coerce"), label=1,
    )
    cci["lr_id"] = cci["ligand"] + "__" + cci["receptor"]
    cci = _balanced_truth_subset(cci, config.maximum_truth_pairs, config.random_seed)
    positives = set(zip(cci["ligand"], cci["receptor"]))
    lr_reference = _clean_reference(config, genes, positives)
    truth_lr = cci[["ligand", "receptor"]].drop_duplicates()
    decoy_slots = max(config.maximum_lr_pairs - len(truth_lr), 0)
    reference_decoys = lr_reference[~lr_reference["is_truth_pair"]][["ligand", "receptor"]].head(decoy_slots)
    lr_pairs = pd.concat([
        truth_lr, reference_decoys,
    ], ignore_index=True).drop_duplicates().head(config.maximum_lr_pairs)
    lr_pairs["lr_id"] = lr_pairs["ligand"] + "__" + lr_pairs["receptor"]
    lr_pairs["signal_type"] = "synthetic_cross_cell_pair"
    lr_pairs["rho"] = 1.0
    lr_pairs["confidence"] = np.nan
    # Candidate negatives have the same observed sender/receiver cell-type pairs
    # but LR identities not implanted by sCCIgen.
    positive_keys = set(zip(cci.sender_cell_type, cci.receiver_cell_type, cci.ligand, cci.receptor))
    cell_type_pairs = cci[["sender_cell_type", "receiver_cell_type"]].drop_duplicates()
    negative_rows = []
    for cell_types in cell_type_pairs.itertuples(index=False):
        for lr in lr_pairs.itertuples(index=False):
            key = (cell_types.sender_cell_type, cell_types.receiver_cell_type, lr.ligand, lr.receptor)
            if key not in positive_keys:
                negative_rows.append({
                    "sender_cell_type": key[0], "receiver_cell_type": key[1],
                    "ligand": key[2], "receptor": key[3], "lr_id": lr.lr_id,
                    "label": 0, "true_score": 0.0, "signed_truth": 0.0,
                })
    random = np.random.default_rng(config.random_seed)
    maximum_negatives = max(5 * len(cci), len(cci))
    if len(negative_rows) > maximum_negatives:
        negative_rows = [negative_rows[index] for index in random.choice(len(negative_rows), maximum_negatives, replace=False)]
    truth = pd.concat([
        cci[["sender_cell_type", "receiver_cell_type", "ligand", "receptor", "lr_id", "label", "true_score", "signed_truth"]],
        pd.DataFrame(negative_rows),
    ], ignore_index=True).drop_duplicates(["sender_cell_type", "receiver_cell_type", "ligand", "receptor"])
    root = config.output_dir
    truth_dir = root / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)
    expression.to_csv(root / "expression.csv")
    cell_types = _cell_type_labels(meta, pattern)
    metadata = pd.DataFrame({
        "cell_id": cells,
        "sample_id": config.sample_id,
        "cell_type": cell_types.to_numpy(),
        "region": meta["region"].astype(str).to_numpy() if "region" in meta else "global",
        "condition": "simulated",
    })
    metadata.to_csv(root / "metadata.csv", index=False)
    x_col = "x.loc" if "x.loc" in meta else "x"
    y_col = "y.loc" if "y.loc" in meta else "y"
    pd.DataFrame({
        "cell_id": cells,
        "x": pd.to_numeric(meta[x_col], errors="raise").to_numpy(),
        "y": pd.to_numeric(meta[y_col], errors="raise").to_numpy(),
    }).set_index("cell_id").to_csv(root / "coordinates.csv")
    tf_ref = read_table(config.tf_reference)
    tf_col = "source_genesymbol" if "source_genesymbol" in tf_ref else "source"
    tfs = [gene for gene in pd.unique(tf_ref[tf_col].astype(str)) if gene in genes][: config.maximum_tfs]
    if not tfs:
        raise ValueError("No TF reference genes overlap sCCIgen expression")
    pd.DataFrame({"tf": tfs}).to_csv(root / "tf_list.csv", index=False)
    lr_pairs[["lr_id", "ligand", "receptor", "signal_type", "rho", "confidence"]].to_csv(root / "lr_pairs.csv", index=False)
    write_empty_priors(root)
    truth.to_csv(truth_dir / "truth_sccigen_cci.csv", index=False)
    pattern[pattern["Type"].astype(str).eq("DistanceAssoGenes")].to_csv(
        truth_dir / "truth_sccigen_distance.csv", index=False
    )
    write_baseline_input(root)
    manifest = finalize_dataset(
        root, dataset_id=config.dataset_id, evidence_track="public_simulator",
        source=config.source_label, evaluation_scope="sccigen_cci",
        truth_kind="exact", species=config.species, tissue=config.tissue,
        truth_files=("truth_sccigen_cci.csv", "truth_sccigen_distance.csv"),
        notes=(
            "Exact implanted ExprAssoGenes truth. Evaluation aggregates cell edges to directed cell-type/gene-pair "
            "groups. Gene pairs are synthetic cross-cell effects and must not be called curated biological LR pairs."
        ),
    )
    rebuild_registry()
    print(f"[prepared sCCIgen] {manifest}")
    return manifest



