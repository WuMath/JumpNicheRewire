r"""Build a mouse TF-target motif prior from the local gene-based cisTarget DB.

Edit ``CisTargetConfig`` and run ``python .\preprocess\prepare_cistarget.py``.
The script reads the existing 10 kb rankings and motif-to-TF annotation, keeps
the strongest targets per TF, then outer-merges motif evidence with the local
CollecTRI/DoRothEA prior.  Missing promoter/ATAC evidence remains NaN.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from preprocess.common import PROJECT_ROOT, file_digest, write_json
from preprocess.evidence_common import (
    normalized_gene_lookup,
    update_summary_item,
    write_frame_atomic,
)


from configs.real_data.settings.prepare_cistarget import CisTargetConfig, CONFIG





def _read_rankings(config: CisTargetConfig) -> tuple[list[str], np.ndarray, np.ndarray]:
    try:
        import pyarrow.feather as feather
    except (ImportError, OSError) as error:
        raise ImportError(
            "cisTarget processing could not load pyarrow. This is often a Windows DLL/ABI "
            "conflict caused by installing a pip pyarrow over Conda arrow-cpp. The project "
            "environment is tested with the matched Conda pyarrow/arrow-cpp 24.0.0 pair; "
            "repair that pair before rerunning this step."
        ) from error

    # Read schema from the complete file because selecting one column hides the
    # database gene columns in some pyarrow versions.
    import pyarrow as pa
    import pyarrow.ipc as ipc

    with pa.memory_map(str(config.ranking_file), "r") as source:
        full_schema = ipc.open_file(source).schema
    columns = list(full_schema.names)
    if config.motif_id_column_in_rankings not in columns:
        raise ValueError(
            f"cisTarget ranking file has no '{config.motif_id_column_in_rankings}' column"
        )
    target_genes = [name for name in columns if name != config.motif_id_column_in_rankings]
    table = feather.read_table(
        config.ranking_file,
        columns=target_genes + [config.motif_id_column_in_rankings],
        memory_map=config.memory_map,
        use_threads=True,
    ).combine_chunks()
    motif_ids = np.asarray(
        table[config.motif_id_column_in_rankings].to_pylist(), dtype=str
    )
    # Rank columns are int16.  Column-stacking keeps peak memory predictable and
    # avoids a 24k-column pandas DataFrame with large Python overhead.
    rank_matrix = np.column_stack(
        [table[name].to_numpy(zero_copy_only=False) for name in target_genes]
    )
    return target_genes, motif_ids, rank_matrix


def build_motif_prior(
    ranking_genes: list[str],
    motif_ids: np.ndarray,
    rank_matrix: np.ndarray,
    annotation: pd.DataFrame,
    tf_names: list[str],
    config: CisTargetConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    motif_column = config.motif_id_column_in_annotation
    if motif_column not in annotation.columns and motif_column.startswith("#"):
        without_hash = motif_column[1:]
        if without_hash in annotation.columns:
            annotation = annotation.rename(columns={without_hash: motif_column})
    required = {motif_column, config.tf_column_in_annotation}
    missing = required.difference(annotation.columns)
    if missing:
        raise ValueError(f"Motif annotation is missing columns: {sorted(missing)}")

    tf_lookup = normalized_gene_lookup(tf_names)
    annotation = annotation[[motif_column, config.tf_column_in_annotation]].dropna().copy()
    annotation["tf_key"] = (
        annotation[config.tf_column_in_annotation].astype(str).str.strip().str.upper()
    )
    annotation["tf"] = annotation["tf_key"].map(tf_lookup)
    annotation["motif_id"] = annotation[motif_column].astype(str).str.strip()
    annotation = annotation.dropna(subset=["tf"]).drop_duplicates(["tf", "motif_id"])

    motif_rows: dict[str, list[int]] = {}
    for row_index, motif_id in enumerate(motif_ids):
        motif_rows.setdefault(str(motif_id), []).append(row_index)
    annotation["ranking_rows"] = annotation["motif_id"].map(motif_rows)
    annotation = annotation.loc[annotation["ranking_rows"].notna()].copy()
    rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    total_database_genes = len(ranking_genes)
    denominator = max(total_database_genes - 1, 1)
    score_cutoff_rank = int(np.floor((1.0 - config.minimum_motif_score) * denominator))

    for tf in tf_names:
        mapped = annotation.loc[annotation["tf"].eq(tf)]
        row_indices = sorted(
            {
                index
                for indices in mapped["ranking_rows"]
                for index in list(indices)
            }
        )
        if len(row_indices) < config.minimum_motifs_per_tf:
            audit_rows.append(
                {
                    "tf": tf,
                    "mapped_motifs": len(row_indices),
                    "retained_targets": 0,
                    "status": "no_usable_motif",
                }
            )
            continue
        ranks = rank_matrix[row_indices, :].astype(np.int64, copy=False)
        best_rank = ranks.min(axis=0)
        keep = np.flatnonzero(best_rank <= score_cutoff_rank)
        maximum = config.maximum_targets_per_tf
        if maximum is not None and maximum > 0 and len(keep) > maximum:
            local = np.argpartition(best_rank[keep], maximum - 1)[:maximum]
            keep = keep[local]
        keep = keep[np.argsort(best_rank[keep], kind="stable")]
        if not len(keep):
            audit_rows.append(
                {
                    "tf": tf,
                    "mapped_motifs": len(row_indices),
                    "retained_targets": 0,
                    "status": "below_score_threshold",
                }
            )
            continue
        selected_ranks = ranks[:, keep]
        best_local_row = selected_ranks.argmin(axis=0)
        frame = pd.DataFrame(
            {
                "tf": tf,
                "target": np.asarray(ranking_genes, dtype=object)[keep],
                "motif_10kb": 1.0 - best_rank[keep] / denominator,
                "motif_500bp": np.nan,
                "motif": 1.0 - best_rank[keep] / denominator,
                "motif_count": (selected_ranks <= score_cutoff_rank).sum(axis=0),
            }
        )
        if config.keep_best_motif_id:
            frame["best_motif_id"] = motif_ids[
                np.asarray(row_indices, dtype=int)[best_local_row]
            ]
        rows.append(frame)
        audit_rows.append(
            {
                "tf": tf,
                "mapped_motifs": len(row_indices),
                "retained_targets": len(frame),
                "status": "used",
            }
        )

    motif_prior = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if motif_prior.empty:
        raise ValueError("No TF-target motif edge passed the configured cisTarget threshold")
    motif_prior = motif_prior.sort_values(
        ["tf", "motif", "target"], ascending=[True, False, True]
    ).reset_index(drop=True)
    return motif_prior, pd.DataFrame(audit_rows)


def merge_with_reference_prior(prior: pd.DataFrame, motif_prior: pd.DataFrame) -> pd.DataFrame:
    required = {"tf", "target", "motif", "atac", "grn"}
    missing = required.difference(prior.columns)
    if missing:
        raise ValueError(f"Reference TF-target prior is missing columns: {sorted(missing)}")
    motif_edges = motif_prior[["tf", "target", "motif"]].rename(
        columns={"motif": "motif_cistarget"}
    )
    merged = prior.merge(motif_edges, on=["tf", "target"], how="outer")
    existing = pd.to_numeric(merged["motif"], errors="coerce")
    incoming = pd.to_numeric(merged.pop("motif_cistarget"), errors="coerce")
    merged["motif"] = pd.concat([existing, incoming], axis=1).max(axis=1, skipna=True)
    both_missing = existing.isna() & incoming.isna()
    merged.loc[both_missing, "motif"] = np.nan
    return merged[["tf", "target", "motif", "atac", "grn"]].sort_values(
        ["tf", "target"]
    ).reset_index(drop=True)


def main(config: CisTargetConfig = CONFIG) -> Path:
    for path in (
        config.ranking_file,
        config.motif_annotation_file,
        config.reference_dir / config.tf_file_name,
        config.reference_dir / config.prior_file_name,
    ):
        if not Path(path).exists():
            raise FileNotFoundError(f"cisTarget input is missing: {path}")

    tf_table = pd.read_csv(config.reference_dir / config.tf_file_name)
    tf_column = "tf" if "tf" in tf_table.columns else tf_table.columns[0]
    tf_names = tf_table[tf_column].dropna().astype(str).drop_duplicates().tolist()
    annotation = pd.read_csv(
        config.motif_annotation_file,
        sep="\t",
        usecols=lambda name: name
        in {config.motif_id_column_in_annotation, config.tf_column_in_annotation},
        low_memory=False,
    )
    print("[1/4] Reading gene-based cisTarget rankings")
    genes, motif_ids, ranks = _read_rankings(config)
    print(f"[cisTarget] {len(motif_ids)} motifs x {len(genes)} genes")
    print("[2/4] Mapping motifs to mouse TFs and retaining high-ranking targets")
    motif_prior, audit = build_motif_prior(
        genes, motif_ids, ranks, annotation, tf_names, config
    )
    del ranks

    output = Path(config.reference_dir)
    prior_path = output / config.prior_file_name
    backup_path = output / config.backup_prior_name
    if not backup_path.exists():
        shutil.copy2(prior_path, backup_path)
    original_prior = pd.read_csv(prior_path)
    merged_prior = merge_with_reference_prior(original_prior, motif_prior)

    print("[3/4] Writing motif evidence and updated outer-joined reference prior")
    write_frame_atomic(motif_prior, output / config.motif_output_name)
    write_frame_atomic(audit, output / config.audit_output_name)
    write_frame_atomic(merged_prior, prior_path)
    update_summary_item(output / "reference_summary.csv", "tf_target_edges", len(merged_prior))
    update_summary_item(output / "reference_summary.csv", "motif_edges", len(motif_prior))
    write_json(
        output / config.manifest_name,
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": asdict(config),
            "ranking_sha256": file_digest(config.ranking_file),
            "annotation_sha256": file_digest(config.motif_annotation_file),
            "database_motifs": len(motif_ids),
            "database_genes": len(genes),
            "mapped_tfs": int(audit["status"].eq("used").sum()),
            "motif_edges": len(motif_prior),
            "merged_prior_edges": len(merged_prior),
            "promoter_500bp_available": False,
            "note": "motif_500bp remains missing; motif currently equals motif_10kb.",
        },
    )
    print(
        audit.groupby("status").size().rename("TFs").to_string(),
        f"\nMotif edges: {len(motif_prior):,}",
        f"\nMerged prior edges: {len(merged_prior):,}",
    )
    print("[4/4] Complete")
    return output / config.motif_output_name



