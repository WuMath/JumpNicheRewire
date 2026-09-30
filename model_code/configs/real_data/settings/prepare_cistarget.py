from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from grn2_project.paths import *

@dataclass
class CisTargetConfig:
    ranking_file: Path = (
        PROJECT_ROOT
        / "data/raw/ModelData/mm10_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather"
    )
    motif_annotation_file: Path = (
        PROJECT_ROOT / "data/raw/ModelData/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl.txt"
    )
    reference_dir: Path = PROJECT_ROOT / "data/processed/references"
    tf_file_name: str = "tf_list.csv"
    prior_file_name: str = "prior_tf_gene.csv"

    motif_id_column_in_rankings: str = "motifs"
    motif_id_column_in_annotation: str = "#motif_id"
    tf_column_in_annotation: str = "gene_name"
    minimum_motif_score: float = 0.95
    maximum_targets_per_tf: int | None = 1000
    minimum_motifs_per_tf: int = 1
    keep_best_motif_id: bool = True
    memory_map: bool = True

    motif_output_name: str = "motif_tf_gene.parquet"
    audit_output_name: str = "motif_mapping_audit.csv"
    backup_prior_name: str = "prior_tf_gene_before_cistarget.csv"
    manifest_name: str = "cistarget_manifest.json"


CONFIG = CisTargetConfig()

