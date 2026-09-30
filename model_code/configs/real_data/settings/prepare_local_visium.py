from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from grn2_project.paths import *

@dataclass
class LocalVisiumConfig:
    """Edit only this class; no script parameters are accepted."""

    manual_dir: Path = RAW_DATA_ROOT / "manual_downloads" / "visium_ffpe_mouse_brain"
    matrix_source: Path = manual_dir / "Visium_FFPE_Mouse_Brain_filtered_feature_bc_matrix.tar.gz"
    spatial_source: Path = manual_dir / "Visium_FFPE_Mouse_Brain_spatial.tar.gz"
    output_dir: Path = PROCESSED_DATA_ROOT / "imported" / "visium_manual_prepared"
    extraction_dir: Path = manual_dir / "_extracted"

    # omnipath_raw: manually save the two OmniPath TSV responses below.
    # prepared_tables: use existing tf_list.csv/lr_pairs.csv/prior_tf_gene.csv.
    reference_mode: str = "prepared_tables"
    collectri_file: Path = manual_dir / "collectri_mouse.tsv"
    ligand_receptor_network_file: Path = manual_dir / "ligrecextra_mouse.tsv"
    prepared_tf_file: Path = PROCESSED_DATA_ROOT / "reference_tables" / "visium_mouse" / "tf_list.csv"
    prepared_lr_file: Path = PROCESSED_DATA_ROOT / "reference_tables" / "visium_mouse" / "lr_pairs.csv"
    prepared_prior_file: Path = PROCESSED_DATA_ROOT / "reference_tables" / "visium_mouse" / "prior_tf_gene.csv"

    tissue_only: bool = True
    maximum_spots: int | None = 600
    maximum_genes: int = 600
    maximum_tfs: int = 40
    maximum_lr_pairs: int = 80
    minimum_detected_spots: int = 8
    random_seed: int = 20260823
    dataset_name: str = "10x_Visium_manual_download"
    sample_id: str = "sample_1"
    cell_type_label: str = "NA"
    region_label: str = "global"
    condition_label: str = "unspecified"


CONFIG = LocalVisiumConfig()

