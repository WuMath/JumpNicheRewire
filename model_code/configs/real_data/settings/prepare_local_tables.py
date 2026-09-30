from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from grn2_project.paths import *

@dataclass
class LocalTableConfig:
    manual_dir: Path = RAW_DATA_ROOT / "manual_downloads" / "custom_dataset"
    expression_file: Path = manual_dir / "expression_downloaded.csv.gz"
    coordinates_file: Path = manual_dir / "coordinates_downloaded.csv"
    metadata_file: Path | None = manual_dir / "metadata_downloaded.csv"
    tf_file: Path = manual_dir / "tf_list.csv"
    lr_file: Path = manual_dir / "lr_pairs.csv"
    prior_file: Path | None = manual_dir / "prior_tf_gene.csv"
    output_dir: Path = PROCESSED_DATA_ROOT / "imported" / "custom_manual_prepared"

    expression_orientation: str = "cells_by_genes"  # or genes_by_cells
    expression_id_column: str | None = None  # None means first column
    coordinate_id_column: str | None = None
    coordinate_columns: tuple[str, ...] = ("x", "y")
    metadata_id_column: str | None = None
    maximum_cells: int | None = None
    maximum_genes: int | None = 1500
    maximum_tfs: int = 100
    maximum_lr_pairs: int = 300
    minimum_detected_cells: int = 8
    random_seed: int = 20260823
    default_sample_id: str = "sample_1"
    default_cell_type: str = "NA"
    default_region: str = "global"
    default_condition: str = "unspecified"


CONFIG = LocalTableConfig()

