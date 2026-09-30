from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from grn2_project.paths import *

@dataclass
class MERFISHConfig:
    expression_table: Path = (
        PROJECT_ROOT
        / "data/raw/real_data/merfish_mouse_brain_s2r3/STalign_S2R3_to_Visium.csv.gz"
    )
    reference_dir: Path = PROJECT_ROOT / "data/processed/references"
    output_dir: Path = PROJECT_ROOT / "data/processed/pilot/merfish_mouse_brain_s2r3"

    row_id_column: str | None = None  # None uses the first CSV column.
    use_aligned_coordinates: bool = True
    maximum_cells: int | None = 1200  # None uses all 85,958 cells and is expensive downstream.
    maximum_genes: int | None = 500
    maximum_tfs: int | None = 30
    maximum_lr_pairs: int | None = 50
    minimum_detected_cells: int = 8
    random_seed: int = 20260823
    read_chunk_rows: int = 5000
    sample_id: str = "MERFISH_S2R3"
    cell_type_label: str = "NA"
    region_label: str = "global"
    condition_label: str = "unspecified"
    non_gene_columns: tuple[str, ...] = (
        "center_x",
        "center_y",
        "STalign_x",
        "STalign_y",
        "Pmatch",
    )
    excluded_gene_prefixes: tuple[str, ...] = ("Blank.",)


CONFIG = MERFISHConfig()

