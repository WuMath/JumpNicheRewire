from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from grn2_project.paths import *

@dataclass
class LocalMERFISHConfig:
    manual_dir: Path = RAW_DATA_ROOT / "manual_downloads" / "merfish_mouse_brain_s2r3"
    expression_table: Path = manual_dir / "STalign_S2R3_to_Visium.csv.gz"
    output_dir: Path = PROCESSED_DATA_ROOT / "imported" / "merfish_manual_prepared"

    reference_mode: str = "prepared_tables"
    collectri_file: Path = manual_dir / "collectri_mouse.tsv"
    ligand_receptor_network_file: Path = manual_dir / "ligrecextra_mouse.tsv"
    prepared_tf_file: Path = PROCESSED_DATA_ROOT / "reference_tables" / "merfish_mouse" / "tf_list.csv"
    prepared_lr_file: Path = PROCESSED_DATA_ROOT / "reference_tables" / "merfish_mouse" / "lr_pairs.csv"
    prepared_prior_file: Path = PROCESSED_DATA_ROOT / "reference_tables" / "merfish_mouse" / "prior_tf_gene.csv"

    row_id_column: str | None = None  # None means the first column
    use_aligned_coordinates: bool = True
    maximum_cells: int | None = 1200
    maximum_genes: int = 500
    maximum_tfs: int = 30
    maximum_lr_pairs: int = 50
    minimum_detected_cells: int = 8
    random_seed: int = 20260823
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


CONFIG = LocalMERFISHConfig()

