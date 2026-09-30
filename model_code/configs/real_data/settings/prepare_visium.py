from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from grn2_project.paths import *

@dataclass
class VisiumConfig:
    matrix_h5: Path = (
        PROJECT_ROOT
        / "data/raw/real_data/visium_ffpe_mouse_brain/Visium_FFPE_Mouse_Brain_filtered_feature_bc_matrix.h5"
    )
    # The downloaded H5 has no coordinates.  This existing Space Ranger archive is used.
    spatial_source: Path = (
        PROJECT_ROOT / "data/raw/real_data/visium_ffpe_mouse_brain/Visium_FFPE_Mouse_Brain_spatial.tar.gz"
    )
    reference_dir: Path = PROJECT_ROOT / "data/processed/references"
    output_dir: Path = PROJECT_ROOT / "data/processed/pilot/visium_ffpe_mouse_brain"

    tissue_only: bool = True
    maximum_spots: int | None = 600  # None uses all 2,264 tissue spots.
    maximum_genes: int | None = 600
    maximum_tfs: int | None = 40
    maximum_lr_pairs: int | None = 80
    minimum_detected_spots: int = 8
    random_seed: int = 20260823
    sample_id: str = "Visium_FFPE_Mouse_Brain"
    cell_type_label: str = "NA"
    region_label: str = "global"
    condition_label: str = "control"


CONFIG = VisiumConfig()

