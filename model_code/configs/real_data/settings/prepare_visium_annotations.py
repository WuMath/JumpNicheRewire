from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from grn2_project.paths import *

@dataclass
class VisiumAnnotationConfig:
    composition_file: Path = (
        PROJECT_ROOT
        / "data/raw/real_data/merfish_mouse_brain_s2r3/annotations/STalign_Visium_cell_type_annotations.csv.gz"
    )
    merfish_master_annotations: Path = (
        PROJECT_ROOT
        / "data/raw/real_data/merfish_mouse_brain_s2r3/annotations/s2r3_cell_annotations.parquet"
    )
    merfish_aligned_expression: Path = (
        PROJECT_ROOT
        / "data/raw/real_data/merfish_mouse_brain_s2r3/STalign_S2R3_to_Visium.csv.gz"
    )
    visium_spatial_source: Path = (
        PROJECT_ROOT
        / "data/raw/real_data/visium_ffpe_mouse_brain/Visium_FFPE_Mouse_Brain_spatial.tar.gz"
    )
    processed_dirs: tuple[Path, ...] = (
        PROJECT_ROOT / "data/processed/pilot/visium_ffpe_mouse_brain",
        PROJECT_ROOT / "data/processed/full/visium_ffpe_mouse_brain",
    )
    skip_missing_processed_dirs: bool = True
    nearest_merfish_cells: int = 30
    maximum_distance_in_spot_diameters: float = 2.0
    minimum_region_neighbors: int = 3
    minimum_region_confidence: float = 0.50
    composition_output_name: str = "cell_composition.csv"
    audit_output_name: str = "visium_annotation_audit.csv"
    manifest_name: str = "visium_annotation_manifest.json"


CONFIG = VisiumAnnotationConfig()

