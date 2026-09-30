from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from grn2_project.paths import *

@dataclass
class MERFISHAnnotationConfig:
    raw_expression_table: Path = (
        PROJECT_ROOT
        / "data/raw/real_data/merfish_mouse_brain_s2r3/STalign_S2R3_to_Visium.csv.gz"
    )
    annotation_dir: Path = (
        PROJECT_ROOT / "data/raw/real_data/merfish_mouse_brain_s2r3/annotations"
    )
    structure_file_name: str = "STalign_S2R3_with_structure_id_name.csv.gz"
    cluster_cell_type_file_name: str = "STalign_S2R3_cell_type_annotations.csv.gz"
    biological_cell_type_file_name: str = (
        "STalign_celltypeannotations_merfishslices_v2.csv.gz"
    )
    processed_dirs: tuple[Path, ...] = (
        PROJECT_ROOT / "data/processed/pilot/merfish_mouse_brain_s2r3",
        PROJECT_ROOT / "data/processed/full/merfish_mouse_brain_s2r3",
    )
    skip_missing_processed_dirs: bool = True
    id_significant_digits: int = 13
    background_acronyms: tuple[str, ...] = ("bg", "background", "0")
    master_output_name: str = "s2r3_cell_annotations.parquet"
    audit_output_name: str = "s2r3_annotation_audit.csv"
    manifest_name: str = "s2r3_annotation_manifest.json"


CONFIG = MERFISHAnnotationConfig()

