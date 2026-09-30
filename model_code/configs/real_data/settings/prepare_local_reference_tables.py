from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from grn2_project.paths import *

@dataclass
class ReferenceTableConfig:
    manual_dir: Path = RAW_DATA_ROOT / "manual_downloads" / "omnipath_mouse"
    collectri_file: Path = manual_dir / "collectri_mouse.tsv"
    ligand_receptor_file: Path = manual_dir / "ligrecextra_mouse.tsv"
    output_dir: Path = PROCESSED_DATA_ROOT / "reference_tables" / "omnipath_mouse_full"
    organism: str = "mouse"
    taxonomy_id: int = 10090
    exclude_complex_symbols: bool = True


CONFIG = ReferenceTableConfig()

