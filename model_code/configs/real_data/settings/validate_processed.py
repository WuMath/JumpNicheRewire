from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from grn2_project.paths import *

@dataclass
class ValidationConfig:
    input_dirs: tuple[Path, ...] = (
        PROJECT_ROOT / "data/processed/pilot/visium_ffpe_mouse_brain",
        PROJECT_ROOT / "data/processed/pilot/merfish_mouse_brain_s2r3",
    )
    write_excel_report: bool = True
    report_name: str = "validation_report.xlsx"


CONFIG = ValidationConfig()

