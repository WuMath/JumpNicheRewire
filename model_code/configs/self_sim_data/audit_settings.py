from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from grn2_project.paths import *

@dataclass
class ConvergenceAuditConfig:
    output_dir: Path | None = None
    allowed_splits: tuple[str, ...] = ("train", "validation")

