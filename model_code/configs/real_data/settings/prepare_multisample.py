from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from grn2_project.paths import *

@dataclass
class MultiSampleConfig:
    # Add only independent samples processed to the project data contract.
    sample_dirs: tuple[Path, ...] = ()
    sample_ids: tuple[str, ...] = ()
    platform_labels: tuple[str, ...] = ()
    output_dir: Path = PROJECT_ROOT / "data/processed/multisample/mouse_brain_cohort"
    minimum_samples: int = 2
    require_same_platform: bool = True
    gene_join: str = "intersection"  # intersection is safest across samples.
    require_same_lr_definition: bool = True


CONFIG = MultiSampleConfig()

