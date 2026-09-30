"""All evaluation parameters; no command-line arguments are used."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from grn2_project.paths import EVALUATION_RESULT_ROOT, MODEL_RESULT_ROOT, PROCESSED_DATA_ROOT, RESULT_DATA_ROOT


@dataclass
class EvaluationConfig:
    truth_dir: Path = PROCESSED_DATA_ROOT / "examples" / "simulated" / "truth"
    result_workbooks: tuple[Path, ...] = (
        MODEL_RESULT_ROOT / "jump_niche_rewire" / "JumpNicheRewire_results.xlsx",
    )
    discover_additional_results: bool = True
    additional_result_directories: tuple[Path, ...] = (
        RESULT_DATA_ROOT / "baselines",
    )
    result_pattern: str = "*.xlsx"
    output_dir: Path = EVALUATION_RESULT_ROOT
    output_workbook: str = "evaluation_summary.xlsx"
    score_threshold: float = 0.0
    random_seed: int = 20260822


CONFIG = EvaluationConfig()
