"""操作与数据选择配置。模型的全部参数见 common/model.py。"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from configs.paths import OLD_PROCESSED, LOCAL_PROCESSED, LOCAL_RESULTS
from configs.common.model import ProjectConfig


@dataclass
class InputSelection:
    # existing: 只读复用旧的标准化数据；prepared: 使用新预处理结果。
    source: str = "existing"
    existing_registry: Path = OLD_PROCESSED / "evidence_benchmark/03_registries/evidence_registry.csv"
    prepared_registry: Path = LOCAL_PROCESSED / "evidence_benchmark/03_registries/evidence_registry.csv"
    dataset_ids: tuple[str, ...] = ()
    single_input_dir: Path | None = None


@dataclass
class RunOptions:
    output_root: Path = LOCAL_RESULTS
    isolate_datasets: bool = True
    overwrite: bool = False
    # 不为空时，冻结文件中的已选参数覆盖 model 中对应的调参字段。
    # 设为 None 才使用配置类中的手动参数。读取 JSON，不导入旧代码。
    frozen_parameters_file: Path | None = None
    unlock_test: bool = False


@dataclass
class ExperimentConfig:
    name: str
    category: str
    workflow: str  # evidence / single / formal
    processor: str
    inputs: InputSelection = field(default_factory=InputSelection)
    run: RunOptions = field(default_factory=RunOptions)
    model: ProjectConfig = field(default_factory=ProjectConfig)
    preprocessing: Any = None
    conversion: Any = None
    references: Any = None
    formal_data: Any = None
    formal_model: Any = None
    evaluation: Any = None
    jobs: Any = None
    external_import: Any = None
    overwrite_preprocessing: bool = False


@dataclass
class EvidencePreprocessOptions:
    overwrite_prepared: bool = False
    # 参数变动必须显式允许重建，不静默沿用旧预处理。
    acknowledge_perturbation_limitations: bool = False
