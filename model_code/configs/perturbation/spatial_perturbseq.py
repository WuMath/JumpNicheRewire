"""GSE274447：转换与准备分步执行；继承旧方法，不暗中改变统计定义。"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from configs.paths import *
from configs.common.experiment import ExperimentConfig, InputSelection, RunOptions
from configs.common.presets import ModelParameters
from configs.perturbation.settings import PerturbationConfig, StereoGEFConversionConfig


@dataclass
class ConversionParameters(StereoGEFConversionConfig):
    raw_archive: Path = OLD_RAW / "evidence_benchmark/01_raw_downloads/perturbation/GSE274447_RAW.tar"
    extracted_dir: Path = LOCAL_RAW / "perturbation/GSE274447_extracted"
    standardized_h5ad: Path = LOCAL_PROCESSED / "perturbation/GSE274447_processed.h5ad"
    sample_map_file: Path = LOCAL_RAW / "perturbation/GSE274447_sample_map.csv"


@dataclass
class DataParameters(PerturbationConfig):
    standardized_h5ad: Path = ConversionParameters.standardized_h5ad
    output_dir: Path = LOCAL_PROCESSED / "evidence_benchmark/02_prepared/perturbation/spatial_perturbseq_mouse_brain"
    tf_reference: Path = OLD_RAW / "ModelData/Mouse-TF-Target.tsv"
    lr_reference: Path = OLD_PROCESSED / "references/lr_pairs.csv"
    # True 仅表示允许复现旧探索性流程，不表示修复了评价泄漏。
    allow_exploratory_legacy_protocol: bool = False


@dataclass
class DatasetParameters(InputSelection):
    source: str = "prepared"
    dataset_ids: tuple[str, ...] = ("spatial_perturbseq_mouse_brain_controls",)


@dataclass
class RunParameters(RunOptions):
    output_root: Path = LOCAL_RESULTS / "perturbation/spatial_perturbseq"
    frozen_parameters_file: Path | None = OLD_RESULTS / "benchmark/core_final_v1/final/02_runs/frozen_parameters.json"


def build_config():
    return ExperimentConfig("spatial_perturbseq", "perturbation", "evidence", "perturbation",
                            DatasetParameters(), RunParameters(), ModelParameters(),
                            DataParameters(), conversion=ConversionParameters())
