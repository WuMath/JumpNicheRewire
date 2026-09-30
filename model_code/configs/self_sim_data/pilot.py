"""隔离的 pilot：prepare/reuse → tune → audit → freeze → test → evaluate。"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from configs.paths import *
from configs.common.experiment import ExperimentConfig, InputSelection, RunOptions
from configs.common.presets import ModelParameters
from configs.self_sim_data.benchmark_settings import DataStageConfig, ModelStageConfig, AnalysisStageConfig


@dataclass
class DataParameters(DataStageConfig):
    profile: str = "pilot"
    data_root: Path = LOCAL_PROCESSED / "self_sim_data/pilot"
    benchmark_root: Path = LOCAL_RESULTS / "self_sim_data/pilot"


@dataclass
class DatasetParameters(InputSelection):
    source: str = "prepared"
    existing_registry: Path = OLD_PROCESSED / "formal_benchmark/registries/pilot/dataset_registry.csv"
    prepared_registry: Path = DataParameters.data_root / "registries/pilot/dataset_registry.csv"


@dataclass
class RunParameters(RunOptions):
    output_root: Path = DataParameters.benchmark_root
    unlock_test: bool = False


@dataclass
class BenchmarkParameters(ModelStageConfig):
    profile: str = "pilot"
    phase: str = "tune"
    data_root: Path = DataParameters.data_root
    benchmark_root: Path = DataParameters.benchmark_root


@dataclass
class EvaluationParameters(AnalysisStageConfig):
    profile: str = "pilot"
    data_root: Path = DataParameters.data_root
    benchmark_root: Path = DataParameters.benchmark_root


def build_config():
    return ExperimentConfig("pilot", "self_sim_data", "formal", "formal",
                            DatasetParameters(), RunParameters(), ModelParameters(),
                            formal_data=DataParameters(), formal_model=BenchmarkParameters(),
                            evaluation=EvaluationParameters())
