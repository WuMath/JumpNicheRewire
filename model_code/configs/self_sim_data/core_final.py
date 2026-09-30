"""4 场景 × (5 train + 3 validation + 5 test)；不复用旧运行缓存。"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from configs.paths import *
from configs.common.experiment import ExperimentConfig, InputSelection, RunOptions
from configs.common.presets import ModelParameters
from configs.self_sim_data.benchmark_settings import (
    DataStageConfig, ModelStageConfig, AnalysisStageConfig, SplitSpec,
    FINAL_SPLIT_SPEC, final_scenarios,
)


@dataclass
class DataParameters(DataStageConfig):
    profile: str = "final"
    data_root: Path = LOCAL_PROCESSED / "self_sim_data/core_final"
    benchmark_root: Path = LOCAL_RESULTS / "self_sim_data/core_final"
    split: SplitSpec = field(default_factory=lambda: FINAL_SPLIT_SPEC)
    scenario_ids: tuple[str, ...] = ("clean_500", "dropout_500", "low_prior_500", "boundary_500")

    def scenarios(self):
        available = {s.name: s for s in final_scenarios()}
        return tuple(available[name] for name in self.scenario_ids)


@dataclass
class DatasetParameters(InputSelection):
    source: str = "prepared"
    existing_registry: Path = OLD_PROCESSED / "formal_benchmark_core_final_v1/registries/final/dataset_registry.csv"
    prepared_registry: Path = DataParameters.data_root / "registries/final/dataset_registry.csv"


@dataclass
class RunParameters(RunOptions):
    output_root: Path = DataParameters.benchmark_root
    unlock_test: bool = False


@dataclass
class BenchmarkParameters(ModelStageConfig):
    profile: str = "final"
    phase: str = "tune"
    data_root: Path = DataParameters.data_root
    benchmark_root: Path = DataParameters.benchmark_root


@dataclass
class EvaluationParameters(AnalysisStageConfig):
    profile: str = "final"
    data_root: Path = DataParameters.data_root
    benchmark_root: Path = DataParameters.benchmark_root


def build_config():
    return ExperimentConfig("core_final", "self_sim_data", "formal", "formal",
                            DatasetParameters(), RunParameters(), ModelParameters(),
                            formal_data=DataParameters(), formal_model=BenchmarkParameters(),
                            evaluation=EvaluationParameters())
