"""SERGIO 的输入、模拟参数、模型参数和输出位置。"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from configs.paths import *
from configs.common.experiment import ExperimentConfig, InputSelection, RunOptions
from configs.common.presets import ModelParameters
from configs.public_sim_data.settings import SergioConfig
from configs.common.model import DataConfig, BaseGRNConfig, RewiringConfig


@dataclass
class NonSpatialDataConfig(DataConfig):
    is_spatial: bool = False
    # Match the existing tree baselines' full input gene panel (no QC filtering).
    minimum_detection_rate: float = 0.0
    minimum_total_count: float = 0.0
    confounder_columns: tuple[str, ...] = ()


@dataclass
class GlobalGRNConfig(BaseGRNConfig):
    maximum_tfs: int = 100
    maximum_targets: int = 100
    global_max_iterations: int = 5000
    global_tolerance: float = 1e-6
    global_adjust_covariates: bool = False


@dataclass
class GRNOnlyConfig(RewiringConfig):
    disable_communication_rewiring: bool = True
    joint_iterations: int = 1


@dataclass
class SergioModelParameters(ModelParameters):
    data: DataConfig = field(default_factory=NonSpatialDataConfig)
    base_grn: BaseGRNConfig = field(default_factory=GlobalGRNConfig)
    rewiring: RewiringConfig = field(default_factory=GRNOnlyConfig)


@dataclass
class DataParameters(SergioConfig):
    official_repo: Path = SOURCE_PROJECT / "vendor/external_simulators/SERGIO"
    output_dir: Path = LOCAL_PROCESSED / "evidence_benchmark/02_prepared/public_simulators/sergio_grn"
    replicate_seeds: tuple[int, ...] = (20260902, 20260903, 20260904, 20260905, 20260906)


@dataclass
class DatasetParameters(InputSelection):
    source: str = "existing"
    dataset_ids: tuple[str, ...] = tuple(f"sergio_grn_100g_seed_{s}" for s in DataParameters.replicate_seeds)


@dataclass
class RunParameters(RunOptions):
    output_root: Path = LOCAL_RESULTS / "public_sim_data/sergio_nonspatial_v2"
    frozen_parameters_file: Path | None = OLD_RESULTS / "benchmark/core_final_v1/final/02_runs/frozen_parameters.json"


def build_config():
    return ExperimentConfig("sergio", "public_sim_data", "evidence", "sergio",
                            DatasetParameters(), RunParameters(), SergioModelParameters(), DataParameters())
