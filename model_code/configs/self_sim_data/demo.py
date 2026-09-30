"""很小的工程通路测试，不属于正式 benchmark，不用于论文评价。"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from configs.paths import *
from configs.common.experiment import ExperimentConfig, InputSelection, RunOptions
from configs.common.model import ProjectConfig, SpatialConfig, BaseGRNConfig, RewiringConfig, RuntimeConfig
from configs.common.presets import CompactOutputConfig
from configs.self_sim_data.simulation_settings import SimulationConfig


@dataclass
class DataParameters(SimulationConfig):
    output_dir: Path = LOCAL_PROCESSED / "self_sim_data/demo"
    region_count: int = 2
    cells_per_region: int = 12
    tf_count: int = 3
    target_count: int = 4
    lr_count: int = 2
    noise_gene_count: int = 3
    communication_neighbors: int = 6


@dataclass
class DemoSpatial(SpatialConfig):
    spatial_permutations: int = 5
    minimum_spatial_genes: int = 8
    maximum_spatial_genes: int = 30
    screening_neighbors: int = 4
    graph_neighbors: int = 5


@dataclass
class DemoGRN(BaseGRNConfig):
    maximum_tfs: int = 3
    maximum_targets: int = 8
    admm_max_iterations: int = 30
    cg_max_iterations: int = 80


@dataclass
class DemoRewiring(RewiringConfig):
    maximum_active_lr: int = 2
    maximum_interaction_tfs: int = 3
    fista_max_iterations: int = 80
    joint_iterations: int = 2


@dataclass
class DemoRuntime(RuntimeConfig):
    require_convergence: bool = False  # 仅限工程 smoke test，报告会标为非正式。


@dataclass
class ModelParameters(ProjectConfig):
    spatial: SpatialConfig = field(default_factory=DemoSpatial)
    base_grn: BaseGRNConfig = field(default_factory=DemoGRN)
    rewiring: RewiringConfig = field(default_factory=DemoRewiring)
    runtime: RuntimeConfig = field(default_factory=DemoRuntime)
    output: CompactOutputConfig = field(default_factory=CompactOutputConfig)


@dataclass
class DatasetParameters(InputSelection):
    source: str = "prepared"
    single_input_dir: Path = DataParameters.output_dir


@dataclass
class RunParameters(RunOptions):
    output_root: Path = LOCAL_RESULTS / "self_sim_data/demo"


def build_config():
    return ExperimentConfig("demo", "self_sim_data", "single", "demo",
                            DatasetParameters(), RunParameters(), ModelParameters(), DataParameters())
