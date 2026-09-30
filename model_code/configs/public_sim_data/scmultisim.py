"""scMultiSim 独立空间 benchmark。只在这些配置类中手动修改参数。"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from configs.paths import LOCAL_RAW, LOCAL_PROCESSED, LOCAL_RESULTS
from configs.common.experiment import ExperimentConfig, InputSelection, RunOptions
from configs.common.presets import ModelParameters, SmoothGRNConfig
from configs.common.model import BaseGRNConfig, RewiringConfig


@dataclass
class DataParameters:
    rscript: Path = Path("D:/SoftWare/R/R-4.4.1/bin/Rscript.exe")
    expected_package_version: str = "1.2.0"
    raw_root: Path = LOCAL_RAW / "scmultisim_v1"
    output_dir: Path = LOCAL_PROCESSED / "scmultisim_v1"
    # pilot 仅验收；validation 可选参；test 只能用冻结参数。
    split: str = "pilot"
    pilot_seeds: tuple[int, ...] = (101,)
    validation_seeds: tuple[int, ...] = (201, 202, 203)
    test_seeds: tuple[int, ...] = (301, 302, 303, 304, 305)
    scenario: str = "spatial_medium"
    cells: int = 300
    genes: int = 150
    num_cifs: int = 50
    minimum_cells_per_type: int = 15
    grn_resource: str = "GRN_params_100"
    layout: str = "layers"
    grid_size: int = 28
    lr_count: int = 8
    active_lr_min: int = 2
    active_lr_max: int = 4
    cci_effect: float = 5.0
    intrinsic_noise: float = 0.5
    capture_mean: float = 0.1
    capture_sd: float = 0.02
    depth_mean: float = 100000.0
    depth_sd: float = 3000.0
    # simulate: 调用官方 R；prepare: 仅转换已有导出，不重新模拟。
    # 不下载、不替换官方算法、不按真值挑选细胞/基因/LR。
    maximum_truth_rows: int = 1000000

    def seeds(self):
        if self.split not in {"pilot", "validation", "test"}:
            raise ValueError("split must be pilot/validation/test")
        groups = (self.pilot_seeds, self.validation_seeds, self.test_seeds)
        flat = sum(groups, ())
        if len(flat) != len(set(flat)) or not all(groups):
            raise ValueError("三个 split 的种子必须非空、互不重叠")
        return getattr(self, self.split + "_seeds")

    def dataset_ids(self):
        return tuple(f"scmultisim_{self.scenario}_{self.split}_seed_{s}" for s in self.seeds())


@dataclass
class DatasetParameters(InputSelection):
    source: str = "prepared"


@dataclass
class RunParameters(RunOptions):
    output_root: Path = LOCAL_RESULTS / "public_sim_data/scmultisim_v1"
    frozen_parameters_file: Path | None = None
    unlock_test: bool = False


@dataclass
class ScMultiSimGRN(SmoothGRNConfig):
    # 只能在 pilot/validation 修改，test 不调参。
    l1_penalty: float = 0.02
    fusion_penalty: float = 0.15


@dataclass
class ScMultiSimRewiring(RewiringConfig):
    group_penalty: float = 0.002
    ridge_penalty: float = 0.0001


@dataclass
class ScMultiSimModel(ModelParameters):
    base_grn: BaseGRNConfig = field(default_factory=ScMultiSimGRN)
    rewiring: RewiringConfig = field(default_factory=ScMultiSimRewiring)


def build_config():
    data = DataParameters()
    registry = data.output_dir / data.split / "dataset_registry.csv"
    inputs = DatasetParameters(prepared_registry=registry, existing_registry=registry,
                               dataset_ids=data.dataset_ids())
    model = ScMultiSimModel()
    model.data.species = "synthetic"
    model.data.is_spatial = True
    # 使用完整已知候选面板；空间统计仍计算，图约束仍启用。
    # 这样共同 GRN 排名不因仅保留少数空间基因而丢失大量候选。
    model.data.minimum_detection_rate = 0.0
    model.data.minimum_total_count = 0.0
    model.spatial.minimum_spatial_genes = data.genes
    model.spatial.maximum_spatial_genes = data.genes
    model.base_grn.maximum_tfs = data.genes
    model.base_grn.maximum_targets = data.genes
    model.rewiring.maximum_active_lr = data.lr_count
    model.rewiring.maximum_interaction_tfs = data.genes
    model.output.maximum_grn_rows = data.cells * data.genes * data.genes
    return ExperimentConfig("scmultisim", "public_sim_data", "evidence", "scmultisim",
                            inputs, RunParameters(), model, data)
