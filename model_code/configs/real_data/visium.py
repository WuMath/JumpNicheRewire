"""Visium 原始 H5+坐标处理，以及真实数据伪真值。MERFISH 不默认运行。"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from configs.paths import *
from configs.common.experiment import ExperimentConfig, InputSelection, RunOptions
from configs.common.presets import ModelParameters
from configs.real_data.settings.prepare_visium import VisiumConfig
from configs.real_data.pseudotruth_settings import RealDatasetSpec, RealPseudoTruthConfig


@dataclass
class VisiumParameters(VisiumConfig):
    matrix_h5: Path = OLD_RAW / "real_data/visium_ffpe_mouse_brain/Visium_FFPE_Mouse_Brain_filtered_feature_bc_matrix.h5"
    spatial_source: Path = OLD_RAW / "real_data/visium_ffpe_mouse_brain/Visium_FFPE_Mouse_Brain_spatial.tar.gz"
    reference_dir: Path = OLD_PROCESSED / "references"
    output_dir: Path = LOCAL_PROCESSED / "full/visium_ffpe_mouse_brain"
    maximum_spots: int | None = None
    maximum_genes: int | None = 2000
    maximum_tfs: int | None = 100
    maximum_lr_pairs: int | None = 200


@dataclass
class DataParameters(RealPseudoTruthConfig):
    # 默认只读复用旧 full 输入，避免重复处理大矩阵。
    # 运行新的原始 H5 转换后，改成 VisiumParameters.output_dir。
    datasets: tuple[RealDatasetSpec, ...] = (
        RealDatasetSpec("visium_ffpe_mouse_brain", OLD_PROCESSED / "full/visium_ffpe_mouse_brain"),
    )
    output_root: Path = LOCAL_PROCESSED / "evidence_benchmark/02_prepared/real_pseudotruth"
    tf_reference: Path = OLD_RAW / "ModelData/Mouse-TF-Target.tsv"
    lr_reference: Path = OLD_PROCESSED / "references/lr_pairs_detailed.csv"


@dataclass
class DatasetParameters(InputSelection):
    source: str = "existing"
    dataset_ids: tuple[str, ...] = ("real_pseudo__visium_ffpe_mouse_brain",)


@dataclass
class RunParameters(RunOptions):
    output_root: Path = LOCAL_RESULTS / "real_data/visium"
    frozen_parameters_file: Path | None = OLD_RESULTS / "benchmark/core_final_v1/final/02_runs/frozen_parameters.json"


def build_config():
    return ExperimentConfig("visium", "real_data", "evidence", "visium",
                            DatasetParameters(), RunParameters(), ModelParameters(),
                            DataParameters(), conversion=VisiumParameters())
