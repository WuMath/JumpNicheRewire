"""通用真实 H5AD。必须手动指定实际文件及矩阵、坐标、注释键。"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from configs.paths import *
from configs.common.experiment import ExperimentConfig, InputSelection, RunOptions
from configs.common.presets import ModelParameters
from configs.real_data.settings.prepare_local_h5ad import LocalH5ADConfig


@dataclass
class DataParameters(LocalH5ADConfig):
    h5ad_file: Path = OLD_RAW / "real_data/your_dataset.h5ad"
    output_dir: Path = LOCAL_PROCESSED / "real_data/h5ad"
    expression_source: str = "X"
    expression_is_normalized: bool = False
    spatial_key: str = "spatial"
    prepared_tf_file: Path = OLD_PROCESSED / "references/tf_list.csv"
    prepared_lr_file: Path = OLD_PROCESSED / "references/lr_pairs.csv"
    prepared_prior_file: Path = OLD_PROCESSED / "references/prior_tf_gene.csv"


@dataclass
class DatasetParameters(InputSelection):
    source: str = "prepared"
    single_input_dir: Path = DataParameters.output_dir


@dataclass
class RunParameters(RunOptions):
    output_root: Path = LOCAL_RESULTS / "real_data/h5ad"


def build_config():
    model = ModelParameters()
    model.data.expression_is_normalized = DataParameters.expression_is_normalized
    return ExperimentConfig("h5ad", "real_data", "single", "h5ad",
                            DatasetParameters(), RunParameters(), model, DataParameters())
