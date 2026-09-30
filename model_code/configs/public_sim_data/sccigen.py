"""官方 sCCIgen example1；默认处理已手动下载的官方模拟结果。"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from configs.paths import *
from configs.common.experiment import ExperimentConfig, InputSelection, RunOptions
from configs.common.presets import ModelParameters
from configs.public_sim_data.settings import SCCIGenConfig


@dataclass
class DataParameters(SCCIGenConfig):
    raw_dir: Path = OLD_RAW / "evidence_benchmark/01_raw_downloads/public_simulator"
    tf_reference: Path = raw_dir / "Human-TF-Target.tsv"
    lr_reference: Path = raw_dir / "Human-ligand-receptor.tsv"
    output_dir: Path = LOCAL_PROCESSED / "evidence_benchmark/02_prepared/public_simulators/sccigen_example1"
    run_official_r_simulation: bool = False


@dataclass
class DatasetParameters(InputSelection):
    source: str = "existing"
    dataset_ids: tuple[str, ...] = ("sccigen_human_breast_example1",)


@dataclass
class RunParameters(RunOptions):
    output_root: Path = LOCAL_RESULTS / "public_sim_data/sccigen"
    frozen_parameters_file: Path | None = OLD_RESULTS / "benchmark/core_final_v1/final/02_runs/frozen_parameters.json"


def build_config():
    return ExperimentConfig("sccigen", "public_sim_data", "evidence", "sccigen",
                            DatasetParameters(), RunParameters(), ModelParameters(), DataParameters())
