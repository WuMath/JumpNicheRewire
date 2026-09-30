"""Frozen subset benchmark: two real spatial datasets, no truth used for fitting."""
from dataclasses import dataclass
from pathlib import Path

from configs.common.experiment import ExperimentConfig, InputSelection, RunOptions
from configs.common.model import ProjectConfig


ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT.parent


@dataclass(frozen=True)
class SupportBenchmarkSettings:
    raw_root: Path = ROOT / 'data/raw/benchmark_public_20260924'
    external_raw: Path = PARENT / 'data/raw/Data0925'
    tf_catalog: Path = PARENT / 'data/processed/references/tf_list.csv'
    motif_catalog: Path = PARENT / 'data/processed/references/motif_tf_gene.parquet'
    output_root: Path = ROOT / 'data/result/real_support_benchmark_20260925'
    rscript: Path = Path('D:/SoftWare/R/R-4.4.1/bin/Rscript.exe')
    spots: int = 500
    seed: int = 20260925
    l1_penalty: float = .075
    fusion_penalty: float = .0005
    group_penalty: float = .01
    ridge_penalty: float = .0001
    spatial_permutations: int = 99
    cumulative_storage_limit: int = 15 * 1024**3
    worker_rss_limit: int = int(3.5 * 1024**3)
    minimum_available_ram: int = int(1.5 * 1024**3)
    task_limit_seconds: int = 4 * 3600
    minimum_disk_free: int = 5 * 1024**3
    maximum_new_gib: float = 15.0
    maximum_worker_rss_gib: float = 3.5
    minimum_available_ram_gib: float = 1.5
    maximum_task_hours: float = 4.0
    minimum_free_disk_gib: float = 5.0
    budget_roots: tuple[Path, ...] = (ROOT, Path('C:/Anaconda/envs/jumpniche'),
                                      Path('C:/Anaconda/envs/jnr-trees-clean'),
                                      Path('C:/Anaconda/envs/jnr-cespgrn'),
                                      PARENT / 'data/raw/Data0925')
    model_python: Path = Path('C:/Anaconda/envs/jumpniche/python.exe')
    tree_python: Path = Path('C:/Anaconda/envs/jnr-trees-clean/python.exe')
    r_library: Path = ROOT / 'data/raw/optimization_20260921/R_library'
    r_software: Path = ROOT / 'data/raw/optimization_20260921/software'


def build_config():
    settings = SupportBenchmarkSettings()
    model = ProjectConfig()
    model.data.species = 'mouse'
    model.base_grn.l1_penalty = .075
    model.base_grn.fusion_penalty = .0005
    model.rewiring.group_penalty = .01
    model.rewiring.ridge_penalty = .0001
    model.base_grn.maximum_tfs = 20
    model.base_grn.maximum_targets = 180
    model.spatial.maximum_spatial_genes = 180
    model.spatial.minimum_spatial_genes = 180
    model.spatial.spatial_permutations = 99
    model.rewiring.maximum_active_lr = 20
    return ExperimentConfig('support_benchmark', 'real_data', 'evidence', 'real_support',
                            InputSelection(source='prepared', dataset_ids=('gse165141_liver', 'gse205055_p22')),
                            RunOptions(output_root=settings.output_root), model, settings)
