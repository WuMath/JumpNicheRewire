"""仅v2接口检查，尚未开放拟合；参数/路径在此修改，无命令行参数。"""
from dataclasses import dataclass, field
from pathlib import Path
from configs.common.experiment import ExperimentConfig, RunOptions
from jump_niche_rewire_v2.adapter import AdapterConfig
from jump_niche_rewire_v2.core import DecompositionConfig

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DecompositionStudyConfig:
    development_registry: Path = ROOT/'data/processed/optimization_20260921/development_registry.csv'
    round01_plan: Path = ROOT/'data/result/optimization_20260921/round_01/01_plan/search_plan.json'
    output_root: Path = ROOT/'data/result/optimization_20260921/design_v2/interface_checks'
    dataset_ids: tuple[str, ...] = tuple(f'published_tree{tree}_500_cells110_genes_sigma{noise}_1'
                                       for tree in (1, 3, 5) for noise in ('0.1', '0.5'))
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    numerical: DecompositionConfig = field(default_factory=DecompositionConfig)
    maximum_new_gib: float = 9.
    minimum_available_ram_gib: float = 1.5
    maximum_worker_rss_gib: float = 3.5
    maximum_task_hours: float = 4.
    minimum_free_disk_gib: float = 5.
    budget_roots: tuple[Path, ...] = tuple(ROOT/f'data/{kind}/{study}' for kind in ('raw', 'processed', 'result')
                                          for study in ('publication_20260920', 'optimization_20260921'))


def build_config():
    cfg = DecompositionStudyConfig()
    return ExperimentConfig('decomposition_v2', 'public_sim_data', 'development', 'decomposition_v2',
                            run=RunOptions(output_root=cfg.output_root), preprocessing=cfg)
