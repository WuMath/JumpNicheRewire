"""One-hour, one-dataset JNR parameter-only development search."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = 'published_tree1_500_cells110_genes_sigma0.1_1'


@dataclass(frozen=True)
class OneHourJNR:
    dataset_id: str = DATASET
    output_root: Path = ROOT/'data/result/optimization_20260921/single_dataset_jnr_tuning_20260923'
    registry: Path = ROOT/'data/processed/optimization_20260921/development_registry.csv'
    python: Path = Path('C:/Anaconda/envs/jumpniche/python.exe')
    hour_limit_seconds: int = 3600
    worker_rss_limit: int = int(3.5*1024**3)
    minimum_available_ram: int = int(1.5*1024**3)
    minimum_disk_free: int = 5*1024**3
    cumulative_storage_limit: int = 9*1024**3
    storage_reserve: int = 256*1024**2
    budget_roots: tuple[Path, ...] = (
        ROOT/'data/raw/publication_20260920',
        ROOT/'data/processed/publication_20260920',
        ROOT/'data/result/publication_20260920',
        ROOT/'data/raw/optimization_20260921',
        ROOT/'data/processed/optimization_20260921',
        ROOT/'data/result/optimization_20260921',
        Path('C:/Anaconda/envs/jnr-cespgrn'),
        ROOT/'vendor/external_methods/RegDiffusion',
    )

    def candidates(self):
        base = dict(l1_penalty=.1,fusion_penalty=.15,group_penalty=.01,
                    ridge_penalty=.0001,aggregation='mean_abs')
        variants = {
            'l1_004': dict(l1_penalty=.04),
            'l1_007': dict(l1_penalty=.07),
            'l1_015': dict(l1_penalty=.15),
            'l1_025': dict(l1_penalty=.25),
            'fusion_002': dict(fusion_penalty=.02),
            'fusion_007': dict(fusion_penalty=.07),
            'fusion_03': dict(fusion_penalty=.3),
            'fusion_06': dict(fusion_penalty=.6),
            'group_00005': dict(group_penalty=.0005),
            'group_0002': dict(group_penalty=.002),
            'group_0005': dict(group_penalty=.005),
            'group_002': dict(group_penalty=.02),
            'group_004': dict(group_penalty=.04),
            'coupled_a': dict(l1_penalty=.05,fusion_penalty=.05,group_penalty=.002),
            'coupled_b': dict(l1_penalty=.05,fusion_penalty=.3,group_penalty=.002),
            'coupled_c': dict(l1_penalty=.2,fusion_penalty=.05,group_penalty=.02),
            'coupled_d': dict(l1_penalty=.2,fusion_penalty=.3,group_penalty=.02),
            'coupled_e': dict(l1_penalty=.03,fusion_penalty=.3,group_penalty=.001),
            'coupled_f': dict(l1_penalty=.3,fusion_penalty=.03,group_penalty=.03),
            'ridge_00001': dict(ridge_penalty=.00001),
            'ridge_0001': dict(ridge_penalty=.001),
            'ridge_001': dict(ridge_penalty=.01),
        }
        return {name: dict(base,**change) for name,change in variants.items()}
