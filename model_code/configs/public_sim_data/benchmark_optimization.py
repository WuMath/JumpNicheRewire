"""第二轮：只在开发集寻优。改这里，不传命令行参数；旧论文结果不变。"""
from dataclasses import dataclass
from pathlib import Path
from configs.public_sim_data.publication_study import StudyConfig
from configs.common.experiment import ExperimentConfig, RunOptions

ROOT = Path(__file__).resolve().parents[2]

@dataclass
class OptimizationConfig(StudyConfig):
    raw_root: Path = ROOT / 'data/raw/optimization_20260921'
    processed_root: Path = ROOT / 'data/processed/optimization_20260921'
    result_root: Path = ROOT / 'data/result/optimization_20260921/round_01'
    source_registry: Path = ROOT / 'data/processed/publication_20260920/benchmark_registry.csv'
    previous_result_root: Path = ROOT / 'data/result/publication_20260920'
    # 只使用原 validation 的6份。旧test已揭盲，不假装它们是新的独立test。
    development_split: str = 'validation'
    methods: tuple[str, ...] = ('JumpNicheRewire', 'GENIE3', 'GRNBoost2', 'TIGRESS', 'CLR', 'PPCOR')
    cases: tuple[str, ...] = ()
    allow_cases: bool = False
    confirmation_registry: Path | None = None  # 尚未确定未接触的新公开确认集，test硬锁定。
    maximum_task_hours: float = 4.0
    seed: int = 20260921
    # 包括上一轮新增数据、R库、所有寻优轮次；不是每轮另外允许9GiB。
    budget_roots: tuple[Path, ...] = (
        ROOT/'data/raw/publication_20260920', ROOT/'data/processed/publication_20260920',
        ROOT/'data/result/publication_20260920', ROOT/'data/raw/optimization_20260921',
        ROOT/'data/processed/optimization_20260921', ROOT/'data/result/optimization_20260921')
    # 预先声明的有限网格。候选数不是方法数；每个方法最终只选一个配置。
    def candidates(self):
        return {
            'JumpNicheRewire': {
                'default_max': {'l1_penalty': .02, 'fusion_penalty': .15, 'group_penalty': .002, 'aggregation': 'max_abs'},
                'default_mean': {'l1_penalty': .02, 'fusion_penalty': .15, 'group_penalty': .002, 'aggregation': 'mean_abs'},
                'less_sparse_mean': {'l1_penalty': .002, 'fusion_penalty': .05, 'group_penalty': .0002, 'aggregation': 'mean_abs'},
                'more_sparse_mean': {'l1_penalty': .1, 'fusion_penalty': .15, 'group_penalty': .01, 'aggregation': 'mean_abs'},
            },
            'GENIE3': {'official': {}, 'all_features': {'max_features': 1.0},
                       'leaf3': {'min_samples_leaf': 3}, 'half_features': {'max_features': .5}},
            'GRNBoost2': {'official': {}, 'lr005': {'learning_rate': .005},
                         'lr05': {'learning_rate': .05}, 'half_features': {'max_features': .5}},
            'TIGRESS': {'official': {'alpha': .2, 'nstepsLARS': 5, 'nsplit': 100},
                        'alpha04': {'alpha': .4, 'nstepsLARS': 5, 'nsplit': 100},
                        'steps3': {'alpha': .2, 'nstepsLARS': 3, 'nsplit': 100},
                        'steps8': {'alpha': .2, 'nstepsLARS': 8, 'nsplit': 100}},
            'CLR': {'official': {'estimator': 'spearman', 'disc': 'none', 'nbins': 10},
                    'pearson': {'estimator': 'pearson', 'disc': 'none', 'nbins': 10},
                    'empirical10': {'estimator': 'mi.empirical', 'disc': 'equalfreq', 'nbins': 10},
                    'empirical20': {'estimator': 'mi.empirical', 'disc': 'equalfreq', 'nbins': 20}},
            'PPCOR': {'pearson': {'method': 'pearson'}, 'spearman': {'method': 'spearman'},
                      'kendall': {'method': 'kendall'}},
        }

def build_config():
    cfg = OptimizationConfig()
    return ExperimentConfig('benchmark_optimization', 'public_sim_data', 'optimization',
                            'optimization', run=RunOptions(output_root=cfg.result_root), preprocessing=cfg)
