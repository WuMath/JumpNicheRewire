"""有限第二轮开发；路径/参数在类中，不接受脚本参数。旧test及案例未解锁。"""
from dataclasses import dataclass
from pathlib import Path
from configs.public_sim_data.decomposition_v2 import DecompositionStudyConfig
from configs.common.experiment import ExperimentConfig, RunOptions

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Round02Config(DecompositionStudyConfig):
    output_root: Path = ROOT/'data/result/optimization_20260921/round_02'
    old_root: Path = ROOT/'data/result/optimization_20260921/round_01'
    raw_root: Path = ROOT/'data/raw/optimization_20260921'
    model_python: Path = Path('C:/Anaconda/envs/jumpniche/python.exe')
    tree_python: Path = Path('C:/Anaconda/envs/jnr-trees-clean/python.exe')
    rscript: Path = Path('D:/SoftWare/R/R-4.4.1/bin/Rscript.exe')
    seed: int = 20260921
    classification_top_k: tuple[int, ...] = (10, 50, 100)
    methods: tuple[str, ...] = ('JumpNicheRewire', 'GENIE3', 'GRNBoost2', 'TIGRESS', 'CLR', 'PPCOR')
    execution_order: tuple[str, ...] = ('JumpNicheRewire', 'CLR', 'TIGRESS', 'GRNBoost2', 'GENIE3', 'PPCOR')
    # 只允许工程失败后显式恢复；nonconverged不自动重拟合。
    resume_engineering_failures: bool = False

    def candidates(self):
        return {
            'JumpNicheRewire': {
                'v2_anchor': {},
                'v2_less_sparse': {'l1_backbone': .002, 'l1_spatial': .004, 'l1_lr': .004, 'group_penalty': .002},
                'v2_more_sparse': {'l1_backbone': .05, 'l1_spatial': .1, 'l1_lr': .1, 'group_penalty': .05},
                'v2_stable_local': {'ridge_spatial': 1., 'ridge_lr': 1.},
            },
            'GENIE3': {f'leaf{leaf}_{fraction}': {'min_samples_leaf': leaf, 'max_features': value}
                       for leaf in (3, 5) for fraction, value in (('all', 1.), ('half', .5))},
            'GRNBoost2': {
                'half_lr005': {'max_features': .5, 'learning_rate': .005},
                'half_lr05': {'max_features': .5, 'learning_rate': .05},
                'half_depth1': {'max_features': .5, 'max_depth': 1},
                'half_depth5': {'max_features': .5, 'max_depth': 5},
            },
            'TIGRESS': {f'a{alpha}_steps{steps}_split200': {'alpha': alpha, 'nstepsLARS': steps, 'nsplit': 200}
                        for alpha in (.2, .4) for steps in (3, 5)},
            'CLR': {f'{estimator}_{bins}': {'estimator': estimator, 'disc': 'equalfreq', 'nbins': bins}
                    for estimator in ('mi.shrink', 'mi.mm') for bins in (10, 20)},
            'PPCOR': {},  # 三种官方相关类型已在round_01覆盖，核验并复用全部结果。
        }

    def ablations(self):
        # 固定anchor，不参与选参。不把表现好的消融偷偷变为第四/第五候选。
        return {'anchor_no_lr': {'use_lr': False},
                'anchor_no_spatial': {'use_spatial': False},
                'anchor_backbone_only': {'use_lr': False, 'use_spatial': False}}


def build_config():
    cfg = Round02Config()
    return ExperimentConfig('benchmark_round02', 'public_sim_data', 'development', 'benchmark_round02',
                            run=RunOptions(output_root=cfg.output_root), preprocessing=cfg)
