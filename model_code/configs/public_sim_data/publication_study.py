"""公开数据论文实验。所有路径和参数集中在类中；不接受命令行参数。"""
from dataclasses import dataclass
from pathlib import Path
from configs.common.experiment import ExperimentConfig, RunOptions
from configs.common.presets import ModelParameters

ROOT = Path(__file__).resolve().parents[2]

@dataclass
class StudyConfig:
    raw_root: Path = ROOT / "data/raw/publication_20260920"
    processed_root: Path = ROOT / "data/processed/publication_20260920"
    result_root: Path = ROOT / "data/result/publication_20260920"
    public_archive_root: Path = ROOT.parent / "data/raw/formal_benchmark/manual_downloads"
    old_data_root: Path = ROOT.parent / "data"
    rscript: Path = Path("D:/SoftWare/R/R-4.4.1/bin/Rscript.exe")
    model_python: Path = Path("C:/Anaconda/envs/jumpniche/python.exe")
    tree_python: Path = Path("C:/Anaconda/envs/jnr-trees-clean/python.exe")
    maximum_new_gib: float = 9.0  # 为日志、报告和临时文件留 1 GiB 余量。
    minimum_free_disk_gib: float = 5.0
    maximum_worker_rss_gib: float = 3.5
    minimum_available_ram_gib: float = 1.5
    trees: tuple[int, ...] = (1, 3, 5)
    noise_levels: tuple[str, ...] = ("0.1", "0.5")
    validation_replicates: tuple[int, ...] = (1,)
    test_replicates: tuple[int, ...] = (2, 3, 4)
    published_spatial: tuple[str, ...] = ("spatial_5000cells_500genes", "spatial_3000cells_2000genes")
    cases: tuple[str, ...] = ("mouse_brain", "mouse_kidney", "human_breast_cancer")
    methods: tuple[str, ...] = ("JumpNicheRewire", "GENIE3", "GRNBoost2")
    # 所有细胞/spot 保留；大面板仅按表达选择共同基因范围，不读真值。
    benchmark_maximum_targets: int = 200
    case_maximum_genes: int = 600
    case_maximum_tfs: int = 12
    case_maximum_targets: int = 150
    case_maximum_lr_pairs: int = 40
    seed: int = 20260920
    spatial_permutations: int = 999
    classification_top_k: tuple[int, ...] = (10, 50, 100)
    bootstrap_repetitions: int = 2000
    pathway_top_targets: int = 50
    minimum_pathway_size: int = 5
    maximum_pathway_size: int = 500
    pathway_fdr: float = 0.05
    # 已有参数预先冻结；本轮不依测试真值搜索参数，不承诺人为制造领先。
    l1_penalty: float = 0.02
    fusion_penalty: float = 0.15
    group_penalty: float = 0.002
    ridge_penalty: float = 0.0001

def build_config():
    settings = StudyConfig()
    return ExperimentConfig("publication_study", "public_sim_data", "publication", "publication",
        run=RunOptions(output_root=settings.result_root), model=ModelParameters(), preprocessing=settings)
