"""只读取已有方法结果；不会重新训练任何模型。"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class EvaluationConfig:
    methods: tuple[str, ...] = ("JumpNicheRewire","GENIE3","GRNBoost2") # ,"CeSpGRN"
    our_scmultisim_root: Path = Path("C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/reconstruct/data/result/public_sim_data/scmultisim_v1/02_runs/public_simulator")
    our_sergio_root: Path = Path("C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/reconstruct/data/result/public_sim_data/sergio_nonspatial_v2/02_runs/public_simulator")
    # Global evaluation NEVER joins on region/cell_id. Freeze this before scoring.
    # max = an edge exists somewhere; mean = average strength over supplied units.
    global_grn_aggregation: str = "max"
    our_sccigen_root: Path = Path("C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/reconstruct/data/result/public_sim_data/sccigen/02_runs/public_simulator")
    our_visium_root: Path = Path("C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/reconstruct/data/result/real_data/visium/02_runs/real_pseudotruth")
    # 这里只读复用旧代码已经运行的正式 test，不需要在重构版重跑。
    our_formal_root: Path = Path("C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/data/result/benchmark/core_final_v1/final/02_runs")
    our_perturbation_root: Path = Path("C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/reconstruct/data/result/perturbation/spatial_perturbseq/02_runs/perturbation")
    top_k: tuple[int, ...] = (10,50,100)
    # Precision / Recall / F1 三个无后缀指标：预先固定取分数最高的 k 条边。
    # 不同方法的 score 尺度不同，不能共用 0.5 阈值；不要在 test 上选择最佳 k。
    classification_top_k: int = 100
    # 始终计算 F1；可加 0.5（偏重 precision）或 2.0（偏重 recall）。
    f_betas: tuple[float, ...] = (1.0,)
    export_per_metric_tables: bool = True
    bootstrap_iterations: int = 2000
    confidence_level: float = 0.95
    random_seed: int = 20260906
    # 必须在查看这轮排名之前固定；只有这些主指标做跨方法显著性检验。
    primary_metrics: tuple[str, ...] = ("AP",)
    # 默认不覆盖评价结果；增添方法后手动改名 v2，再重新评价即可。
    analysis_name: str = "scmultisim_pilot_grn_v1"
    maximum_prediction_rows: int = 5000000
    maximum_truth_rows: int = 5000000
    # 真实伪真值的资源独立性没有验收时，只生成探索性检索结果。
    real_prior_independence_verified: bool = False
    require_our_convergence: bool = True

EVALUATION = EvaluationConfig()
