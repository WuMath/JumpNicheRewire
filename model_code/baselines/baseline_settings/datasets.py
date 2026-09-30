"""只修改这个类即可选择已有 benchmark；不下载、不读取真值内容。"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DatasetConfig:
    # sergio / sccigen / scmultisim / visium / core_final / perturbation
    suite: str = "scmultisim"
    scmultisim_root: Path = Path("C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/reconstruct/data/processed/scmultisim_v1")
    scmultisim_frozen: Path = Path("C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/reconstruct/data/result/public_sim_data/scmultisim_v1/00_protocol/scmultisim_frozen.json")
    evidence_registry: Path = Path("C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/data/processed/evidence_benchmark/03_registries/evidence_registry.csv")
    formal_registry: Path = Path("C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/data/processed/formal_benchmark_core_final_v1/registries/final/dataset_registry.csv")
    perturbation_registry: Path = Path("C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/reconstruct/data/processed/evidence_benchmark/03_registries/evidence_registry.csv")
    # None 使用下方该 suite 的明确清单；补跑时写 ("某个dataset_id",)。
    dataset_ids: tuple[str, ...] | None = None
    sergio_ids: tuple[str, ...] = tuple(f"sergio_grn_100g_seed_{s}" for s in (20260902,20260903,20260904,20260905,20260906))
    sccigen_ids: tuple[str, ...] = ("sccigen_human_breast_example1",)
    visium_ids: tuple[str, ...] = ("real_pseudo__visium_ffpe_mouse_brain",)
    core_scenarios: tuple[str, ...] = ("clean_500","dropout_500","low_prior_500","boundary_500")
    # 正式模拟只选择一种 split；不能用 test 指标反过来调参。
    split: str = "pilot"
    unlock_test: bool = False
    # 旧扰动流程尚未通过盲法/重复验收，默认拒绝正式评价。
    perturbation_protocol_validated: bool = False
    output_root: Path = Path("C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/reconstruct/data/result/independent_comparison_v1")
    # CSV 内为旧 Windows 路径时，Linux/WSL/超算可明确设置两个路径前缀。
    path_prefix_from: str = ""
    path_prefix_to: str = ""
    # 文件数值代表的尺度。raw_counts 才执行 log1p(1e4 * count / library)。
    expression_scale: str = "raw_counts"  # raw_counts / log1p_normalized
    target_sum: float = 10000.0
    # 所有方法使用同一份无真值筛选的基因面板；默认不截断。
    gene_panel_file: Path | None = None  # 一列 gene，手动选定后冻结
    cell_type_column: str = "cell_type"

DATA = DatasetConfig()
