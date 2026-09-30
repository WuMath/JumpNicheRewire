"""手动修改此类；mode 顺序：check → freeze → run。数据位置见 datasets.py。"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class MethodConfig:
    method: str = "CLARIFY"
    mode: str = "check"
    seed: int = 20260906
    repository: Path = Path("C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/reconstruct/vendor/external_methods/clarify")
    expected_commit: str = "de5dccfdfc9c22d95c29d2a3e5ffdb05adc5de03"
    epochs: int = 120
    learning_rate: float = 0.01
    hidden_dim: int = 64
    internal_edge_split: float = 0.1
    neighbors: int = 5
    genes_per_cell: int = 45
    lr_database: int = 0
    initial_grn_bandwidth: float = 0.1
    initial_grn_neighbors: int = 30
    initial_grn_lamb: float = 0.1
    initial_grn_iterations: int = 1000
    memory_limit_gib: float = 8.0
    overwrite: bool = False
    preview_rows: int = 10000

CONFIG = MethodConfig()
