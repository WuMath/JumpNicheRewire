"""手动修改此类；mode 顺序：check → freeze → run。数据位置见 datasets.py。"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class MethodConfig:
    method: str = "iMIRACLE-LRComponent"
    mode: str = "check"
    seed: int = 20260906
    repository: Path = Path("C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/reconstruct/vendor/external_methods/iMiracle")
    expected_commit: str = "6f8c554a746c23d9df227a20d97288f3bf49dee7"
    device: str = "cpu"
    hidden_dim: int = 32
    gnn_type: str = "Transformer"
    epochs: int = 1000
    patience: int = 50
    learning_rate: float = 0.01
    threshold: float = 0.001
    neighbors: int = 5
    expressed_prop: float = 0.15
    iterative: bool = True
    omega: float = 0.0
    spatial_loss_epoch: int = 1000
    memory_limit_gib: float = 8.0
    overwrite: bool = False
    preview_rows: int = 10000

CONFIG = MethodConfig()
