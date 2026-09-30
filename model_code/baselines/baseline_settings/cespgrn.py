"""手动修改此类；mode 顺序：check → freeze → run。数据位置见 datasets.py。"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class MethodConfig:
    method: str = "CeSpGRN"
    mode: str = "check"
    seed: int = 20260906
    repository: Path = Path("C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/reconstruct/vendor/external_methods/CeSpGRN")
    expected_commit: str = "2fb222f8a26edf1bdae14a95b1491543f0aaa4e8"
    kernel_input: str = "expression_pca"
    pca_components: int = 10
    bandwidth: float = 1.0
    neighbors: int = 30
    kernel_k: int = 5
    lamb: float = 0.1
    max_iters: int = 1000
    batch_size: int = 32
    device: str = "cpu"
    memory_limit_gib: float = 8.0
    overwrite: bool = False
    preview_rows: int = 10000

CONFIG = MethodConfig()
