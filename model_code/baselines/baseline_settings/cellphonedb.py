"""手动修改此类；mode 顺序：check → freeze → run。数据位置见 datasets.py。"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class MethodConfig:
    method: str = "CellPhoneDB"
    mode: str = "check"
    seed: int = 20260906
    database: Path = Path("C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/reconstruct/data/raw/baseline_resources/cellphonedb.zip")
    iterations: int = 1000
    threads: int = 1
    threshold: float = 0.1
    result_precision: int = 8
    overwrite: bool = False
    preview_rows: int = 10000

CONFIG = MethodConfig()
