"""手动修改此类；mode 顺序：check → freeze → run。数据位置见 datasets.py。"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class MethodConfig:
    method: str = "GRNBoost2"
    mode: str = "run"
    seed: int = 20260906
    n_jobs: int = 4
    overwrite: bool = False
    preview_rows: int = 10000

CONFIG = MethodConfig()
