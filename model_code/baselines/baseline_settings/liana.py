"""手动修改此类；mode 顺序：check → freeze → run。数据位置见 datasets.py。"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class MethodConfig:
    method: str = "LIANA"
    mode: str = "check"
    seed: int = 20260906
    n_perms: int = 1000
    min_cells: int = 5
    expr_prop: float = 0.1
    resource_mode: str = "matched"
    resource_name: str = "consensus"
    resource_species: str = "human"  # 仅 native 模式；与所选内置资源真实物种一致
    overwrite: bool = False
    preview_rows: int = 10000

CONFIG = MethodConfig()
