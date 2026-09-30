"""手动修改此类；mode 顺序：check → freeze → run。数据位置见 datasets.py。"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class MethodConfig:
    method: str = "SpaGRN"
    mode: str = "check"
    seed: int = 20260906
    n_jobs: int = 4
    repository: Path = Path("C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/reconstruct/vendor/external_methods/SpaGRN")
    expected_commit: str = "61690ca69a766f14c580de5dd47d4bc7b603813a"
    database: Path = Path("C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/data/raw/ModelData/mm10_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather")
    motif_annotations: Path = Path("C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/data/raw/ModelData/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl.txt")
    reference_species: str = "mouse"
    neighbors: int = 30
    hotspot_model: str = "danb"
    overwrite: bool = False
    preview_rows: int = 10000

CONFIG = MethodConfig()
