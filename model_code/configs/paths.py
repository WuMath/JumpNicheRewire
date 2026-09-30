"""只在此配置项目位置；旧项目仅作为只读数据来源。"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

RECONSTRUCT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PathsConfig:
    source_project: Path = Path("C:/Users/Wuyan/Desktop/GRN/Futher/GRN2")
    data_root: Path = RECONSTRUCT_ROOT / "data"
    vendor_root: Path = RECONSTRUCT_ROOT / "vendor"
    # 如果要输出到另一块磁盘，必须在这里显式增加允许写入的目录。
    allowed_write_roots: tuple[Path, ...] = (
        RECONSTRUCT_ROOT / "data", RECONSTRUCT_ROOT / "vendor",
    )


PATHS = PathsConfig()
SOURCE_PROJECT = PATHS.source_project
OLD_RAW = SOURCE_PROJECT / "data/raw"
OLD_PROCESSED = SOURCE_PROJECT / "data/processed"
OLD_RESULTS = SOURCE_PROJECT / "data/result"
LOCAL_RAW = PATHS.data_root / "raw"
LOCAL_PROCESSED = PATHS.data_root / "processed"
LOCAL_RESULTS = PATHS.data_root / "result"
