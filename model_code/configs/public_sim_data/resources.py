"""手动下载目录检查，以及已下载官方源码的本地解压。"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from configs.paths import *
from configs.public_sim_data.settings import CatalogConfig, InstallConfig


@dataclass
class CatalogParameters(CatalogConfig):
    raw_root: Path = OLD_RAW / "evidence_benchmark/01_raw_downloads"
    manifest_root: Path = LOCAL_RESULTS / "downloads"
    download_enabled: bool = False


@dataclass
class InstallParameters(InstallConfig):
    external_root: Path = PATHS.vendor_root / "external_simulators"
    sergio_archive: Path = OLD_RAW / "evidence_benchmark/01_raw_downloads/public_simulator/SERGIO-master.zip"
    install_sccigen_with_r: bool = False
    install_spatialperturbseq_with_r: bool = False
