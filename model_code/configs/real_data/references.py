"""先验预处理参数。默认写入 reconstruct，不覆盖旧 references。"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from configs.paths import *
from configs.real_data.settings.prepare_references import ReferenceConfig
from configs.real_data.settings.prepare_cistarget import CisTargetConfig
from configs.real_data.settings.prepare_curated_lr import CuratedLRConfig
from configs.real_data.settings.prepare_lr_tf import LRTFConfig


@dataclass
class ReferenceParameters(ReferenceConfig):
    tf_target_file: Path = OLD_RAW / "ModelData/Mouse-TF-Target.tsv"
    ligand_receptor_file: Path = OLD_RAW / "ModelData/Mouse-ligand-receptor.tsv"
    output_dir: Path = LOCAL_PROCESSED / "references"


@dataclass
class MotifParameters(CisTargetConfig):
    ranking_file: Path = OLD_RAW / "ModelData/mm10_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather"
    motif_annotation_file: Path = OLD_RAW / "ModelData/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl.txt"
    reference_dir: Path = ReferenceParameters.output_dir


@dataclass
class LRParameters(CuratedLRConfig):
    candidate_lr_file: Path = OLD_RAW / "ModelData/Mouse-ligand-receptor.tsv"
    ligand_role_file: Path = OLD_RAW / "ModelData/omnipath_support/human_intercell_ligand.tsv"
    receptor_role_file: Path = OLD_RAW / "ModelData/omnipath_support/human_intercell_receptor.tsv"
    reference_dir: Path = ReferenceParameters.output_dir


@dataclass
class LRTFParameters(LRTFConfig):
    signaling_file: Path = OLD_RAW / "ModelData/omnipath_support/mouse_signaling_network.tsv"
    reference_dir: Path = ReferenceParameters.output_dir


@dataclass
class ReferenceWorkflow:
    steps: tuple[str, ...] = ("standardize", "motif", "lr", "lr_tf")
    standardize: ReferenceParameters = field(default_factory=ReferenceParameters)
    motif: MotifParameters = field(default_factory=MotifParameters)
    lr: LRParameters = field(default_factory=LRParameters)
    lr_tf: LRTFParameters = field(default_factory=LRTFParameters)
