from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from grn2_project.paths import *

@dataclass
class LRTFConfig:
    signaling_file: Path = (
        PROJECT_ROOT / "data/raw/ModelData/omnipath_support/mouse_signaling_network.tsv"
    )
    reference_dir: Path = PROJECT_ROOT / "data/processed/references"
    lr_file_name: str = "lr_pairs.csv"
    tf_file_name: str = "tf_list.csv"

    minimum_curation_effort: int = 1
    require_directed: bool = True
    maximum_path_length: int = 4
    path_length_penalty: float = 0.25
    maximum_frontier_nodes: int = 5000
    maximum_tfs_per_receptor: int = 100
    maximum_tfs_per_lr: int = 50
    minimum_path_score: float = 0.15
    minimum_lr_tf_score: float = 0.10

    output_name: str = "prior_lr_tf.csv"
    path_output_name: str = "receptor_to_tf_paths.parquet"
    audit_output_name: str = "lr_tf_audit.csv"
    manifest_name: str = "lr_tf_manifest.json"


CONFIG = LRTFConfig()

