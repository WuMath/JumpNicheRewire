from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from grn2_project.paths import *

@dataclass
class CuratedLRConfig:
    candidate_lr_file: Path = PROJECT_ROOT / "data/raw/ModelData/Mouse-ligand-receptor.tsv"
    ligand_role_file: Path = (
        PROJECT_ROOT / "data/raw/ModelData/omnipath_support/human_intercell_ligand.tsv"
    )
    receptor_role_file: Path = (
        PROJECT_ROOT / "data/raw/ModelData/omnipath_support/human_intercell_receptor.tsv"
    )
    reference_dir: Path = PROJECT_ROOT / "data/processed/references"

    expert_resources: tuple[str, ...] = (
        "Guide2Pharma",
        "HPMR",
        "ICELLNET",
        "Kirouac2010",
        "CellTalkDB",
        "CellChatDB",
        "connectomeDB2020",
        "ConnectomeDB2025",
        "CellPhoneDB",
        "Cellinker",
        "talklr",
        "SignaLink",
    )
    minimum_curation_effort: int = 1
    minimum_role_consensus: int = 1
    require_literature_reference: bool = True
    require_directed: bool = True
    require_ligand_secreted_or_surface: bool = True
    require_receptor_transmembrane: bool = True
    exclude_unmapped_uniprot_accessions: bool = True
    secreted_rho: float = 1.5
    contact_rho: float = 0.5

    model_output_name: str = "lr_pairs.csv"
    detailed_output_name: str = "lr_pairs_detailed.csv"
    audit_output_name: str = "lr_filter_audit.csv"
    rejected_output_name: str = "lr_rejected_examples.csv"
    maximum_rejected_examples: int = 5000
    backup_name: str = "lr_pairs_before_strict_curation.csv"
    manifest_name: str = "curated_lr_manifest.json"


CONFIG = CuratedLRConfig()

