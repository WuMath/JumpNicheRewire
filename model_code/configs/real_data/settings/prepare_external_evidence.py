from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from grn2_project.paths import *

@dataclass
class ExternalEvidenceConfig:
    processed_input_dir: Path = PROJECT_ROOT / "data/processed/full/visium_ffpe_mouse_brain"
    external_dir: Path = PROJECT_ROOT / "data/raw/external_evidence"
    ligand_activity_source: Path = PROJECT_ROOT / "data/raw/external_evidence/ligand_activity.csv"
    receptor_activity_source: Path = PROJECT_ROOT / "data/raw/external_evidence/receptor_activity.csv"
    tf_activity_source: Path = PROJECT_ROOT / "data/raw/external_evidence/tf_activity.csv"
    perturbation_source: Path = PROJECT_ROOT / "data/raw/external_evidence/perturbations.csv"

    confirm_activity_is_independently_measured: bool = False
    confirm_perturbation_has_control_and_biological_replicates: bool = False
    require_all_expression_ids_in_activity: bool = True
    overwrite_existing_evidence: bool = False
    audit_name: str = "external_evidence_audit.csv"
    manifest_name: str = "external_evidence_manifest.json"


CONFIG = ExternalEvidenceConfig()

