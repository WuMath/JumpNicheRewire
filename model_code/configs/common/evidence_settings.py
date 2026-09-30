from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from grn2_project.paths import *
from configs.common.evidence_paths import *
from configs.common.model import ProjectConfig

@dataclass
class EvidenceModelConfig:
    project_model: ProjectConfig | None = None
    implementation_signature: str = ""
    registry_path: Path = DATA_ROOT / "03_registries" / "evidence_registry.csv"
    output_root: Path = WORKSPACE_ROOT / "02_runs"
    evidence_tracks: tuple[str, ...] = ("public_simulator", "real_pseudotruth", "perturbation")
    # Only these datasets run.  Use None explicitly to run every registered
    # dataset; keeping a tuple makes accidental all-data reruns impossible.
    dataset_ids: tuple[str, ...] | None = SCCIGEN_DATASET_IDS
    isolate_datasets: bool = True
    reconcile_stale_running_statuses: bool = True
    overwrite: bool = False
    require_convergence: bool = True
    minimum_convergence_rate: float = 0.90
    maximum_tfs: int = 30
    maximum_targets: int = 200
    maximum_active_lr: int = 20
    spatial_permutations: int = 999
    l1_penalty: float = 0.06
    fusion_penalty: float = 0.10
    group_penalty: float = 0.004
    ridge_penalty: float = 3e-4
    admm_max_iterations: int = 5000
    fista_max_iterations: int = 5000
    joint_iterations: int = 120
    iteration_tolerance: float = 1e-4


@dataclass
class EvidenceEvaluationConfig:
    registry_path: Path = DATA_ROOT / "03_registries" / "evidence_registry.csv"
    run_root: Path = WORKSPACE_ROOT / "02_runs"
    output_root: Path = WORKSPACE_ROOT / "03_evaluation"
    method_name: str = "JumpNicheRewire"
    external_result_root: Path = WORKSPACE_ROOT / "02_runs" / "baselines"
    overwrite: bool = False
    bootstrap_iterations: int = 2000
    confidence_level: float = 0.95
    random_seed: int = 20260902


@dataclass(frozen=True)
class ExternalResultSpec:
    method: str
    dataset_id: str
    grn_file: Path | None = None
    cci_file: Path | None = None
    tf_column: str = "tf"
    target_column: str = "target"
    score_column: str = "score"
    signed_score_column: str = "signed_score"
    sender_column: str = "sender_id"
    receiver_column: str = "receiver_id"
    ligand_column: str = "ligand"
    receptor_column: str = "receptor"
    cci_unit_type: str = "cell"  # "cell" or "cell_type"


@dataclass
class ExternalImportConfig:
    """Add one entry per independently executed method result."""

    specifications: tuple[ExternalResultSpec, ...] = ()
    output_root: Path = WORKSPACE_ROOT / "02_runs" / "baselines"
    overwrite: bool = False


@dataclass
class MethodTaskSpec:
    method: str
    tasks: tuple[str, ...]


@dataclass
class EvidenceJobConfig:
    registry_path: Path = DATA_ROOT / "03_registries" / "evidence_registry.csv"
    output_root: Path = WORKSPACE_ROOT / "01_jobs"
    methods: tuple[MethodTaskSpec, ...] = (
        MethodTaskSpec("CeSpGRN", ("grn_edges",)),
        MethodTaskSpec("SpaGRN", ("grn_edges",)),
        MethodTaskSpec("iMIRACLE", ("grn_edges",)),
        MethodTaskSpec("CLARIFY", ("grn_edges",)),
        MethodTaskSpec("CellChat", ("cci_edges",)),
        MethodTaskSpec("CellPhoneDB", ("cci_edges",)),
        MethodTaskSpec("LIANA", ("cci_edges",)),
    )

