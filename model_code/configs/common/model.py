"""All user-editable parameters for JumpNicheRewire.

The project intentionally has no command-line arguments.  Edit the class fields
in this file, then execute the corresponding top-level Python script.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from grn2_project.paths import MODEL_RESULT_ROOT, PROCESSED_DATA_ROOT, PROJECT_ROOT


@dataclass
class DataConfig:
    """Input paths and data-validation rules (paper Sections 4 and 7)."""

    input_dir: Path = PROCESSED_DATA_ROOT / "pilot/visium_ffpe_mouse_brain"
    expression_file: str = "expression.csv"
    coordinates_file: str = "coordinates.csv"
    metadata_file: str = "metadata.csv"
    tf_file: str = "tf_list.csv"
    lr_file: str = "lr_pairs.csv"
    tf_gene_prior_file: str = "prior_tf_gene.csv"
    lr_tf_prior_file: str = "prior_lr_tf.csv"
    ligand_activity_file: str = "ligand_activity.csv"
    receptor_activity_file: str = "receptor_activity.csv"
    tf_activity_file: str = "tf_activity.csv"
    perturbation_file: str = "perturbations.csv"
    species: str = "mouse"
    # False for ordinary scRNA-seq/SERGIO: coordinates and regions are not evidence.
    is_spatial: bool = True
    expression_is_normalized: bool = False
    library_scale: float = 10_000.0
    minimum_detection_rate: float = 0.02
    minimum_total_count: float = 10.0
    sample_column: str = "sample_id"
    cell_type_column: str = "cell_type"
    region_column: str = "region"
    condition_column: str = "condition"
    confounder_columns: tuple[str, ...] = (
        "cell_type",
        "sample_id",
        "condition",
        "measured_confounder",
    )


@dataclass
class SpatialConfig:
    """Module 1 and Appendix A parameters."""

    screening_neighbors: int = 8
    graph_neighbors: int = 10
    long_edge_factor: float = 2.5
    bandwidth_lower_quantile: float = 0.05
    bandwidth_upper_quantile: float = 0.95
    pca_components: int = 20
    pca_variance_threshold: float = 0.80
    state_bandwidth: float = 1.0
    regulation_bandwidth: float = 1.0
    type_compatibility_same: float = 1.0
    type_compatibility_different: float = 0.70
    spatial_statistics: tuple[str, ...] = ("moran", "geary", "getis")
    spatial_permutations: int = 99
    # Bounds edge-by-gene temporary arrays during permutation testing.  This
    # changes only evaluation order, not the statistic or permutation count.
    spatial_statistic_chunk_genes: int = 32
    spatial_fdr: float = 0.10
    minimum_spatial_genes: int = 20
    maximum_spatial_genes: int = 500
    hvg_count: int = 2_000
    edge_weight_floor: float = 1e-8


@dataclass
class PriorConfig:
    """Module 3 double-layer soft-prior parameters."""

    intercept: float = -0.5
    motif_weight: float = 1.0
    atac_weight: float = 1.0
    grn_weight: float = 0.75
    expression_weight: float = 1.0
    prior_penalty_scale: float = 0.03
    lr_penalty_scale: float = 0.02
    epsilon: float = 0.05


@dataclass
class BaseGRNConfig:
    """Module 2 graph-fused sparse GRN parameters."""

    maximum_tfs: int = 30
    global_max_iterations: int = 5000
    global_tolerance: float = 1e-6
    global_adjust_covariates: bool = False
    maximum_targets: int = 200
    hard_prior_threshold: float | None = None
    l1_penalty: float = 0.03
    fusion_penalty: float = 0.08
    admm_rho: float = 1.0
    admm_max_iterations: int = 5000
    admm_absolute_tolerance: float = 1e-4
    admm_relative_tolerance: float = 1e-3
    admm_direct_max_variables: int = 2000
    admm_adaptive_rho: bool = True
    admm_rho_update_interval: int = 25
    admm_rho_adapt_until: int = 1000
    admm_residual_balance: float = 5.0
    admm_rho_min: float = 1e-4
    admm_rho_max: float = 1e4
    cg_max_iterations: int = 1000
    cg_tolerance: float = 1e-8
    coefficient_threshold: float = 1e-4


@dataclass
class CommunicationConfig:
    """Module 4 LR exposure parameters."""

    communication_neighbors: int = 12
    bandwidth_neighbors: int = 10
    default_range_factor: float = 1.0
    contact_neighbors: int = 4
    normalize_sender_kernel: bool = True
    saturation_quantile: float = 0.50
    complex_epsilon: float = 1e-6
    completeness_power_ligand: float = 1.0
    completeness_power_receptor: float = 1.0
    detection_threshold: float = 0.0
    allow_autocrine: bool = False
    minimum_contribution: float = 1e-10


@dataclass
class RewiringConfig:
    """Communication-conditioned GRN and engineering-scale parameterization."""

    parameterization: str = "region_shared"
    disable_communication_rewiring: bool = False
    joint_iterations: int = 120
    communication_mixing: float = 0.50
    adaptive_communication_mixing: bool = True
    maximum_communication_mixing: float = 0.995
    mixing_step_contraction: float = 0.50
    mixing_stall_ratio: float = 0.98
    mixing_oscillation_cosine: float = -0.50
    iteration_tolerance: float = 1e-4
    maximum_active_lr: int = 20
    maximum_interaction_tfs: int = 20
    group_penalty: float = 0.002
    ridge_penalty: float = 1e-4
    fista_max_iterations: int = 5000
    fista_tolerance: float = 1e-6
    coefficient_threshold: float = 1e-5
    # Output candidates are selected numerically in bounded chunks.  These
    # limits prevent construction of millions of Python dictionaries.
    top_grn_rows: int = 500_000
    top_lr_gene_rows: int = 50_000
    top_path_rows: int = 50_000
    output_build_chunk_rows: int = 250_000
    # Numerical tensors are assembled by cell batches; this is separate from
    # the result-table chunk size above.
    numerical_cell_chunk_size: int = 512


@dataclass
class CausalConfig:
    """Core observational diagnostics and optional evidence enhancement."""

    minimum_group_size: int = 8
    minimum_exposure_standard_deviation: float = 1e-4
    lower_exposure_quantile: float = 0.25
    upper_exposure_quantile: float = 0.75
    minimum_overlap_fraction: float = 0.70
    overlap_ridge_penalty: float = 1e-8
    require_adjustment_covariates_for_level1: bool = True
    activity_evidence_level: int = 2
    perturbation_evidence_level: int = 3


@dataclass
class OutputConfig:
    """Output locations and result-volume controls."""

    output_dir: Path = MODEL_RESULT_ROOT / "jump_niche_rewire"
    method_name: str = "JumpNicheRewire"
    workbook_name: str = "JumpNicheRewire_results.xlsx"
    save_parquet: bool = True
    parquet_dir_name: str = "parquet"
    parquet_compression: str = "zstd"
    parquet_row_group_size: int = 100_000
    save_intermediate_csv: bool = False
    maximum_grn_rows: int = 500_000
    maximum_spatial_edge_rows: int = 200_000
    maximum_excel_data_rows_per_sheet: int = 1_000_000
    # Excel is a human-readable preview.  Full large tables always remain in
    # Parquet, avoiding multi-million-row workbooks on real datasets/HPC.
    maximum_excel_rows_per_table: int = 500_000
    maximum_history_excel_rows: int = 1000  # Complete history is always in Parquet.


@dataclass
class RuntimeConfig:
    """Reproducibility and diagnostics."""

    random_seed: int = 20260822
    verbose: bool = True
    fail_on_nonfinite: bool = True
    require_convergence: bool = True
    minimum_target_convergence_rate: float = 0.90
    blas_threads: int = 1
    enable_numerical_checkpoint: bool = True
    resume_numerical_checkpoint: bool = True
    keep_numerical_checkpoint_after_success: bool = False
    numerical_checkpoint_name: str = "numerical_checkpoint.pkl"
    enable_target_checkpoints: bool = True
    resume_target_checkpoints: bool = True
    target_checkpoint_dir_name: str = "numerical_target_checkpoints"
    keep_target_checkpoints_after_success: bool = False
    # The evidence runner fills this with the prepared-dataset signature.
    checkpoint_data_signature: str = ""


@dataclass
class ProjectConfig:
    """Single configuration object used by the main pipeline."""

    data: DataConfig = field(default_factory=DataConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    base_grn: BaseGRNConfig = field(default_factory=BaseGRNConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    rewiring: RewiringConfig = field(default_factory=RewiringConfig)
    causal: CausalConfig = field(default_factory=CausalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


CONFIG = ProjectConfig()
