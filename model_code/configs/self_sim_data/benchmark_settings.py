"""All user-editable formal benchmark parameters.

There are no command-line parameters.  Edit the dataclass fields in this file
and execute the corresponding top-level ``run_benchmark_*.py`` file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from grn2_project.paths import BENCHMARK_RESULT_ROOT, PROCESSED_DATA_ROOT, PROJECT_ROOT, RAW_DATA_ROOT
from configs.common.model import ProjectConfig


BENCHMARK_ROOT = BENCHMARK_RESULT_ROOT / "convergence_v2"
DATA_ROOT = PROCESSED_DATA_ROOT / "formal_benchmark"
RAW_ROOT = RAW_DATA_ROOT / "formal_benchmark"
BENCHMARK_PROFILES = ("pilot", "final")


def validate_benchmark_profile(profile: str) -> str:
    profile = str(profile).strip().lower()
    if profile not in BENCHMARK_PROFILES:
        raise ValueError(f"benchmark profile must be one of {BENCHMARK_PROFILES}, got {profile!r}")
    return profile


def profile_registry_path(data_root: Path, profile: str) -> Path:
    return Path(data_root) / "registries" / validate_benchmark_profile(profile) / "dataset_registry.csv"


def profile_benchmark_root(benchmark_root: Path, profile: str) -> Path:
    return Path(benchmark_root) / validate_benchmark_profile(profile)


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    region_count: int
    cells_per_region: int
    tf_count: int
    target_count: int
    lr_count: int
    noise_gene_count: int
    base_edge_density: float
    rewiring_edge_density: float
    dropout_rate: float = 0.0
    prior_coverage: float = 1.0
    coordinate_spread: float = 0.65
    region_separation: float = 5.0
    count_dispersion: float = 12.0
    causal_confounding: str = "none"
    confounder_strength: float = 0.60
    include_activity_evidence: bool = False
    include_perturbation_evidence: bool = False
    truth_effect_threshold: float = 0.08


def pilot_scenarios() -> tuple[ScenarioSpec, ...]:
    """Small scenarios for validating the benchmark machinery, not publication."""

    common = dict(
        region_count=3,
        cells_per_region=12,
        tf_count=8,
        target_count=16,
        lr_count=4,
        noise_gene_count=10,
        base_edge_density=0.015,
        rewiring_edge_density=0.008,
    )
    return (
        ScenarioSpec("pilot_clean", **common),
        ScenarioSpec("pilot_dropout", **common, dropout_rate=0.35),
        ScenarioSpec(
            "pilot_boundary",
            **common,
            coordinate_spread=1.10,
            region_separation=3.0,
        ),
    )


def final_scenarios() -> tuple[ScenarioSpec, ...]:
    """Publication-oriented stress tests; edit sizes to match available RAM."""

    common = dict(
        region_count=4,
        cells_per_region=125,
        tf_count=20,
        target_count=60,
        lr_count=8,
        noise_gene_count=80,
        base_edge_density=0.010,
        rewiring_edge_density=0.003,
    )
    return (
        ScenarioSpec("clean_500", **common),
        ScenarioSpec("dropout_500", **common, dropout_rate=0.40),
        ScenarioSpec("low_prior_500", **common, prior_coverage=0.40),
        ScenarioSpec(
            "boundary_500",
            **common,
            coordinate_spread=1.20,
            region_separation=3.2,
        ),
        ScenarioSpec(
            "measured_confounding_500",
            **common,
            causal_confounding="measured",
            confounder_strength=0.90,
        ),
        ScenarioSpec(
            "unmeasured_confounding_500",
            **common,
            causal_confounding="unmeasured",
            confounder_strength=0.90,
        ),
        ScenarioSpec(
            "clean_1000",
            **{**common, "cells_per_region": 250},
        ),
        ScenarioSpec(
            "clean_1500",
            **{**common, "cells_per_region": 375},
        ),
    )


@dataclass(frozen=True)
class SplitSpec:
    # Pilot defaults.  For the final paper use at least 5/3/5 disjoint seeds.
    train_seeds: tuple[int, ...] = (1101, 1102)
    validation_seeds: tuple[int, ...] = (2101, 2102)
    test_seeds: tuple[int, ...] = (3101, 3102)


FINAL_SPLIT_SPEC = SplitSpec(
    train_seeds=(1101, 1102, 1103, 1104, 1105),
    validation_seeds=(2101, 2102, 2103),
    test_seeds=(3101, 3102, 3103, 3104, 3105),
)


@dataclass
class DataStageConfig:
    benchmark_root: Path = BENCHMARK_ROOT
    data_root: Path = DATA_ROOT
    # None keeps custom/test configurations self-contained under data_root.
    # The project defaults below explicitly use the canonical raw-data path.
    manual_download_root: Path | None = None
    profile: str = "pilot"  # pilot or final
    split: SplitSpec = field(default_factory=SplitSpec)
    overwrite: bool = False
    minimum_truth_prevalence: float = 0.005
    maximum_truth_prevalence: float = 0.20
    write_h5ad_when_available: bool = True
    zenodo_record_id: int = 13119261
    # The code never downloads these files.  It creates direct links and checks
    # files that the user manually puts in data/formal_benchmark/manual_downloads.
    manual_download_files: tuple[str, ...] = (
        "main_tree1.zip",
        "main_tree3.zip",
        "main_tree5.zip",
        "spatial_3000cells_2000genes.zip",
        "spatial_5000cells_500genes.zip",
    )
    include_reference_output_archive: bool = False
    require_all_selected_manual_downloads: bool = False

    @property
    def registry_path(self) -> Path:
        return profile_registry_path(self.data_root, self.profile)

    @property
    def profile_benchmark_root(self) -> Path:
        return profile_benchmark_root(self.benchmark_root, self.profile)

    def scenarios(self) -> tuple[ScenarioSpec, ...]:
        if self.profile == "pilot":
            return pilot_scenarios()
        if self.profile == "final":
            return final_scenarios()
        raise ValueError("profile must be 'pilot' or 'final'")


@dataclass(frozen=True)
class ParameterSet:
    parameter_id: str
    l1_penalty: float
    fusion_penalty: float
    group_penalty: float
    ridge_penalty: float
    admm_rho: float = 1.0
    spatial_permutations: int = 99
    admm_max_iterations: int = 5000
    fista_max_iterations: int = 5000
    joint_iterations: int = 120
    iteration_tolerance: float = 1e-4


def pilot_parameter_grid() -> tuple[ParameterSet, ...]:
    return (
        ParameterSet("balanced", 0.03, 0.08, 0.002, 1e-3),
        ParameterSet("sparser", 0.06, 0.10, 0.004, 3e-4),
    )


def final_parameter_grid() -> tuple[ParameterSet, ...]:
    return (
        ParameterSet("balanced", 0.03, 0.08, 0.002, 1e-3, spatial_permutations=999, admm_max_iterations=5000, fista_max_iterations=5000, joint_iterations=120),
        ParameterSet("sparser", 0.06, 0.10, 0.004, 3e-4, spatial_permutations=999, admm_max_iterations=5000, fista_max_iterations=5000, joint_iterations=120),
        ParameterSet("smooth", 0.02, 0.15, 0.002, 1e-4, spatial_permutations=999, admm_max_iterations=5000, fista_max_iterations=5000, joint_iterations=120),
    )


@dataclass
class ModelStageConfig:
    project_model: ProjectConfig | None = None
    benchmark_root: Path = BENCHMARK_ROOT
    data_root: Path = DATA_ROOT
    profile: str = "pilot"
    phase: str = "tune"  # tune, freeze, or test
    unlock_test: bool = False
    overwrite: bool = False
    minimum_convergence_rate: float = 0.90
    maximum_tfs: int = 30
    maximum_targets: int = 200
    maximum_active_lr: int = 20
    maximum_interaction_tfs: int = 20
    save_parquet: bool = True
    verbose: bool = True

    @property
    def registry_path(self) -> Path:
        return profile_registry_path(self.data_root, self.profile)

    @property
    def profile_benchmark_root(self) -> Path:
        return profile_benchmark_root(self.benchmark_root, self.profile)

    def parameter_grid(self) -> tuple[ParameterSet, ...]:
        return pilot_parameter_grid() if self.profile == "pilot" else final_parameter_grid()


@dataclass
class BaselineBatchConfig:
    benchmark_root: Path = BENCHMARK_ROOT
    data_root: Path = DATA_ROOT
    profile: str = "pilot"
    method: str = "CeSpGRN"
    phase: str = "validation"  # train, validation, freeze, or test
    unlock_test: bool = False
    overwrite: bool = False
    launch_official: bool = False
    official_repository_dir: Path = PROJECT_ROOT / "vendor" / "external_methods" / "CeSpGRN"
    # Tokens are expanded per dataset: {input_dir}, {raw_dir}, {dataset_id}.
    # Keep this empty until the official repository and its fixed wrapper exist.
    official_command_template: tuple[str, ...] = ()
    raw_grn_name: str = "grn_edges.csv"
    raw_lr_gene_name: str = "lr_gene.csv"
    raw_receptor_paths_name: str = "receptor_paths.csv"
    raw_cci_name: str = "cci_edges.csv"
    raw_activity_name: str = "regulon_activity.csv"
    raw_causal_name: str = "causal_edges.csv"

    @property
    def registry_path(self) -> Path:
        return profile_registry_path(self.data_root, self.profile)

    @property
    def profile_benchmark_root(self) -> Path:
        return profile_benchmark_root(self.benchmark_root, self.profile)


@dataclass
class AnalysisStageConfig:
    benchmark_root: Path = BENCHMARK_ROOT
    data_root: Path = DATA_ROOT
    profile: str = "pilot"
    overwrite: bool = False
    bootstrap_iterations: int = 2000
    confidence_level: float = 0.95
    random_seed: int = 20260829
    minimum_pairs_for_test: int = 3
    minimum_convergence_rate: float = 0.90
    primary_method: str = "JumpNicheRewire"
    expected_methods: tuple[str, ...] = (
        "JumpNicheRewire",
        "CeSpGRN",
        "SpaGRN",
        "iMIRACLE",
        "CLARIFY",
    )
    primary_metric_by_task: dict[str, str] = field(
        default_factory=lambda: {
            "GRN recovery": "Average Precision",
            "LR-gene attribution": "NDCG",
            "Receptor capture": "Receptor capture ratio",
            "CCI reconstruction": "Average Precision",
            "CCC-to-GRN causal reconstruction": "Spearman",
        }
    )

    @property
    def registry_path(self) -> Path:
        return profile_registry_path(self.data_root, self.profile)

    @property
    def profile_benchmark_root(self) -> Path:
        return profile_benchmark_root(self.benchmark_root, self.profile)


DATA_CONFIG = DataStageConfig(manual_download_root=RAW_ROOT / "manual_downloads")
FINAL_DATA_CONFIG = DataStageConfig(
    profile="final",
    split=FINAL_SPLIT_SPEC,
    manual_download_root=RAW_ROOT / "manual_downloads",
)
MODEL_CONFIG = ModelStageConfig()
ANALYSIS_CONFIG = AnalysisStageConfig()
