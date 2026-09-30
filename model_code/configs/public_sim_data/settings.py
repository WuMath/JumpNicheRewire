from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from grn2_project.paths import *
from configs.common.evidence_paths import *

@dataclass(frozen=True)
class ResourceSpec:
    resource_id: str
    evidence_track: str
    filename: str
    url: str
    source_page: str
    required: bool
    expected_bytes: int | None
    md5: str = ""
    purpose: str = ""


def public_resources() -> tuple[ResourceSpec, ...]:
    """Official resources; every downloaded file receives a local SHA-256 fingerprint."""

    return (
        ResourceSpec(
            "sergio_source", "public_simulator", "SERGIO-master.zip",
            "https://github.com/PayamDiba/SERGIO/archive/refs/heads/master.zip",
            "https://github.com/PayamDiba/SERGIO", True, None,
            purpose="Official SERGIO v1 source, demo inputs, and GRN truth.",
        ),
        ResourceSpec(
            "sccigen_source", "public_simulator", "sCCIgen-main.zip",
            "https://github.com/songxiaoyu/sCCIgen/archive/refs/heads/main.zip",
            "https://github.com/songxiaoyu/sCCIgen", False, None,
            purpose="Official sCCIgen R package; needed only when generating new simulations.",
        ),
        ResourceSpec(
            "sccigen_example1_count", "public_simulator", "example1_count.tsv",
            "https://raw.githubusercontent.com/songxiaoyu/sCCIgen_data/5c6dd387a1f58fc75c29bb6fc82ecea54888a49f/example_data/example1_count.tsv",
            "https://github.com/songxiaoyu/sCCIgen_data/tree/5c6dd387a1f58fc75c29bb6fc82ecea54888a49f/example_data",
            True, 38_676_256,
            purpose="Commit-pinned, internally matched sCCIgen example1 expression matrix.",
        ),
        ResourceSpec(
            "sccigen_example1_meta", "public_simulator", "example1_meta.tsv",
            "https://raw.githubusercontent.com/songxiaoyu/sCCIgen_data/5c6dd387a1f58fc75c29bb6fc82ecea54888a49f/example_data/example1_meta.tsv",
            "https://github.com/songxiaoyu/sCCIgen_data/tree/5c6dd387a1f58fc75c29bb6fc82ecea54888a49f/example_data",
            True, 140_268,
            purpose="Commit-pinned sCCIgen example1 cell coordinates and annotations.",
        ),
        ResourceSpec(
            "sccigen_example1_truth", "public_simulator", "example1_expr_pattern.tsv",
            "https://raw.githubusercontent.com/songxiaoyu/sCCIgen_data/5c6dd387a1f58fc75c29bb6fc82ecea54888a49f/example_data/example1_expr_pattern.tsv",
            "https://github.com/songxiaoyu/sCCIgen_data/tree/5c6dd387a1f58fc75c29bb6fc82ecea54888a49f/example_data",
            True, 49_557,
            purpose="Commit-pinned sCCIgen example1 implanted distance and cross-cell expression truth.",
        ),
        ResourceSpec(
            "sccigen_example1_parameters", "public_simulator", "example1_parameter.yml",
            "https://raw.githubusercontent.com/songxiaoyu/sCCIgen_data/5c6dd387a1f58fc75c29bb6fc82ecea54888a49f/example_data/example1_parameter.yml",
            "https://github.com/songxiaoyu/sCCIgen_data/tree/5c6dd387a1f58fc75c29bb6fc82ecea54888a49f/example_data",
            False, 2_456,
            purpose="Commit-pinned sCCIgen example1 parameter/provenance record.",
        ),
        ResourceSpec(
            "human_tf_target", "public_simulator", "Human-TF-Target.tsv",
            "https://omnipathdb.org/interactions/?format=tsv&types=transcriptional&organisms=9606&genesymbols=1&fields=sources,references,curation_effort",
            "https://r.omnipathdb.org/reference/omnipath-interactions.html",
            True, None,
            purpose="Human OmniPath transcriptional interactions used only to define expressed TF candidates.",
        ),
        ResourceSpec(
            "human_ligand_receptor", "public_simulator", "Human-ligand-receptor.tsv",
            "https://omnipathdb.org/interactions/?format=tsv&datasets=ligrecextra&organisms=9606&genesymbols=1&fields=sources,references,curation_effort",
            "https://r.omnipathdb.org/reference/omnipath-interactions.html",
            True, None,
            purpose="Human OmniPath ligand-receptor candidates used as non-implanted candidate pairs.",
        ),
        ResourceSpec(
            "sccigen_example2_count", "public_simulator", "example2_count.tsv",
            "https://raw.githubusercontent.com/songxiaoyu/sCCIgen_data/main/example_data/example2_count.tsv",
            "https://github.com/songxiaoyu/sCCIgen_data/tree/main/example_data", True, None,
            purpose="Quarantined: current main-branch example2 count is from an older simulation than its metadata.",
        ),
        ResourceSpec(
            "sccigen_example2_meta", "public_simulator", "example2_meta.tsv",
            "https://raw.githubusercontent.com/songxiaoyu/sCCIgen_data/main/example_data/example2_meta.tsv",
            "https://github.com/songxiaoyu/sCCIgen_data/tree/main/example_data", True, None,
            purpose="Quarantined: current main-branch example2 metadata does not match its count matrix.",
        ),
        ResourceSpec(
            "sccigen_example2_truth", "public_simulator", "example2_expr_pattern.tsv",
            "https://raw.githubusercontent.com/songxiaoyu/sCCIgen_data/main/example_data/example2_expr_pattern.tsv",
            "https://github.com/songxiaoyu/sCCIgen_data/tree/main/example_data", True, None,
            purpose="Quarantined: current main-branch example2 truth does not match its count matrix.",
        ),
        ResourceSpec(
            "sccigen_example2_parameters", "public_simulator", "example2_parameter.yml",
            "https://raw.githubusercontent.com/songxiaoyu/sCCIgen_data/main/example_data/example2_parameter.yml",
            "https://github.com/songxiaoyu/sCCIgen_data/tree/main/example_data", False, None,
            purpose="Quarantined: current main-branch example2 parameters do not match its count matrix.",
        ),
        ResourceSpec(
            "visium_counts", "real_pseudotruth", "Visium_FFPE_Mouse_Brain_filtered_feature_bc_matrix.h5",
            "https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_FFPE_Mouse_Brain/Visium_FFPE_Mouse_Brain_filtered_feature_bc_matrix.h5",
            "https://www.10xgenomics.com/cn/analysis-guides/integrating-10x-visium-and-chromium-data-with-r",
            False, None, purpose="Official 10x FFPE mouse-brain count matrix; already present in this project.",
        ),
        ResourceSpec(
            "visium_spatial", "real_pseudotruth", "Visium_FFPE_Mouse_Brain_spatial.tar.gz",
            "https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_FFPE_Mouse_Brain/Visium_FFPE_Mouse_Brain_spatial.tar.gz",
            "https://www.10xgenomics.com/cn/analysis-guides/integrating-10x-visium-and-chromium-data-with-r",
            False, None, purpose="Official spot coordinates and images; already present in this project.",
        ),
        ResourceSpec(
            "spatialperturbseq_source", "perturbation", "spatialperturbseq-master.zip",
            "https://github.com/kimberle9/spatialperturbseq/archive/refs/heads/master.zip",
            "https://github.com/kimberle9/spatialperturbseq", False, None,
            purpose="Official analysis package and vignette for Spatial Perturb-Seq.",
        ),
        ResourceSpec(
            "spatial_perturbseq_stereo", "perturbation", "GSE274447_RAW.tar",
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE274nnn/GSE274447/suppl/GSE274447_RAW.tar",
            "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE274447",
            False, None,
            purpose="Mouse-brain Stereo-seq with real coordinates and CRISPR barcodes (~2 GB).",
        ),
        ResourceSpec(
            "spatial_perturbseq_scrna", "perturbation", "GSE274058_RAW.tar",
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE274nnn/GSE274058/suppl/GSE274058_RAW.tar",
            "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE274058",
            False, None,
            purpose="Matched dissociated in-vivo Perturb-seq; optional orthogonal check.",
        ),
        ResourceSpec(
            "replogle_k562_bulk", "perturbation", "K562_essential_raw_bulk_01.h5ad",
            "https://ndownloader.figshare.com/files/35773070",
            "https://plus.figshare.com/articles/dataset/20029387", False, 79_766_954,
            md5="8321d5d3ffc99db2a5c71edca4189735",
            purpose="Optional human K562 CRISPRi pseudobulk generalization benchmark.",
        ),
    )


@dataclass
class CatalogConfig:
    raw_root: Path = RAW_ROOT / "01_raw_downloads"
    manifest_root: Path = RAW_ROOT / "00_manifests"
    download_enabled: bool = False
    selected_resource_ids: tuple[str, ...] = (
        "sergio_source",
        "sccigen_example1_count",
        "sccigen_example1_meta",
        "sccigen_example1_truth",
        "sccigen_example1_parameters",
        "human_tf_target",
        "human_ligand_receptor",
        "spatialperturbseq_source",
        "spatial_perturbseq_stereo",
    )
    chunk_bytes: int = 1024 * 1024
    timeout_seconds: int = 120
    resources: tuple[ResourceSpec, ...] = field(default_factory=public_resources)


@dataclass
class InstallConfig:
    external_root: Path = PROJECT_ROOT / "vendor" / "external_simulators"
    sergio_archive: Path = RAW_ROOT / "01_raw_downloads" / "public_simulator" / "SERGIO-master.zip"
    install_sergio_source: bool = True
    # The downloaded official sCCIgen example is prepared directly by Python.
    # Enable this only when authoring a new simulation with the R package; that
    # optional path also requires the large Giotto R dependency.
    install_sccigen_with_r: bool = False
    install_spatialperturbseq_with_r: bool = False
    rscript_executable: str = "Rscript"
    sccigen_repository: str = "songxiaoyu/sCCIgen"
    spatialperturbseq_repository: str = "kimberle9/spatialperturbseq"


@dataclass
class SergioConfig:
    official_repo: Path = PROJECT_ROOT / "vendor" / "external_simulators" / "SERGIO"
    official_dataset_relative: Path = Path("data_sets/De-noised_100G_9T_300cPerT_4_DS1")
    interaction_filename: str = "Interaction_cID_4.txt"
    regulator_filename: str = "Regs_cID_4.txt"
    output_dir: Path = DATA_ROOT / "02_prepared" / "public_simulators" / "sergio_grn"
    number_genes: int = 100
    number_bins: int = 9
    cells_per_bin: int = 50
    noise_parameter: float = 1.0
    decay: float = 0.8
    sampling_state: int = 15
    noise_type: str = "dpd"
    shared_coop_state: float = 2.0
    add_technical_noise: bool = True
    dropout_shape: float = 6.5
    dropout_percentile: float = 82.0
    random_seed: int = 20260902
    replicate_seeds: tuple[int, ...] = (20260902, 20260903, 20260904, 20260905, 20260906)
    coordinate_sd: float = 0.35
    run_official_simulator: bool = True


@dataclass
class SCCIGenConfig:
    raw_dir: Path = RAW_ROOT / "01_raw_downloads" / "public_simulator"
    count_file: str = "example1_count.tsv"
    metadata_file: str = "example1_meta.tsv"
    pattern_file: str = "example1_expr_pattern.tsv"
    parameter_file: str = "example1_parameter.yml"
    output_dir: Path = DATA_ROOT / "02_prepared" / "public_simulators" / "sccigen_example1"
    tf_reference: Path = raw_dir / "Human-TF-Target.tsv"
    lr_reference: Path = raw_dir / "Human-ligand-receptor.tsv"
    dataset_id: str = "sccigen_human_breast_example1"
    sample_id: str = "sCCIgen_example1"
    source_label: str = "sCCIgen official commit-pinned pre-simulated example1"
    species: str = "human"
    tissue: str = "breast"
    maximum_tfs: int = 30
    maximum_lr_pairs: int = 200
    maximum_truth_pairs: int = 100
    random_seed: int = 20260902
    # Optional: run a newly authored official YAML through R before conversion.
    run_official_r_simulation: bool = False
    rscript_executable: str = "Rscript"
    official_parameter_file: Path = RAW_ROOT / "01_raw_downloads" / "public_simulator" / "custom_sccigen.yml"
    official_model_fit_file: Path | None = None

