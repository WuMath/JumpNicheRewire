from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from grn2_project.paths import *
from configs.common.evidence_paths import *

@dataclass
class PerturbationConfig:
    standardized_h5ad: Path = RAW_ROOT / "01_raw_downloads" / "perturbation" / "GSE274447_processed.h5ad"
    output_dir: Path = DATA_ROOT / "02_prepared" / "perturbation" / "spatial_perturbseq_mouse_brain"
    tf_reference: Path = MODEL_DATA / "Mouse-TF-Target.tsv"
    lr_reference: Path = PROCESSED_DATA_ROOT / "references/lr_pairs.csv"
    spatial_key: str = "spatial"
    perturbation_column_candidates: tuple[str, ...] = ("perturbation", "guide", "gRNA", "barcode", "target")
    sample_column_candidates: tuple[str, ...] = ("sample_id", "sample", "mouse", "batch")
    cell_type_column_candidates: tuple[str, ...] = ("cell_type", "annotation", "celltype")
    control_patterns: tuple[str, ...] = ("msafe", "safe", "control", "non-targeting", "ntc")
    minimum_cells_per_group: int = 20
    maximum_cells_per_group: int = 1000
    gene_screening_batch_cells: int = 256
    maximum_genes: int = 3000
    minimum_detection_fraction: float = 0.02
    log2fc_threshold: float = 0.5
    fdr_threshold: float = 0.05
    random_seed: int = 20260902


@dataclass
class StereoGEFConversionConfig:
    raw_archive: Path = RAW_ROOT / "01_raw_downloads/perturbation/GSE274447_RAW.tar"
    extracted_dir: Path = RAW_ROOT / "01_raw_downloads/perturbation/GSE274447_extracted"
    standardized_h5ad: Path = RAW_ROOT / "01_raw_downloads/perturbation/GSE274447_processed.h5ad"
    sample_map_file: Path = RAW_ROOT / "01_raw_downloads/perturbation/GSE274447_sample_map.csv"
    extract_archive: bool = True
    overwrite_extracted_files: bool = False
    gef_glob: str = "*.adjusted.cellbin.gef"
    bin_type: str = "cell_bins"
    gene_name_index: bool = True
    guide_detection_threshold: float = 0.0
    inter_section_gap: float = 100_000.0
    output_compression: str = "gzip"
    guide_features: tuple[str, ...] = (
        "sgrna-C9orf72", "sgrna-Cfap410", "sgrna-clu", "sgrna-dpp5",
        "sgrna-fasn", "sgrna-flcn", "sgrna-gfap", "sgrna-lrrk2",
        "sgrna-msafe", "sgrna-ndufaf", "sgrna-oligo2", "sgrna-rbfox",
        "sgrna-rraga", "sgrna-sh3gl2", "sgrna-stk39", "sgrna-srf",
        "sgrna-tbk1", "sgrna-trem2",
    )

