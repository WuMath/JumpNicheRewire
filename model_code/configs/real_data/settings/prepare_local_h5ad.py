from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from grn2_project.paths import *

@dataclass
class LocalH5ADConfig:
    """All H5AD keys, paths, and model-input parameters are configured here."""

    h5ad_file: Path = RAW_DATA_ROOT / "real_data" / "visium_ffpe_mouse_brain" / "dataset.h5ad"
    output_dir: Path = PROCESSED_DATA_ROOT / "imported" / "h5ad_manual_prepared"
    inspect_only: bool = False

    # expression_source: X, layer, or raw. A counts layer is preferred when available.
    expression_source: str = "X"
    layer_name: str = "counts"
    expression_is_normalized: bool = True
    gene_symbol_column: str | None = None  # None uses var_names/raw.var_names

    spatial_key: str = "spatial"
    spatial_dimensions: tuple[int, ...] = (0, 1)
    maximum_cells: int | None = None
    maximum_genes: int | None = 1500
    minimum_detected_cells: int = 8
    random_seed: int = 20260823

    # Metadata columns inside adata.obs; missing columns receive explicit defaults.
    sample_column_in_obs: str | None = "sample_id"
    cell_type_column_in_obs: str | None = "cell_type"
    region_column_in_obs: str | None = "region"
    condition_column_in_obs: str | None = "condition"
    additional_obs_columns: tuple[str, ...] = ()
    default_sample_id: str = "sample_1"
    default_cell_type: str = "NA"
    default_region: str = "global"
    default_condition: str = "unspecified"

    reference_mode: str = "prepared_tables"  # or omnipath_raw
    manual_reference_dir: Path = RAW_DATA_ROOT / "manual_downloads" / "omnipath_mouse"
    collectri_file: Path = manual_reference_dir / "collectri_mouse.tsv"
    ligand_receptor_network_file: Path = manual_reference_dir / "ligrecextra_mouse.tsv"
    prepared_tf_file: Path = PROCESSED_DATA_ROOT / "reference_tables" / "visium_mouse" / "tf_list.csv"
    prepared_lr_file: Path = PROCESSED_DATA_ROOT / "reference_tables" / "visium_mouse" / "lr_pairs.csv"
    prepared_prior_file: Path = PROCESSED_DATA_ROOT / "reference_tables" / "visium_mouse" / "prior_tf_gene.csv"
    maximum_tfs: int = 100
    maximum_lr_pairs: int = 300


CONFIG = LocalH5ADConfig()

