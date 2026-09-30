"""Manual, one-method-at-a-time replication of the fixed one-dataset table.

Edit only `action` here; no experiment arguments or environment path routing.
The historical round_01/round_02 plan and output directories are untouched.
"""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = 'published_tree1_500_cells110_genes_sigma0.1_1'


@dataclass(frozen=True)
class ManualScMultiSim:
    # check / JumpNicheRewire / TIGRESS / CLR / PPCOR / GRNBoost2 /
    # GENIE3 / CeSpGRN / RegDiffusion / CellOracle_Net_core / evaluate
    action: str = 'evaluate'
    dataset_id: str = DATASET
    dataset_dir: Path = ROOT/'data/processed/publication_20260920'/DATASET
    output_root: Path = ROOT/'data/result/optimization_20260921/manual_scmultisim_reproduction'
    model_python: Path = Path('C:/Anaconda/envs/jumpniche/python.exe')
    tree_python: Path = Path('C:/Anaconda/envs/jnr-trees-clean/python.exe')
    # Launch the manual script with this interpreter for RegDiffusion.
    regdiffusion_python: Path = ROOT/'vendor/external_methods/RegDiffusion/venv/Scripts/python.exe'
    cesp_source: Path = ROOT/'vendor/external_methods/CeSpGRN/src'
    reg_source: Path = ROOT/'vendor/external_methods/RegDiffusion'
    cell_package: Path = (ROOT/'data/result/optimization_20260921/single_dataset_extension/'
                          'CellOracle/venv/Lib/site-packages/celloracle')
    rscript: Path = Path('D:/SoftWare/R/R-4.4.1/bin/Rscript.exe')
    r_library: Path = Path('C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/reconstruct/data/raw/optimization_20260921/R_library')
    r_software: Path = Path('C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/reconstruct/data/raw/optimization_20260921/software')
    maximum_new_gib: float = 9.
    maximum_worker_rss_gib: float = 3.5
    minimum_available_ram_gib: float = 1.5
    minimum_free_disk_gib: float = 5.
    maximum_task_hours: float = 4.
    storage_reserve_bytes: int = 256*1024**2
    budget_roots: tuple[Path, ...] = (
        ROOT/'data/raw/publication_20260920',
        ROOT/'data/processed/publication_20260920',
        ROOT/'data/result/publication_20260920',
        ROOT/'data/raw/optimization_20260921',
        ROOT/'data/processed/optimization_20260921',
        ROOT/'data/result/optimization_20260921',
        Path('C:/Anaconda/envs/jnr-cespgrn'),
        ROOT/'vendor/external_methods/RegDiffusion',
    )
