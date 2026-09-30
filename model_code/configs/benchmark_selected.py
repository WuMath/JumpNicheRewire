"""One editable selection for the three-dataset benchmark.

Only edit SELECTED. Invoke the existing main_jumpniche.py without arguments.
Historical inputs, references, fits and tables remain read-only.
"""
from dataclasses import dataclass
from pathlib import Path

from configs.common.experiment import ExperimentConfig, InputSelection, RunOptions
from configs.common.model import ProjectConfig
from configs.real_data.support_benchmark import SupportBenchmarkSettings
from configs.real_data.support_benchmark_extension import ExtensionSettings


ROOT = Path(__file__).resolve().parents[1]
SCMULTISIM = 'scmultisim'
LIVER = 'gse165141_liver'
P22 = 'gse205055_p22'
METHODS = ('JumpNicheRewire', 'TIGRESS', 'CLR', 'PPCOR', 'GRNBoost2',
           'GENIE3', 'CeSpGRN', 'RegDiffusion', 'CellOracle_Net_core')


@dataclass(frozen=True)
class BenchmarkSelection:
    # check / run / status / evaluate; change this field, never a CLI argument.
    action: str = 'evaluate'
    datasets: tuple[str, ...] = (SCMULTISIM, P22)
    methods: tuple[str, ...] = ('JumpNicheRewire', 'CeSpGRN', 'CLR', 'GENIE3',
                                'GRNBoost2', 'PPCOR', 'RegDiffusion')
    # Fresh fits go here. Change output_root for a different scientific plan.
    output_root: Path = ROOT.parent / 'reproduction_runs/selected_benchmark'
    sc_input: Path = ROOT.parent / 'reproduction_runs/prepared/scmultisim'
    real_input_root: Path = ROOT.parent / 'reproduction_runs/prepared/p22_aux/01_model_input'
    real_reference_root: Path = ROOT.parent / 'reproduction_runs/prepared/p22_aux/02_reference'
    motif_catalog: Path = ROOT.parent / 'reference_data/motif_tf_gene.parquet'
    model_python: Path = SupportBenchmarkSettings().model_python
    tree_python: Path = SupportBenchmarkSettings().tree_python
    cesp_python: Path = ExtensionSettings().cesp_python
    reg_python: Path = Path('C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/reconstruct/vendor/external_methods/RegDiffusion/venv/Scripts/python.exe')
    cell_python: Path = ExtensionSettings().cell_python
    rscript: Path = SupportBenchmarkSettings().rscript
    r_library: Path = Path('C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/reconstruct/data/raw/optimization_20260921/R_library')
    r_software: Path = Path('C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/reconstruct/data/raw/optimization_20260921/software')
    cesp_source: Path = ROOT / 'vendor/external_methods/CeSpGRN/src'
    reg_source: Path = ROOT / 'vendor/external_methods/RegDiffusion'
    cell_package: Path = ExtensionSettings().cell_package
    seed: int = 20260925
    l1_penalty: float = .075
    fusion_penalty: float = .0005
    group_penalty: float = .01
    ridge_penalty: float = .0001
    spatial_permutations: int = 99
    # Cumulative limit includes old results, raw data, software and new output.
    maximum_new_gib: float = 15.
    cumulative_storage_limit: int = 15 * 1024**3
    maximum_worker_rss_gib: float = 3.5
    worker_rss_limit: int = int(3.5 * 1024**3)
    minimum_available_ram_gib: float = 1.5
    minimum_available_ram: int = int(1.5 * 1024**3)
    minimum_free_disk_gib: float = 5.
    minimum_disk_free: int = 5 * 1024**3
    maximum_task_hours: float = 4.
    task_limit_seconds: int = 4 * 3600
    storage_reserve_bytes: int = 256 * 1024**2
    budget_roots: tuple[Path, ...] = (ROOT.parent,)


SELECTED = BenchmarkSelection()


def build_config():
    return ExperimentConfig('benchmark_selected', 'selected', 'evidence',
                            'selected_benchmark', InputSelection(source='prepared'),
                            RunOptions(output_root=SELECTED.output_root),
                            ProjectConfig(), SELECTED)
