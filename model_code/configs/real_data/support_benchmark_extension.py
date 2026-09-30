"""Six additional real-support fits; original 14-fit study remains immutable."""

from dataclasses import dataclass
from pathlib import Path

from configs.common.experiment import ExperimentConfig, InputSelection, RunOptions
from configs.common.model import ProjectConfig
from configs.real_data.support_benchmark import ROOT, SupportBenchmarkSettings


@dataclass(frozen=True)
class ExtensionSettings(SupportBenchmarkSettings):
    base_root: Path = ROOT / 'data/result/real_support_benchmark_20260925'
    output_root: Path = ROOT / 'data/result/real_support_benchmark_20260925/05_extension'
    cesp_python: Path = Path('C:/Anaconda/envs/jnr-cespgrn/python.exe')
    reg_python: Path = ROOT / 'vendor/external_methods/RegDiffusion/venv/Scripts/python.exe'
    cell_python: Path = (ROOT / 'data/result/optimization_20260921/single_dataset_extension/'
                         'CellOracle/venv/Scripts/python.exe')
    cesp_source: Path = ROOT / 'vendor/external_methods/CeSpGRN/src'
    reg_source: Path = ROOT / 'vendor/external_methods/RegDiffusion'
    cell_package: Path = (ROOT / 'data/result/optimization_20260921/single_dataset_extension/'
                          'CellOracle/venv/Lib/site-packages/celloracle')


def build_config():
    settings = ExtensionSettings()
    return ExperimentConfig('support_benchmark_extension', 'real_data', 'evidence',
                            'real_support_extension',
                            InputSelection(source='prepared',
                                           dataset_ids=('gse165141_liver', 'gse205055_p22')),
                            RunOptions(output_root=settings.output_root), ProjectConfig(), settings)
