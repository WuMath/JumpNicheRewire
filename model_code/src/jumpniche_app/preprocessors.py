"""One dispatcher, one processor class per input family; imports are lazy."""

from __future__ import annotations
from dataclasses import asdict, is_dataclass
from pathlib import Path
import json
from .registry import MODEL_FILES, selected_rows, snapshot, validate_inputs
from .safety import writable, write_json, digest, implementation_signature


class Processor:
    def __init__(self, config):
        self.config = config

    def prepare(self):
        raise NotImplementedError

    def outputs(self, action):
        if action == 'convert':
            c = self.config.conversion
            return [getattr(c, 'standardized_h5ad', getattr(c, 'output_dir', None))]
        c = self.config.preprocessing
        return [getattr(c, 'output_dir', getattr(c, 'output_root', None))]

    def required_inputs(self, action):
        return []


class SergioProcessor(Processor):
    def required_inputs(self, action):
        c = self.config.preprocessing
        network = c.official_repo / c.official_dataset_relative
        return [network / c.interaction_filename, network / c.regulator_filename,
                c.official_repo / 'SERGIO/sergio.py']

    def prepare(self):
        from evidence_benchmark.prepare_sergio import run_sergio
        return run_sergio(self.config.preprocessing)


class SCCIGenProcessor(Processor):
    def required_inputs(self, action):
        c = self.config.preprocessing
        paths = [c.raw_dir / name for name in (c.count_file, c.metadata_file, c.pattern_file, c.parameter_file)]
        return paths + [c.tf_reference, c.lr_reference]

    def prepare(self):
        from evidence_benchmark.prepare_sccigen import run_sccigen
        c = self.config.preprocessing
        if c.run_official_r_simulation:
            # ParaSimulation writes into raw_dir; it must not be the original.
            writable(c.raw_dir)
        return run_sccigen(c)


class VisiumProcessor(Processor):
    def required_inputs(self, action):
        if action == 'convert':
            c = self.config.conversion
            return [c.matrix_h5, c.spatial_source] + [c.reference_dir / name for name in ('tf_list.csv', 'lr_pairs.csv', 'prior_tf_gene.csv', 'prior_lr_tf.csv')]
        c = self.config.preprocessing
        return [c.tf_reference, c.lr_reference] + [spec.input_dir / name for spec in c.datasets for name in MODEL_FILES]

    def convert(self):
        from preprocess.prepare_visium import main
        return main(self.config.conversion)

    def prepare(self):
        from evidence_benchmark.prepare_real import run_real
        return run_real(self.config.preprocessing)


class H5ADProcessor(Processor):
    def required_inputs(self, action):
        c = self.config.preprocessing
        return [c.h5ad_file, c.prepared_tf_file, c.prepared_lr_file, c.prepared_prior_file]

    def prepare(self):
        from data_acquisition.prepare_local_h5ad import main
        return main(self.config.preprocessing)


class PerturbationProcessor(Processor):
    def required_inputs(self, action):
        if action == 'convert':
            return [self.config.conversion.raw_archive]
        c = self.config.preprocessing
        return [c.standardized_h5ad, c.tf_reference, c.lr_reference]

    def convert(self):
        from evidence_benchmark.convert_stereo_gef import run_conversion
        c = self.config.conversion
        for path in (c.extracted_dir, c.standardized_h5ad, c.sample_map_file):
            writable(path)
        return run_conversion(c)

    def prepare(self):
        c = self.config.preprocessing
        if not c.allow_exploratory_legacy_protocol:
            raise RuntimeError('Perturbation protocol is not yet independently validated. Formal preparation is blocked; see docs/KNOWN_LIMITATIONS.md.')
        print('[EXPLORATORY ONLY] Legacy perturbation protocol; not a leakage-free formal benchmark.')
        from evidence_benchmark.prepare_perturbation import run_perturbation
        return run_perturbation(c)


class DemoProcessor(Processor):
    def prepare(self):
        from jump_niche_rewire.simulation import generate_simulated_project
        return generate_simulated_project(self.config.preprocessing)


class FormalProcessor(Processor):
    def outputs(self, action):
        return [self.config.formal_data.data_root]

    def prepare(self):
        from formal_benchmark.stage1_data import run_simulations
        writable(self.config.formal_data.data_root)
        writable(self.config.formal_data.benchmark_root)
        return run_simulations(self.config.formal_data)


PROCESSORS = {'sergio': SergioProcessor, 'sccigen': SCCIGenProcessor,
              'visium': VisiumProcessor, 'h5ad': H5ADProcessor,
              'perturbation': PerturbationProcessor, 'demo': DemoProcessor, 'formal': FormalProcessor}


def _input_fingerprints(paths):
    result = {}
    for value in paths:
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f'Missing manually downloaded/input file: {path}')
        stat = path.stat()
        result[str(path.resolve())] = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    return result


def _cache_run(config, processor, action):
    outputs = [writable(path) for path in processor.outputs(action) if path is not None]
    inputs = processor.required_inputs(action)
    fingerprint = _input_fingerprints(inputs)
    settings = config.conversion if action == 'convert' else config.preprocessing
    if config.processor == 'perturbation' and action == 'convert':
        mapping = config.conversion.sample_map_file
        if mapping.exists():
            fingerprint.update(_input_fingerprints([mapping]))
    signature = digest({'processor': config.processor, 'action': action, 'settings': settings,
                        'inputs_size_mtime': fingerprint, 'code': implementation_signature()})
    marker = config.run.output_root / '00_preprocess' / (action + '_complete.json')
    if marker.exists() and not config.overwrite_preprocessing:
        old = json.loads(marker.read_text(encoding='utf-8'))
        complete = old.get('files') and all(Path(p).is_file() and Path(p).stat().st_size == size for p, size in old['files'].items())
        if old.get('signature') == signature and complete:
            print(f'[unchanged preprocessing] {marker}')
            return old['result']
        raise FileExistsError(f'Preprocessing changed/incomplete. Use a new output directory or set overwrite_preprocessing=True: {marker}')
    if not marker.exists() and not config.overwrite_preprocessing and action != 'convert':
        occupied = [str(p) for p in outputs if p.is_file() or (p.is_dir() and any(p.iterdir()))]
        # Real pseudo-truth shares a parent with other datasets; only protect
        # its explicitly selected destination subdirectories.
        if config.processor == 'visium':
            occupied = [str(outputs[0] / s.dataset_id) for s in config.preprocessing.datasets
                        if (outputs[0] / s.dataset_id).exists()]
        if occupied:
            raise FileExistsError(f'Existing untracked prepared output: {occupied}. Set overwrite_preprocessing=True to rebuild explicitly.')
    operation = getattr(processor, 'convert' if action == 'convert' else 'prepare', None)
    if operation is None:
        raise ValueError(f'{config.processor} does not support {action}')
    result = operation()
    files = {}
    for root in outputs:
        for path in ([root] if root.is_file() else sorted(root.rglob('*'))):
            if path.is_file():
                files[str(path.resolve())] = path.stat().st_size
    if not files:
        raise RuntimeError(f'Preprocessing did not produce the expected output at {outputs}')
    write_json(marker, {'signature': signature, 'result': str(result), 'files': files,
                        'inputs_size_mtime': fingerprint, 'config': settings,
                        'cache_validation': 'input size/mtime + settings/code hash + output file existence/size'})
    return result


def run(config, action):
    if config.processor == 'real_support_extension':
        if action != 'check':
            raise ValueError('Extension uses the two frozen prepared inputs; check only')
        from real_support_benchmark.extension_workflow import run as run_extension
        return run_extension(config.preprocessing, 'check')
    if config.processor == 'real_support':
        from real_support_benchmark.prepare import prepare
        if action == 'check':
            for path in (config.preprocessing.raw_root / 'gse165141_liver/GSE165141_RAW.tar',
                         config.preprocessing.external_raw / 'GSM6753043_MouseBrain_20um_100barcodes_ATAC_matrix.tsv.gz',
                         config.preprocessing.external_raw / 'CellChatDB.mouse.rda'):
                print(f"[{'present' if path.is_file() else 'MISSING'}] {path}")
            return None
        if action == 'prepare':
            return prepare(config.preprocessing)
        if action == 'references':
            from real_support_benchmark.references import prepare as prepare_references
            return prepare_references(config.preprocessing)
        raise ValueError('Real-support preprocessing supports check / prepare / references only')
    if config.processor == 'benchmark_round02':
        from benchmark_round02.workflow import run
        if action not in ('check', 'prepare', 'reuse'):
            raise ValueError('Round02 preprocessing: check / prepare / reuse only')
        return run(config.preprocessing, action)
    if config.processor == 'decomposition_v2':
        from jump_niche_rewire_v2.interface import run
        return run(config.preprocessing, action)
    if config.processor == 'optimization':
        from benchmark_optimization.setup import run
        return run(config.preprocessing, action)
    if config.processor == 'publication':
        from publication_study.prepare import prepare
        return prepare(config.preprocessing, action)
    if config.processor == 'scmultisim':
        from evidence_benchmark.prepare_scmultisim import preprocess
        return preprocess(config, action)
    processor = PROCESSORS[config.processor](config)
    if action == 'check':
        print(f'[preprocess profile] {config.category}.{config.name}')
        # For perturbation conversion is the first step; no false requirement
        # for an H5AD which has not yet been generated.
        check_action = 'convert' if config.processor == 'perturbation' else 'prepare'
        for path in processor.required_inputs(check_action):
            print(f"[{'present' if Path(path).exists() else 'MISSING'}] {path}")
        for path in processor.outputs(check_action):
            if path is not None:
                print(f'[new output] {writable(path)}')
        print('[check only] No conversion, download, simulation or model fitting.')
        return None
    writable(config.run.output_root)
    if action == 'reuse':
        if config.workflow == 'single':
            validate_inputs(config.inputs.single_input_dir)
            return config.inputs.single_input_dir
        rows = selected_rows(config, source='existing')
        destination = config.inputs.prepared_registry
        from .registry import write_rows, read_rows
        if config.workflow == 'evidence' and Path(destination).exists():
            merged = {r['dataset_id']: r for r in read_rows(destination)}
            merged.update({r['dataset_id']: r for r in rows})
            rows = list(merged.values())
        return write_rows(destination, rows)
    if action == 'catalog':
        from configs.public_sim_data.resources import CatalogParameters
        from evidence_benchmark.catalog import run_catalog
        settings = CatalogParameters()
        if settings.download_enabled:
            raise PermissionError('This reconstructed workflow uses manual downloads; keep download_enabled=False.')
        writable(settings.manifest_root)
        return run_catalog(settings)
    if action == 'install':
        from configs.public_sim_data.resources import InstallParameters
        from evidence_benchmark.install import run_install
        settings = InstallParameters()
        writable(settings.external_root)
        if settings.install_sccigen_with_r or settings.install_spatialperturbseq_with_r:
            raise PermissionError('Use separate official R environments; this action only extracts already downloaded SERGIO source.')
        return run_install(settings)
    if action == 'references':
        if config.category != 'real_data':
            raise ValueError('Reference processing belongs to a real_data profile')
        return _references(config)
    if action == 'simulate':
        if config.processor not in {'sergio', 'sccigen', 'demo', 'formal'}:
            raise ValueError('simulate is only available for simulation profiles')
        action = 'prepare'
    if action not in {'prepare', 'convert'}:
        raise ValueError('Preprocess action: check/prepare/reuse/convert/simulate/catalog/install/references')
    if config.workflow == 'formal':
        if action != 'prepare':
            raise ValueError('Formal simulation supports prepare/simulate/reuse')
        return processor.prepare()  # Original stage 1 has its own resumable signatures.
    return _cache_run(config, processor, action)


def _references(config):
    from configs.real_data.references import ReferenceWorkflow
    from importlib import import_module
    settings = config.references or ReferenceWorkflow()
    modules = {'standardize': 'prepare_references', 'motif': 'prepare_cistarget',
               'lr': 'prepare_curated_lr', 'lr_tf': 'prepare_lr_tf'}
    for step in settings.steps:
        if step not in modules:
            raise ValueError(f'Unknown reference step: {step}')
        params = getattr(settings, step)
        destination = getattr(params, 'output_dir', getattr(params, 'reference_dir', None))
        writable(destination)
        print(f'[reference {step}] {destination}')
        import_module('preprocess.' + modules[step]).main(params)
    write_json(config.run.output_root / '00_preprocess/references_config.json', settings)
    return settings.standardize.output_dir
