"""Internal real-data RegDiffusion/CellOracle worker for selected output roots."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT, ROOT/'src'):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))


def main():
    from configs.real_data.support_benchmark_extension import ExtensionSettings
    from real_support_benchmark.extension_workers import fit
    from real_support_benchmark.prepare import sha

    if len(sys.argv) != 1:
        raise ValueError('No experiment CLI arguments')
    out = Path.cwd()
    job = json.loads((out/'job.json').read_text(encoding='utf-8'))
    try:
        method, dataset = job['method'], job['dataset']
        if method not in ('RegDiffusion', 'CellOracle_Net_core'):
            raise ValueError('Wrong selected real-data worker method')
        interpreter = job['reg_python'] if method == 'RegDiffusion' else job['cell_python']
        if Path(sys.executable).resolve() != Path(interpreter).resolve():
            raise ValueError('Wrong pinned interpreter')
        cfg = replace(ExtensionSettings(), base_root=Path(job['base_root']),
                      output_root=Path(job['output_root']),
                      reg_source=Path(job['reg_source']),
                      cell_package=Path(job['cell_package']))
        root = cfg.base_root/'01_model_input'/dataset
        if root.resolve() != Path(job['input_root']).resolve():
            raise ValueError('Selected input root changed')
        for name, digest in job['input_hashes'].items():
            if sha(root/name) != digest:
                raise ValueError('Blind model input changed: '+name)
        result = fit(cfg, dataset, method)
        (out/'fit_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    except BaseException as error:
        (out/'fit_result.json').write_text(json.dumps({
            'state': 'failed', 'error_type': type(error).__name__,
            'error': str(error), 'traceback': traceback.format_exc()}, indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
