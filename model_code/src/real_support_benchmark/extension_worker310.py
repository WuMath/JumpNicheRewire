"""Internal Python 3.10 worker for two pinned real-support estimators."""

import json
from pathlib import Path
import sys
import traceback


PROJECT = Path(__file__).resolve().parents[2]
for folder in (PROJECT, PROJECT / 'src'):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))


def main():
    from configs.real_data.support_benchmark_extension import ExtensionSettings
    from real_support_benchmark.extension_workers import fit
    from real_support_benchmark.prepare import sha

    out = Path.cwd()
    job = json.loads((out / 'job.json').read_text(encoding='utf-8'))
    try:
        cfg = ExtensionSettings()
        dataset = job['dataset']
        method = job['method']
        if method not in ('RegDiffusion', 'CellOracle_Net_core'):
            raise ValueError('Unknown isolated Python extension method')
        executable = cfg.reg_python if method == 'RegDiffusion' else cfg.cell_python
        if Path(sys.executable).resolve() != executable.resolve():
            raise ValueError('Wrong pinned interpreter for extension method')
        root = cfg.base_root / '01_model_input' / dataset
        for name, digest in job['input_hashes'].items():
            if sha(root / name) != digest:
                raise ValueError('Blind input changed: ' + name)
        result = fit(cfg, dataset, method)
        (out / 'fit_result.json').write_text(json.dumps(result, indent=2),
                                             encoding='utf-8')
        return 0
    except BaseException as error:
        (out / 'fit_result.json').write_text(json.dumps(dict(
            state='failed', error_type=type(error).__name__, error=str(error),
            traceback=traceback.format_exc()), indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
