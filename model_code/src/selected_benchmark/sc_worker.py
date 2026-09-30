"""Internal isolated worker; invoked by the one-entry controller, never by users."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT, ROOT/'src'):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))


def main():
    from configs.public_sim_data.manual_scmultisim import ManualScMultiSim
    from reports.run_manual_scmultisim import run_one

    if len(sys.argv) != 1:
        raise ValueError('No experiment CLI arguments')
    payload = json.loads(Path('job.json').read_text(encoding='utf-8'))
    paths = ('dataset_dir', 'output_root', 'model_python', 'tree_python',
             'regdiffusion_python', 'rscript', 'r_library', 'r_software',
             'cesp_source', 'reg_source',
             'cell_package')
    for key in paths:
        payload[key] = Path(payload[key])
    payload['budget_roots'] = tuple(Path(value) for value in payload['budget_roots'])
    run_one(ManualScMultiSim(**payload))


if __name__ == '__main__':
    main()
