"""Only input checks are enabled until the new development plan is registered."""
from dataclasses import asdict
import json
from .adapter import load_development
from .persistence import ResourceGuard


def run(settings, action):
    if action not in ('check', 'prepare', 'reuse'):
        raise PermissionError('v2: no registered round_02 plan; run/freeze/test/evaluate/cases are locked')
    guard = ResourceGuard(settings)
    guard()
    rows = []
    for dataset in settings.dataset_ids:
        guard()
        prepared = load_development(settings, dataset)
        masked = {target: [lr for lr, members in zip(prepared.lr_ids, prepared.lr_members) if target in members]
                  for target in prepared.targets}
        rows.append(dict(**prepared.audit, target_lr_mask=masked))
        print(f'[input only] {dataset}: {len(prepared.cells)} cells, {len(prepared.tfs)*len(prepared.targets)} candidates', flush=True)
    report = dict(state='interface_checks_only', adapter=asdict(settings.adapter), datasets=rows,
                  fitted_tasks=0, truth_read=False, round02_registered=False, cases_enabled=False)
    if action == 'check': return report
    settings.output_root.mkdir(parents=True, exist_ok=True)
    path = settings.output_root/'input_adapter_audit.json'
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    guard(reserve_bytes=2*len(serialized.encode('utf8')))
    if path.exists():
        if json.loads(path.read_text(encoding='utf8')) != report:
            raise FileExistsError('Input/settings changed: preserve old audit and choose a new output directory')
        return path
    temporary = path.with_suffix('.writing')
    with temporary.open('x', encoding='utf8') as stream: stream.write(serialized)
    temporary.replace(path)
    return path
