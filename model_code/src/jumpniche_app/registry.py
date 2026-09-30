"""Selection is snapshotted locally; original registries are never rewritten."""

from __future__ import annotations
import csv
from pathlib import Path
import os
from .safety import writable


MODEL_FILES = ('expression.csv', 'coordinates.csv', 'metadata.csv', 'tf_list.csv',
               'lr_pairs.csv', 'prior_tf_gene.csv', 'prior_lr_tf.csv')


def read_rows(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset registry missing: {path}. Run preprocess prepare/reuse first.")
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Registry is empty: {path}")
    return rows


def write_rows(path, rows):
    target = writable(path)
    if not rows:
        raise ValueError('Cannot write an empty dataset selection')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name('.' + target.name + '.writing')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, target)
    return target


def selected_rows(config, source=None):
    selection = config.inputs
    origin = source or selection.source
    if origin not in {'existing', 'prepared'}:
        raise ValueError('DatasetParameters.source must be existing or prepared')
    path = selection.existing_registry if origin == 'existing' else selection.prepared_registry
    rows = read_rows(path)
    ids = [str(row.get('dataset_id', '')) for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError('Registry needs non-empty, unique dataset IDs')
    if selection.dataset_ids:
        lookup = dict(zip(ids, rows))
        missing = set(selection.dataset_ids) - set(lookup)
        if missing:
            raise ValueError(f'Dataset IDs missing from {path}: {sorted(missing)}')
        rows = [lookup[value] for value in dict.fromkeys(selection.dataset_ids)]
    elif config.workflow == 'evidence':
        raise ValueError('Evidence profiles require explicit dataset_ids; implicit all-data runs are disabled')
    if config.workflow == 'formal':
        profile = config.formal_data.profile
        if {row.get('benchmark_profile') for row in rows} != {profile}:
            raise ValueError(f'Pilot/final registry isolation failed; expected {profile}')
        if hasattr(config.formal_data, 'scenario_ids'):
            rows = [row for row in rows if row.get('scenario') in config.formal_data.scenario_ids]
        if not rows:
            raise ValueError('No selected formal scenarios')
    for row in rows:
        root = Path(row['dataset_dir'])
        if not root.is_absolute():
            root = (Path(path).parent / root).resolve()
        row['dataset_dir'] = str(root)
        if 'truth_dir' in row:
            row['truth_dir'] = str(root / 'truth')
        if 'baseline_input_dir' in row:
            row['baseline_input_dir'] = str(root / 'baseline_input')
    return rows


def snapshot(config, rows=None):
    rows = rows or selected_rows(config)
    if config.workflow == 'formal':
        destination = config.formal_model.registry_path
    else:
        destination = config.run.output_root / '00_selection/evidence_registry.csv'
    return write_rows(destination, rows)


def validate_inputs(root):
    missing = [name for name in MODEL_FILES if not (Path(root) / name).is_file()]
    if missing:
        raise FileNotFoundError(f'Model inputs incomplete at {root}: {missing}')
