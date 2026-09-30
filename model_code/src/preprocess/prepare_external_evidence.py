"""Validate and install independent activity and perturbation evidence.

This script never derives activity from the same RNA matrix and never fabricates
perturbations.  The confirmation flags must be changed manually after checking
the experimental design and identifiers of user-supplied files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from preprocess.common import PROJECT_ROOT, write_json
from preprocess.evidence_common import write_frame_atomic


from configs.real_data.settings.prepare_external_evidence import ExternalEvidenceConfig, CONFIG





def _validate_activity(
    source: Path,
    expression_ids: pd.Index,
    allowed_columns: set[str],
    label: str,
    require_all_ids: bool,
) -> pd.DataFrame:
    frame = pd.read_csv(source)
    if frame.shape[1] < 2:
        raise ValueError(f"{label} needs cell_id plus at least one activity column")
    first = frame.columns[0]
    frame = frame.rename(columns={first: "cell_id"})
    frame["cell_id"] = frame["cell_id"].astype(str)
    if frame["cell_id"].duplicated().any():
        raise ValueError(f"{label} contains duplicated cell IDs")
    activity_columns = [column for column in frame.columns if column != "cell_id"]
    overlap = set(activity_columns).intersection(allowed_columns)
    if not overlap:
        raise ValueError(f"{label} has no columns matching the model entities")
    values = frame[activity_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{label} must contain only finite activity values")
    frame[activity_columns] = values
    indexed = frame.set_index("cell_id")
    missing = expression_ids.difference(indexed.index)
    if require_all_ids and len(missing):
        raise ValueError(f"{label} is missing {len(missing)} model cell IDs")
    indexed = indexed.loc[expression_ids.intersection(indexed.index)]
    indexed.index.name = "cell_id"
    return indexed.reset_index()


def _install(frame: pd.DataFrame, destination: Path, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Evidence file already exists: {destination}. "
            "Set overwrite_existing_evidence=True only after reviewing the new experiment."
        )
    write_frame_atomic(frame, destination)


def main(config: ExternalEvidenceConfig = CONFIG) -> Path:
    root = Path(config.processed_input_dir)
    expression_path = root / "expression.csv"
    expression_header = pd.read_csv(expression_path, nrows=0).columns.astype(str).tolist()
    expression_id_column = expression_header[0]
    expression_ids = pd.Index(
        pd.read_csv(
            expression_path,
            usecols=[expression_id_column],
            dtype={expression_id_column: "string"},
        )[expression_id_column].astype(str),
        name="cell_id",
    )
    expression_genes = set(expression_header[1:])
    tf_table = pd.read_csv(root / "tf_list.csv")
    lr_table = pd.read_csv(root / "lr_pairs.csv")
    lr_ids = set(lr_table["lr_id"].astype(str))
    ligand_entities = lr_ids | set(lr_table["ligand"].astype(str))
    receptor_entities = lr_ids | set(lr_table["receptor"].astype(str))
    tf_entities = set(tf_table["tf"].astype(str))
    audit_rows: list[dict[str, object]] = []

    activity_specs = (
        (config.ligand_activity_source, "ligand_activity.csv", ligand_entities, "ligand_activity"),
        (config.receptor_activity_source, "receptor_activity.csv", receptor_entities, "receptor_activity"),
        (config.tf_activity_source, "tf_activity.csv", tf_entities, "tf_activity"),
    )
    present_activity = [spec for spec in activity_specs if Path(spec[0]).exists()]
    if present_activity and not config.confirm_activity_is_independently_measured:
        raise ValueError(
            "Activity files are present but independence is not confirmed. Verify that they are "
            "not scores derived from the same expression.csv, then set the confirmation flag True."
        )
    for source, destination_name, allowed, label in activity_specs:
        source = Path(source)
        if not source.exists():
            audit_rows.append({"item": label, "available": False, "action": "not installed"})
            continue
        frame = _validate_activity(
            source,
            expression_ids,
            allowed,
            label,
            config.require_all_expression_ids_in_activity,
        )
        _install(frame, root / destination_name, config.overwrite_existing_evidence)
        audit_rows.append(
            {"item": label, "available": True, "action": f"installed {len(frame)} rows"}
        )

    perturbation_source = Path(config.perturbation_source)
    if perturbation_source.exists():
        if not config.confirm_perturbation_has_control_and_biological_replicates:
            raise ValueError(
                "Perturbation table is present but design quality is not confirmed. "
                "Verify controls and biological replicates, then set the confirmation flag True."
            )
        perturbation = pd.read_csv(perturbation_source)
        required = {"region", "lr_id", "tf", "target", "delta_observed", "design_quality"}
        missing = required.difference(perturbation.columns)
        if missing:
            raise ValueError(f"Perturbation table is missing columns: {sorted(missing)}")
        perturbation["delta_observed"] = pd.to_numeric(
            perturbation["delta_observed"], errors="raise"
        )
        if not np.isfinite(perturbation["delta_observed"]).all():
            raise ValueError("Perturbation delta_observed must be finite")
        unknown_lr = set(perturbation["lr_id"].astype(str)).difference(lr_ids)
        unknown_tf = set(perturbation["tf"].astype(str)).difference(tf_entities)
        unknown_genes = set(perturbation["target"].astype(str)).difference(expression_genes)
        if unknown_lr or unknown_tf or unknown_genes:
            raise ValueError(
                f"Perturbation identifiers do not match model inputs: "
                f"unknown_lr={len(unknown_lr)}, unknown_tf={len(unknown_tf)}, "
                f"unknown_target={len(unknown_genes)}"
            )
        _install(
            perturbation,
            root / "perturbations.csv",
            config.overwrite_existing_evidence,
        )
        audit_rows.append(
            {"item": "perturbations", "available": True, "action": f"installed {len(perturbation)} rows"}
        )
    else:
        audit_rows.append(
            {"item": "perturbations", "available": False, "action": "no Level-3 evidence"}
        )

    audit = pd.DataFrame(audit_rows)
    write_frame_atomic(audit, root / config.audit_name)
    write_json(
        root / config.manifest_name,
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": asdict(config),
            "audit": audit_rows,
            "warning": (
                "Installed activity is accepted as Level-2 only if experimentally independent; "
                "Level-3 additionally requires direction-consistent perturbation and design review."
            ),
        },
    )
    print(audit.to_string(index=False))
    return root / config.audit_name



