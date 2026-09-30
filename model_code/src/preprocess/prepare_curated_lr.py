"""Create a strict mouse ligand-receptor catalog from the local candidates.

The raw ``ligrecextra`` table is treated as a candidate network.  An edge is
retained only when it has an expert LR source, a literature reference, usable
mouse gene symbols, and orthologous OmniPath intercell localization compatible
with an extracellular/cell-surface transmitter and a transmembrane receiver.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from preprocess.common import (
    PROJECT_ROOT,
    SIMPLE_SYMBOL,
    file_digest,
    is_uniprot_accession,
    write_json,
)
from preprocess.evidence_common import (
    as_bool,
    read_csv_auto,
    update_summary_item,
    write_frame_atomic,
)


from configs.real_data.settings.prepare_curated_lr import CuratedLRConfig, CONFIG





def _role_table(path: Path, role: str) -> pd.DataFrame:
    frame = read_csv_auto(path, sep="\t", low_memory=False)
    if "genesymbol" not in frame.columns:
        raise ValueError(f"OmniPath {role} table has no genesymbol column: {path}")
    frame = frame.loc[frame["genesymbol"].notna()].copy()
    frame["gene_key"] = frame["genesymbol"].astype(str).str.strip().str.upper()
    frame = frame.loc[
        frame["gene_key"].ne("") & ~frame["gene_key"].str.startswith("COMPLEX:")
    ].copy()
    for column in (
        "secreted",
        "plasma_membrane_transmembrane",
        "plasma_membrane_peripheral",
        "transmitter",
        "receiver",
    ):
        frame[column] = as_bool(frame[column]) if column in frame.columns else False
    frame["consensus_score"] = pd.to_numeric(
        frame.get("consensus_score", 0), errors="coerce"
    ).fillna(0.0)
    grouped = frame.groupby("gene_key", as_index=False).agg(
        role_consensus=("consensus_score", "max"),
        secreted=("secreted", "max"),
        transmembrane=("plasma_membrane_transmembrane", "max"),
        membrane_peripheral=("plasma_membrane_peripheral", "max"),
    )
    grouped[f"has_{role}_role"] = True
    return grouped


def _resource_support(value: object, allowed: set[str]) -> tuple[int, str]:
    tokens = {token.strip() for token in str(value).split(";") if token.strip()}
    matched = sorted(tokens.intersection(allowed))
    return len(matched), ";".join(matched)


def curate_lr(
    raw: pd.DataFrame,
    ligand_roles: pd.DataFrame,
    receptor_roles: pd.DataFrame,
    config: CuratedLRConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_column = "source_genesymbol" if "source_genesymbol" in raw.columns else "source"
    target_column = "target_genesymbol" if "target_genesymbol" in raw.columns else "target"
    required = {source_column, target_column, "sources"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Candidate LR table is missing columns: {sorted(missing)}")
    work = raw.copy()
    work["ligand"] = work[source_column].astype("string").str.strip()
    work["receptor"] = work[target_column].astype("string").str.strip()
    work["ligand_key"] = work["ligand"].str.upper()
    work["receptor_key"] = work["receptor"].str.upper()
    work["reject_reason"] = ""

    def reject(mask: pd.Series, reason: str) -> None:
        available = mask & work["reject_reason"].eq("")
        work.loc[available, "reject_reason"] = reason

    simple = work["ligand"].map(
        lambda value: bool(SIMPLE_SYMBOL.fullmatch(str(value)))
    ) & work["receptor"].map(lambda value: bool(SIMPLE_SYMBOL.fullmatch(str(value))))
    reject(~simple, "non_simple_or_complex_symbol")
    if config.exclude_unmapped_uniprot_accessions:
        reject(
            is_uniprot_accession(work["ligand"])
            | is_uniprot_accession(work["receptor"]),
            "unmapped_uniprot_accession",
        )
    reject(work["ligand_key"].eq(work["receptor_key"]), "self_interaction")
    if config.require_directed and "is_directed" in work.columns:
        reject(~as_bool(work["is_directed"]), "not_directed")

    expert = set(config.expert_resources)
    support = work["sources"].map(lambda value: _resource_support(value, expert))
    work["expert_resource_count"] = support.map(lambda value: value[0])
    work["expert_sources"] = support.map(lambda value: value[1])
    reject(work["expert_resource_count"].eq(0), "no_expert_lr_resource")
    curation = pd.to_numeric(work.get("curation_effort", 0), errors="coerce").fillna(0.0)
    work["curation_effort_numeric"] = curation
    reject(curation.lt(config.minimum_curation_effort), "low_curation_effort")
    if config.require_literature_reference:
        references = work.get("references", pd.Series("", index=work.index)).fillna("").astype(str)
        reject(references.str.strip().isin({"", "nan", "None"}), "no_literature_reference")

    ligand_roles = ligand_roles.add_prefix("ligand_").rename(
        columns={"ligand_gene_key": "ligand_key"}
    )
    receptor_roles = receptor_roles.add_prefix("receptor_").rename(
        columns={"receptor_gene_key": "receptor_key"}
    )
    work = work.merge(ligand_roles, on="ligand_key", how="left").merge(
        receptor_roles, on="receptor_key", how="left"
    )
    reject(work["ligand_has_ligand_role"].ne(True), "ligand_role_not_confirmed")
    reject(work["receptor_has_receptor_role"].ne(True), "receptor_role_not_confirmed")
    reject(
        work["ligand_role_consensus"].fillna(0).lt(config.minimum_role_consensus),
        "low_ligand_role_consensus",
    )
    reject(
        work["receptor_role_consensus"].fillna(0).lt(config.minimum_role_consensus),
        "low_receptor_role_consensus",
    )
    if config.require_ligand_secreted_or_surface:
        ligand_localized = (
            work["ligand_secreted"].eq(True)
            | work["ligand_transmembrane"].eq(True)
            | work["ligand_membrane_peripheral"].eq(True)
        )
        reject(~ligand_localized, "ligand_not_secreted_or_surface")
    if config.require_receptor_transmembrane:
        reject(work["receptor_transmembrane"].ne(True), "receptor_not_transmembrane")

    rejected = work.loc[work["reject_reason"].ne("")].copy()
    kept = work.loc[work["reject_reason"].eq("")].copy()
    if kept.empty:
        raise ValueError("Strict LR filtering retained zero pairs; inspect lr_filter_audit.csv")

    maximum_curation = max(float(np.log1p(curation).max()), 1.0)
    kept["curation_score"] = np.log1p(kept["curation_effort_numeric"]) / maximum_curation
    kept["resource_score"] = (kept["expert_resource_count"] / 3.0).clip(0.0, 1.0)
    kept["role_score"] = (
        np.minimum(kept["ligand_role_consensus"].fillna(0) / 3.0, 1.0)
        * np.minimum(kept["receptor_role_consensus"].fillna(0) / 3.0, 1.0)
    )
    kept["confidence"] = (
        0.35
        + 0.25 * kept["curation_score"]
        + 0.15 * kept["resource_score"]
        + 0.25 * kept["role_score"]
    ).clip(0.0, 1.0)
    kept["signal_type"] = np.where(kept["ligand_secreted"].eq(True), "secreted", "contact")
    kept["rho"] = np.where(
        kept["signal_type"].eq("secreted"), config.secreted_rho, config.contact_rho
    )
    kept["lr_id"] = kept["ligand"].astype(str) + "__" + kept["receptor"].astype(str)
    kept = kept.sort_values(
        ["confidence", "curation_effort_numeric", "lr_id"],
        ascending=[False, False, True],
    ).drop_duplicates(["ligand_key", "receptor_key"])

    detailed_columns = [
        "lr_id",
        "ligand",
        "receptor",
        "signal_type",
        "rho",
        "confidence",
        "expert_sources",
        "expert_resource_count",
        "curation_effort_numeric",
        "ligand_role_consensus",
        "receptor_role_consensus",
        "ligand_secreted",
        "ligand_transmembrane",
        "receptor_transmembrane",
        "sources",
        "references",
    ]
    detailed = kept[[column for column in detailed_columns if column in kept.columns]].copy()
    model = detailed[["lr_id", "ligand", "receptor", "signal_type", "rho", "confidence"]].copy()
    reasons = rejected["reject_reason"].value_counts()
    audit = pd.DataFrame(
        {
            "item": ["input_rows", "retained_unique_pairs", *[f"rejected_{x}" for x in reasons.index]],
            "value": [len(raw), len(model), *reasons.tolist()],
        }
    )
    rejected_examples = rejected[
        [
            column
            for column in (
                "ligand",
                "receptor",
                "reject_reason",
                "sources",
                "references",
                "curation_effort_numeric",
            )
            if column in rejected.columns
        ]
    ].head(config.maximum_rejected_examples)
    return (
        model.reset_index(drop=True),
        detailed.reset_index(drop=True),
        audit,
        rejected_examples.reset_index(drop=True),
    )


def main(config: CuratedLRConfig = CONFIG) -> Path:
    for path in (config.candidate_lr_file, config.ligand_role_file, config.receptor_role_file):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Strict LR input is missing: {path}. Run download_public_resources.py first."
            )
    print("[1/4] Reading LR candidates and OmniPath intercell roles")
    raw = pd.read_csv(config.candidate_lr_file, sep="\t", low_memory=False)
    ligand_roles = _role_table(config.ligand_role_file, "ligand")
    receptor_roles = _role_table(config.receptor_role_file, "receptor")
    print("[2/4] Applying expert-source, literature, role and topology gates")
    model, detailed, audit, rejected_examples = curate_lr(
        raw, ligand_roles, receptor_roles, config
    )

    output = Path(config.reference_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / config.model_output_name
    backup = output / config.backup_name
    if model_path.exists() and not backup.exists():
        shutil.copy2(model_path, backup)
    print("[3/4] Writing strict LR model table and audit")
    write_frame_atomic(model, model_path)
    write_frame_atomic(detailed, output / config.detailed_output_name)
    write_frame_atomic(audit, output / config.audit_output_name)
    write_frame_atomic(rejected_examples, output / config.rejected_output_name)
    update_summary_item(output / "reference_summary.csv", "lr_pairs", len(model))
    update_summary_item(output / "reference_summary.csv", "strict_lr_pairs", len(model))
    write_json(
        output / config.manifest_name,
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": asdict(config),
            "candidate_lr_sha256": file_digest(config.candidate_lr_file),
            "ligand_role_sha256": file_digest(config.ligand_role_file),
            "receptor_role_sha256": file_digest(config.receptor_role_file),
            "retained_pairs": len(model),
            "role_mapping": "human OmniPath intercell genesymbol -> mouse orthologous genesymbol, case-insensitive",
            "limitations": [
                "Gene-symbol orthology does not cover mouse-specific or renamed orthologs.",
                "Protein complexes absent from the raw mouse candidate table cannot be reconstructed here.",
            ],
        },
    )
    print(audit.to_string(index=False))
    print("[4/4] Complete")
    return model_path



