"""Standardize the two local ModelData reference tables without networking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import pandas as pd

from preprocess.common import PROJECT_ROOT, file_digest, standardize_reference_tables, write_json


from configs.real_data.settings.prepare_references import ReferenceConfig, CONFIG





def main(config: ReferenceConfig = CONFIG) -> Path:
    for path in (config.tf_target_file, config.ligand_receptor_file):
        if not Path(path).exists():
            raise FileNotFoundError(f"Reference file is missing: {path}")
    print("[1/3] Reading local ModelData TSV files")
    tf_list, prior, lr_pairs, filter_summary = standardize_reference_tables(
        Path(config.tf_target_file),
        Path(config.ligand_receptor_file),
        exclude_unmapped_uniprot_accessions=config.exclude_unmapped_uniprot_accessions,
    )
    print("[2/3] Writing standardized reference tables")
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tf_list.to_csv(output / "tf_list.csv", index=False)
    prior.to_csv(output / "prior_tf_gene.csv", index=False)
    lr_pairs.to_csv(output / "lr_pairs.csv", index=False)
    pd.DataFrame(columns=["lr_id", "tf", "score"]).to_csv(
        output / "prior_lr_tf.csv", index=False
    )
    summary = pd.DataFrame(
        {
            "item": ["organism", "taxonomy_id", "tfs", "tf_target_edges", "lr_pairs"],
            "value": [config.organism, config.taxonomy_id, len(tf_list), len(prior), len(lr_pairs)],
        }
    )
    summary.to_csv(output / "reference_summary.csv", index=False)
    pd.DataFrame(
        {"item": list(filter_summary), "value": list(filter_summary.values())}
    ).to_csv(output / "reference_filter_summary.csv", index=False)
    write_json(
        output / "source_manifest.json",
        {
            "mode": "offline_modeldata_reference_standardization",
            "organism": config.organism,
            "taxonomy_id": config.taxonomy_id,
            "exclude_unmapped_uniprot_accessions": config.exclude_unmapped_uniprot_accessions,
            "filter_summary": filter_summary,
            "tf_target_file": {
                "path": str(Path(config.tf_target_file).resolve()),
                "sha256": file_digest(Path(config.tf_target_file)),
            },
            "ligand_receptor_file": {
                "path": str(Path(config.ligand_receptor_file).resolve()),
                "sha256": file_digest(Path(config.ligand_receptor_file)),
            },
            "notes": [
                "TF-target confidence is derived from normalized curation_effort and used as a soft prior.",
                "LR signal_type remains unknown and rho is 1.0 until separately curated.",
                "Entries that cannot be represented as one simple gene symbol are excluded.",
                "UniProt accessions left unresolved in gene-symbol fields are excluded by default.",
            ],
        },
    )
    print(summary.to_string(index=False))
    print(pd.Series(filter_summary, name="value").to_string())
    print("[3/3] Complete")
    return output



