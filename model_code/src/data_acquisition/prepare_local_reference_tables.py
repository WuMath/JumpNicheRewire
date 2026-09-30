"""Convert manually downloaded OmniPath TSVs into offline TF/LR tables.

This module never accesses the network. Edit ``ReferenceTableConfig`` and run
without command-line arguments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from grn2_project.paths import PROCESSED_DATA_ROOT, RAW_DATA_ROOT

from data_acquisition.common import file_digest, read_network, write_manifest


SIMPLE_SYMBOL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-]*$")


from configs.real_data.settings.prepare_local_reference_tables import ReferenceTableConfig, CONFIG





def _columns(frame: pd.DataFrame) -> tuple[str, str]:
    source = "source_genesymbol" if "source_genesymbol" in frame.columns else "source"
    target = "target_genesymbol" if "target_genesymbol" in frame.columns else "target"
    if source not in frame.columns or target not in frame.columns:
        raise ValueError("OmniPath table requires source/target gene-symbol columns")
    return source, target


def _confidence(frame: pd.DataFrame) -> np.ndarray:
    if "curation_effort" not in frame.columns:
        return np.ones(len(frame), dtype=float)
    values = pd.to_numeric(frame["curation_effort"], errors="coerce").fillna(0.0).to_numpy()
    maximum = float(np.log1p(values).max()) if len(values) else 0.0
    return 0.5 + 0.5 * np.log1p(values) / maximum if maximum > 0 else np.ones(len(frame))


def _simple(values: pd.Series) -> pd.Series:
    return values.notna() & values.astype(str).map(lambda value: bool(SIMPLE_SYMBOL.fullmatch(value)))


def main(config: ReferenceTableConfig) -> Path:
    print("[1/3] Reading manually downloaded OmniPath TSVs (offline)")
    collectri_path = Path(config.collectri_file)
    ligrec_path = Path(config.ligand_receptor_file)
    collectri = read_network(collectri_path)
    ligrec = read_network(ligrec_path)

    tf_source, tf_target = _columns(collectri)
    tf_edges = collectri.copy()
    tf_edges["tf"] = tf_edges[tf_source].astype(str)
    tf_edges["target"] = tf_edges[tf_target].astype(str)
    if config.exclude_complex_symbols:
        tf_edges = tf_edges.loc[_simple(tf_edges["tf"]) & _simple(tf_edges["target"])].copy()
    if tf_edges.empty:
        raise ValueError("No simple TF-target gene-symbol edges remain after filtering CollecTRI")
    tf_edges["grn"] = _confidence(tf_edges)
    prior = (
        tf_edges.groupby(["tf", "target"], as_index=False)["grn"].max()
        .assign(motif=np.nan, atac=np.nan)
        [["tf", "target", "motif", "atac", "grn"]]
    )
    tf_list = pd.DataFrame({"tf": sorted(prior["tf"].unique())})

    lr_source, lr_target = _columns(ligrec)
    lr_edges = ligrec.copy()
    lr_edges["ligand"] = lr_edges[lr_source].astype(str)
    lr_edges["receptor"] = lr_edges[lr_target].astype(str)
    if config.exclude_complex_symbols:
        lr_edges = lr_edges.loc[_simple(lr_edges["ligand"]) & _simple(lr_edges["receptor"])].copy()
    lr_edges = lr_edges.loc[lr_edges["ligand"] != lr_edges["receptor"]].copy()
    if lr_edges.empty:
        raise ValueError("No simple ligand-receptor gene-symbol pairs remain after filtering")
    lr_edges["confidence"] = _confidence(lr_edges)
    lr_pairs = (
        lr_edges.sort_values("confidence", ascending=False)
        .drop_duplicates(["ligand", "receptor"])
        [["ligand", "receptor", "confidence"]]
    )
    lr_pairs.insert(0, "lr_id", lr_pairs["ligand"] + "__" + lr_pairs["receptor"])
    lr_pairs["signal_type"] = "unknown"
    lr_pairs["rho"] = 1.0
    lr_pairs = lr_pairs[["lr_id", "ligand", "receptor", "signal_type", "rho", "confidence"]]

    print("[2/3] Writing full offline reference tables")
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tf_list.to_csv(output / "tf_list.csv", index=False)
    prior.to_csv(output / "prior_tf_gene.csv", index=False)
    lr_pairs.to_csv(output / "lr_pairs.csv", index=False)
    pd.DataFrame(columns=["lr_id", "tf", "score"]).to_csv(output / "prior_lr_tf.csv", index=False)
    summary = pd.DataFrame(
        {
            "item": ["organism", "taxonomy_id", "tfs", "tf_target_edges", "lr_pairs"],
            "value": [config.organism, config.taxonomy_id, len(tf_list), len(prior), len(lr_pairs)],
        }
    )
    summary.to_csv(output / "reference_summary.csv", index=False)
    write_manifest(
        output / "source_manifest.json",
        {
            "mode": "manual_download_local_reference_processing",
            "organism": config.organism,
            "taxonomy_id": config.taxonomy_id,
            "collectri": {
                "path": str(collectri_path.resolve()),
                "sha256": file_digest(collectri_path),
            },
            "ligand_receptor": {
                "path": str(ligrec_path.resolve()),
                "sha256": file_digest(ligrec_path),
            },
            "notes": [
                "CollecTRI is used as a soft TF-target prior, not as benchmark truth.",
                "ligrecextra pairs are exported with signal_type=unknown and rho=1.0.",
                "Complex symbols are excluded by default; add curated complex definitions separately.",
            ],
        },
    )
    print(summary.to_string(index=False))
    print("[3/3] Complete")
    return output



