"""Build receptor-to-TF and LR-to-TF soft priors from OmniPath signaling.

This is a reference-path prior, not measured activity and not causal evidence.
The model consumes only ``lr_id,tf,score``; additional path columns are retained
for mechanism auditing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import math
import re
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from preprocess.common import PROJECT_ROOT, SIMPLE_SYMBOL, is_uniprot_accession, write_json
from preprocess.evidence_common import as_bool, update_summary_item, write_frame_atomic


from configs.real_data.settings.prepare_lr_tf import LRTFConfig, CONFIG





def _edge_sign(frame: pd.DataFrame) -> pd.Series:
    stimulation = as_bool(frame["is_stimulation"]) if "is_stimulation" in frame else False
    inhibition = as_bool(frame["is_inhibition"]) if "is_inhibition" in frame else False
    if isinstance(stimulation, bool):
        stimulation = pd.Series(stimulation, index=frame.index)
    if isinstance(inhibition, bool):
        inhibition = pd.Series(inhibition, index=frame.index)
    return pd.Series(
        np.where(stimulation & ~inhibition, 1, np.where(inhibition & ~stimulation, -1, 0)),
        index=frame.index,
        dtype=int,
    )


def prepare_signaling_edges(raw: pd.DataFrame, config: LRTFConfig) -> pd.DataFrame:
    source = "source_genesymbol" if "source_genesymbol" in raw.columns else "source"
    target = "target_genesymbol" if "target_genesymbol" in raw.columns else "target"
    missing = {source, target}.difference(raw.columns)
    if missing:
        raise ValueError(f"Signaling network is missing columns: {sorted(missing)}")
    edges = raw.copy()
    edges["source_gene"] = edges[source].astype("string").str.strip()
    edges["target_gene"] = edges[target].astype("string").str.strip()
    simple = edges["source_gene"].map(
        lambda value: bool(SIMPLE_SYMBOL.fullmatch(str(value)))
    ) & edges["target_gene"].map(lambda value: bool(SIMPLE_SYMBOL.fullmatch(str(value))))
    edges = edges.loc[simple].copy()
    edges = edges.loc[
        ~is_uniprot_accession(edges["source_gene"])
        & ~is_uniprot_accession(edges["target_gene"])
        & edges["source_gene"].ne(edges["target_gene"])
    ].copy()
    if config.require_directed and "is_directed" in edges.columns:
        edges = edges.loc[as_bool(edges["is_directed"])].copy()
    curation = pd.to_numeric(edges.get("curation_effort", 0), errors="coerce").fillna(0.0)
    edges = edges.loc[curation.ge(config.minimum_curation_effort)].copy()
    curation = curation.loc[edges.index]
    maximum = max(float(np.log1p(curation).max()), 1.0)
    source_count = edges.get("sources", pd.Series("", index=edges.index)).fillna("").map(
        lambda value: len({token for token in str(value).split(";") if token})
    )
    has_reference = edges.get("references", pd.Series("", index=edges.index)).fillna("").astype(str).str.len().gt(0)
    edges["edge_confidence"] = (
        0.35
        + 0.35 * np.log1p(curation) / maximum
        + 0.15 * np.minimum(source_count / 3.0, 1.0)
        + 0.15 * has_reference.astype(float)
    ).clip(1e-6, 1.0)
    edges["edge_sign"] = _edge_sign(edges)
    edges["edge_sources"] = edges.get("sources", "")
    edges = edges.sort_values("edge_confidence", ascending=False).drop_duplicates(
        ["source_gene", "target_gene"]
    )
    return edges[
        ["source_gene", "target_gene", "edge_confidence", "edge_sign", "edge_sources"]
    ].reset_index(drop=True)


def _components(value: object) -> list[str]:
    return [token for token in re.split(r"[&|]", str(value)) if token]


def receptor_paths(
    receptor: str,
    adjacency: dict[str, list[tuple[str, float, int, str]]],
    tf_keys: set[str],
    tf_name: dict[str, str],
    config: LRTFConfig,
) -> list[dict[str, object]]:
    # node, path, sum(log(edge confidence)), sign, sources
    frontier: list[tuple[str, tuple[str, ...], float, int, tuple[str, ...]]] = [
        (receptor, (receptor,), 0.0, 1, tuple())
    ]
    best_tf: dict[str, dict[str, object]] = {}
    for depth in range(1, config.maximum_path_length + 1):
        next_by_node: dict[str, tuple[str, tuple[str, ...], float, int, tuple[str, ...]]] = {}
        for node, path, log_sum, path_sign, sources in frontier:
            for target, confidence, edge_sign, edge_sources in adjacency.get(node, []):
                if target in path:
                    continue
                new_path = path + (target,)
                new_log = log_sum + math.log(max(confidence, 1e-12))
                new_sign = 0 if path_sign == 0 or edge_sign == 0 else path_sign * edge_sign
                new_sources = sources + (edge_sources,)
                score = math.exp(new_log / depth) * math.exp(
                    -config.path_length_penalty * (depth - 1)
                )
                if target in tf_keys and score >= config.minimum_path_score:
                    record = {
                        "receptor": receptor,
                        "tf": tf_name[target],
                        "path_score": score,
                        "path_length": depth,
                        "path_sign": new_sign,
                        "pathway": ">".join(new_path),
                        "path_sources": ";".join(
                            sorted(
                                {
                                    token
                                    for value in new_sources
                                    for token in str(value).split(";")
                                    if token
                                }
                            )
                        ),
                    }
                    previous = best_tf.get(target)
                    if previous is None or score > float(previous["path_score"]):
                        best_tf[target] = record
                    continue
                state = (target, new_path, new_log, new_sign, new_sources)
                previous_state = next_by_node.get(target)
                if previous_state is None or new_log > previous_state[2]:
                    next_by_node[target] = state
        frontier = sorted(
            next_by_node.values(), key=lambda state: state[2], reverse=True
        )[: config.maximum_frontier_nodes]
        if not frontier:
            break
    return sorted(
        best_tf.values(), key=lambda record: (-float(record["path_score"]), str(record["tf"]))
    )[: config.maximum_tfs_per_receptor]


def build_lr_tf_prior(
    edges: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    tf_table: pd.DataFrame,
    config: LRTFConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tf_column = "tf" if "tf" in tf_table.columns else tf_table.columns[0]
    tf_values = tf_table[tf_column].dropna().astype(str).drop_duplicates().tolist()
    tf_name = {value.upper(): value for value in tf_values}
    adjacency: dict[str, list[tuple[str, float, int, str]]] = {}
    for row in edges.itertuples(index=False):
        adjacency.setdefault(str(row.source_gene).upper(), []).append(
            (
                str(row.target_gene).upper(),
                float(row.edge_confidence),
                int(row.edge_sign),
                str(row.edge_sources),
            )
        )

    receptor_names = sorted(
        {
            component.upper()
            for receptor in lr_pairs["receptor"].astype(str)
            for component in _components(receptor)
        }
    )
    path_rows: list[dict[str, object]] = []
    for index, receptor in enumerate(receptor_names, start=1):
        path_rows.extend(receptor_paths(receptor, adjacency, set(tf_name), tf_name, config))
        if index % 50 == 0 or index == len(receptor_names):
            print(f"[paths] receptors {index}/{len(receptor_names)}")
    paths = pd.DataFrame(path_rows)
    if paths.empty:
        raise ValueError("No receptor-to-TF path passed the configured filters")

    paths_by_receptor = {
        receptor: frame for receptor, frame in paths.groupby("receptor", sort=False)
    }
    lr_rows: list[pd.DataFrame] = []
    for row in lr_pairs.itertuples(index=False):
        components = [value.upper() for value in _components(row.receptor)]
        available = [paths_by_receptor[value] for value in components if value in paths_by_receptor]
        if not available:
            continue
        combined = pd.concat(available, ignore_index=True).sort_values(
            "path_score", ascending=False
        ).drop_duplicates("tf")
        lr_confidence = float(row.confidence) if np.isfinite(float(row.confidence)) else 1.0
        combined["lr_id"] = str(row.lr_id)
        combined["score"] = (combined["path_score"] * lr_confidence).clip(0.0, 1.0)
        combined = combined.loc[combined["score"].ge(config.minimum_lr_tf_score)]
        lr_rows.append(combined.head(config.maximum_tfs_per_lr))
    prior = pd.concat(lr_rows, ignore_index=True) if lr_rows else pd.DataFrame()
    if prior.empty:
        raise ValueError("No LR-to-TF prior passed the configured filters")
    prior["evidence_level"] = "reference_path_only"
    prior = prior.sort_values(
        ["lr_id", "score", "tf"], ascending=[True, False, True]
    ).reset_index(drop=True)
    audit = pd.DataFrame(
        {
            "item": [
                "signaling_edges",
                "candidate_receptors",
                "receptors_with_tf_path",
                "candidate_lr_pairs",
                "lr_pairs_with_tf_path",
                "lr_tf_edges",
            ],
            "value": [
                len(edges),
                len(receptor_names),
                paths["receptor"].nunique(),
                len(lr_pairs),
                prior["lr_id"].nunique(),
                len(prior),
            ],
        }
    )
    return prior, paths, audit


def main(config: LRTFConfig = CONFIG) -> Path:
    signaling_path = Path(config.signaling_file)
    lr_path = Path(config.reference_dir) / config.lr_file_name
    tf_path = Path(config.reference_dir) / config.tf_file_name
    for path in (signaling_path, lr_path, tf_path):
        if not path.exists():
            raise FileNotFoundError(
                f"LR-to-TF input is missing: {path}. Run the preceding reference steps first."
            )
    print("[1/4] Reading and filtering mouse OmniPath signaling edges")
    raw = pd.read_csv(signaling_path, sep="\t", low_memory=False)
    edges = prepare_signaling_edges(raw, config)
    lr_pairs = pd.read_csv(lr_path)
    tf_table = pd.read_csv(tf_path)
    print("[2/4] Searching bounded receptor-to-TF paths")
    prior, paths, audit = build_lr_tf_prior(edges, lr_pairs, tf_table, config)
    print("[3/4] Writing LR-to-TF soft prior and path audit")
    output = Path(config.reference_dir)
    write_frame_atomic(prior, output / config.output_name)
    write_frame_atomic(paths, output / config.path_output_name)
    write_frame_atomic(audit, output / config.audit_output_name)
    update_summary_item(output / "reference_summary.csv", "lr_tf_edges", len(prior))
    write_json(
        output / config.manifest_name,
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": asdict(config),
            "audit": dict(zip(audit["item"], audit["value"])),
            "interpretation": "reference pathway support only; not measured activity or perturbation evidence",
        },
    )
    print(audit.to_string(index=False))
    print("[4/4] Complete")
    return output / config.output_name



