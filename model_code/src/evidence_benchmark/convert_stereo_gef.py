"""Convert official GSE274447 cell-bin GEF files to one audited AnnData file.

This is format conversion only.  It does not invent cell types or biological
replicates.  The user must fill the sample map from the GEO sample metadata.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import tarfile

import numpy as np
import pandas as pd
from scipy import sparse

from .common import sha256_file, write_json
from .config import STEREO_CONVERSION_CONFIG, StereoGEFConversionConfig


def _safe_extract(config: StereoGEFConversionConfig) -> None:
    if not config.raw_archive.exists():
        raise FileNotFoundError(config.raw_archive)
    config.extracted_dir.mkdir(parents=True, exist_ok=True)
    root = config.extracted_dir.resolve()
    with tarfile.open(config.raw_archive, mode="r") as archive:
        for member in archive.getmembers():
            destination = (root / member.name).resolve()
            if os.path.commonpath([str(root), str(destination)]) != str(root):
                raise ValueError(f"Unsafe archive member: {member.name}")
            if destination.exists() and not config.overwrite_extracted_files:
                continue
            # The destination has already passed the common-path traversal gate.
            # Avoid Python 3.12-only ``filter=`` so this also runs in Stereopy's
            # officially recommended Python 3.8 environment.
            archive.extract(member, path=root)


def _sample_map(config: StereoGEFConversionConfig, files: list[Path]) -> pd.DataFrame:
    if not config.sample_map_file.exists():
        template = pd.DataFrame({
            "gef_filename": [path.name for path in files],
            "section_id": [path.name.split(".")[0] for path in files],
            "biological_sample_id": "",
            "condition": "in_vivo_CRISPR",
        })
        config.sample_map_file.parent.mkdir(parents=True, exist_ok=True)
        template.to_csv(config.sample_map_file, index=False)
        raise ValueError(
            f"Created {config.sample_map_file}. Fill biological_sample_id from the GEO sample metadata, "
            "then rerun. Different sections from the same mouse must have the same biological_sample_id."
        )
    frame = pd.read_csv(config.sample_map_file, dtype=str).fillna("")
    required = {"gef_filename", "section_id", "biological_sample_id", "condition"}
    if not required.issubset(frame.columns):
        raise ValueError(f"sample map needs columns {sorted(required)}")
    expected = {path.name for path in files}
    observed = set(frame.gef_filename)
    if expected != observed:
        raise ValueError(
            f"sample map/file mismatch; missing={sorted(expected-observed)}, extra={sorted(observed-expected)}"
        )
    if frame.biological_sample_id.str.strip().eq("").any():
        raise ValueError("Every GEF needs a non-empty biological_sample_id; do not treat sections as mice.")
    if frame.gef_filename.duplicated().any() or frame.section_id.duplicated().any():
        raise ValueError("gef_filename and section_id must be unique")
    return frame.set_index("gef_filename")


def _assign_guides(adata, guide_features: tuple[str, ...], threshold: float) -> tuple[np.ndarray, list[str]]:
    # GEO GEF uses e.g. ``sgrna_msafe`` while the original config uses
    # ``sgrna-msafe``. Match punctuation-insensitively, but reject ambiguous
    # gene names instead of silently choosing one.
    normalize = lambda name: re.sub(r"[^a-z0-9]", "", str(name).lower())
    expected = {normalize(guide) for guide in guide_features}
    lookup: dict[str, int] = {}
    for index, gene in enumerate(adata.var_names.astype(str)):
        key = normalize(gene)
        if key not in expected:
            continue
        if key in lookup:
            raise ValueError(f"Ambiguous guide/gene identifier after normalization: {gene}")
        lookup[key] = index
    matched = [(guide, lookup[normalize(guide)]) for guide in guide_features if normalize(guide) in lookup]
    if not matched:
        raise ValueError(
            "No expected sgRNA features were found in the GEF. Inspect gene names and edit "
            "StereoGEFConversionConfig.guide_features."
        )
    indices = [index for _, index in matched]
    matrix = adata.X[:, indices]
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix)
    positive = matrix > threshold
    assignments = np.full(adata.n_obs, "Neg", dtype=object)
    one = positive.sum(axis=1) == 1
    assignments[one] = np.asarray([name for name, _ in matched], dtype=object)[positive[one].argmax(axis=1)]
    assignments[positive.sum(axis=1) > 1] = "Multiple"
    return assignments, [str(adata.var_names[index]) for index in indices]


def run_conversion(config: StereoGEFConversionConfig = STEREO_CONVERSION_CONFIG) -> Path:
    if config.extract_archive:
        _safe_extract(config)
    files = sorted(config.extracted_dir.glob(config.gef_glob))
    if not files:
        raise FileNotFoundError(f"No GEF files matching {config.extracted_dir / config.gef_glob}")
    mapping = _sample_map(config, files)
    try:
        import anndata as ad
        import stereo as st
    except ImportError as error:
        raise ImportError(
            "Use the separate Stereopy environment from evidence_benchmark/environment_stereopy.yml."
        ) from error
    objects = []
    audit_rows = []
    x_offset = 0.0
    for path in files:
        row = mapping.loc[path.name]
        print(f"[GEF] {path.name}")
        stereo_data = st.io.read_gef(
            file_path=str(path), bin_type=config.bin_type, is_sparse=True,
            gene_name_index=config.gene_name_index,
        )
        adata = st.io.stereo_to_anndata(stereo_data, flavor="seurat")
        if "spatial" not in adata.obsm:
            adata.obsm["spatial"] = np.asarray(stereo_data.position)[:, :2]
        coordinates = np.asarray(adata.obsm["spatial"], dtype=float)[:, :2]
        coordinates[:, 0] = coordinates[:, 0] - coordinates[:, 0].min() + x_offset
        x_offset = coordinates[:, 0].max() + config.inter_section_gap
        adata.obsm["spatial"] = coordinates
        guides, guide_variables = _assign_guides(
            adata, config.guide_features, config.guide_detection_threshold
        )
        section = str(row.section_id)
        adata.obs_names = pd.Index([f"{section}__{cell}" for cell in adata.obs_names.astype(str)])
        adata.obs["perturbation"] = guides
        adata.obs["sample_id"] = str(row.biological_sample_id)
        adata.obs["section_id"] = section
        adata.obs["condition"] = str(row.condition)
        if "cell_type" not in adata.obs:
            adata.obs["cell_type"] = "NA"
        keep = ~adata.var_names.astype(str).isin(guide_variables)
        adata = adata[:, keep].copy()
        objects.append(adata)
        values, counts = np.unique(guides, return_counts=True)
        guide_counts = dict(zip(values.tolist(), counts.tolist()))
        audit_rows.append({
            "gef_filename": path.name, "section_id": section,
            "biological_sample_id": str(row.biological_sample_id),
            "cells": adata.n_obs, "genes_after_guide_removal": adata.n_vars,
            "negative_cells": int(guide_counts.get("Neg", 0)),
            "multiple_guide_cells": int(guide_counts.get("Multiple", 0)),
            "assigned_single_guide_cells": int(adata.n_obs - guide_counts.get("Neg", 0) - guide_counts.get("Multiple", 0)),
            "source_sha256": sha256_file(path),
        })
    combined = ad.concat(
        objects, join="inner", merge="same", index_unique=None,
        label="section_batch", keys=[str(mapping.loc[path.name, "section_id"]) for path in files],
    )
    if combined.obs_names.duplicated().any():
        raise ValueError("Cell IDs are not unique after section prefixing")
    if "spatial" not in combined.obsm or np.asarray(combined.obsm["spatial"]).shape[1] < 2:
        raise ValueError("Real XY coordinates were lost during conversion")
    config.standardized_h5ad.parent.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(config.standardized_h5ad, compression=config.output_compression)
    audit = pd.DataFrame(audit_rows)
    audit_path = config.standardized_h5ad.with_name("GSE274447_conversion_audit.csv")
    audit.to_csv(audit_path, index=False)
    write_json(config.standardized_h5ad.with_name("GSE274447_conversion_manifest.json"), {
        "source": "GSE274447 official adjusted cell-bin GEF",
        "output": str(config.standardized_h5ad.resolve()),
        "cells": combined.n_obs, "genes": combined.n_vars,
        "biological_samples": combined.obs["sample_id"].nunique(),
        "sections": combined.obs["section_id"].nunique(),
        "coordinate_policy": "real within-section XY; sections translated apart without rotation/scaling",
        "blinding_note": "Perturbation cells remain in H5AD but are excluded from model input by prepare_perturbation.py.",
        "output_sha256": sha256_file(config.standardized_h5ad),
    })
    print(f"[standardized H5AD] {config.standardized_h5ad}")
    return config.standardized_h5ad



