"""Typed data containers shared by the four model modules."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class InputBundle:
    expression: pd.DataFrame
    coordinates: pd.DataFrame
    metadata: pd.DataFrame
    tf_names: list[str]
    lr_pairs: pd.DataFrame
    tf_gene_prior: pd.DataFrame
    lr_tf_prior: pd.DataFrame
    ligand_activity: pd.DataFrame | None = None
    receptor_activity: pd.DataFrame | None = None
    tf_activity: pd.DataFrame | None = None
    perturbations: pd.DataFrame | None = None
    audit: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class SpatialGraph:
    source: np.ndarray
    target: np.ndarray
    distance: np.ndarray
    distance_weight: np.ndarray
    state_weight: np.ndarray
    weight: np.ndarray
    bandwidth: np.ndarray
    state_embedding: np.ndarray


@dataclass
class SpatialResult:
    normalized_expression: pd.DataFrame
    qc_genes: list[str]
    spatial_genes: list[str]
    spatial_statistics: pd.DataFrame
    graph: SpatialGraph
    candidate_tfs: list[str]
    candidate_targets: list[str]
    communication_genes: list[str]


@dataclass
class PriorResult:
    confidence: pd.DataFrame
    penalty: pd.DataFrame
    lr_tf_penalty: pd.DataFrame


@dataclass
class BaseGRNResult:
    beta: np.ndarray
    tf_names: list[str]
    target_names: list[str]
    tf_matrix: np.ndarray
    target_matrix: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    covariate_prediction: np.ndarray
    base_prediction: np.ndarray
    candidate_mask: np.ndarray
    diagnostics: pd.DataFrame
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    unit_type: str = "cell"


@dataclass
class CommunicationResult:
    exposures: pd.DataFrame
    sender_contributions: pd.DataFrame
    cci_edges: pd.DataFrame
    ligand_effective: pd.DataFrame
    receptor_effective: pd.DataFrame


@dataclass
class RewiringResult:
    gamma: np.ndarray
    regions: list[str]
    region_index: np.ndarray
    beta_ccc: np.ndarray
    grn_edges: pd.DataFrame
    lr_gene: pd.DataFrame
    receptor_paths: pd.DataFrame
    regulon_activity: pd.DataFrame
    causal_edges: pd.DataFrame
    diagnostics: pd.DataFrame
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    # gamma's second axis contains only these columns from the complete LR
    # exposure table.  Older checkpoints without this field are interpreted as
    # full-axis gamma by the output helpers.
    active_lr_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))


@dataclass
class PipelineResult:
    spatial: SpatialResult
    prior: PriorResult
    base: BaseGRNResult
    communication: CommunicationResult
    rewiring: RewiringResult
    workbook_path: str
