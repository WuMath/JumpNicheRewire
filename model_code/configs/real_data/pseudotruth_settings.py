from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from grn2_project.paths import *
from configs.common.evidence_paths import *

@dataclass(frozen=True)
class RealDatasetSpec:
    dataset_id: str
    input_dir: Path
    species: str = "mouse"
    tissue: str = "brain"


@dataclass
class RealPseudoTruthConfig:
    datasets: tuple[RealDatasetSpec, ...] = (
        RealDatasetSpec("visium_ffpe_mouse_brain", PROCESSED_DATA_ROOT / "full/visium_ffpe_mouse_brain"),
        RealDatasetSpec("merfish_mouse_brain_s2r3", PROCESSED_DATA_ROOT / "full/merfish_mouse_brain_s2r3"),
    )
    output_root: Path = DATA_ROOT / "02_prepared" / "real_pseudotruth"
    tf_reference: Path = MODEL_DATA / "Mouse-TF-Target.tsv"
    lr_reference: Path = PROCESSED_DATA_ROOT / "references/lr_pairs_detailed.csv"
    minimum_detection_fraction: float = 0.02
    minimum_curation_effort: int = 2
    minimum_independent_sources: int = 2
    maximum_target_universe: int = 3000
    decoys_per_positive: int = 5
    random_seed: int = 20260902
    # Curated TF-target evidence is withheld from model priors to avoid circular evaluation.
    withhold_curated_tf_prior: bool = True

