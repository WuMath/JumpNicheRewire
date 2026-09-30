from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from grn2_project.paths import *

@dataclass
class SimulationConfig:
    """Edit these fields and run ``generate_simulation.py`` without arguments."""

    output_dir: Path = PROCESSED_DATA_ROOT / "examples" / "simulated"
    truth_dir_name: str = "truth"
    random_seed: int = 20260822
    region_count: int = 3
    cells_per_region: int = 35
    tf_count: int = 5
    target_count: int = 10
    lr_count: int = 4
    noise_gene_count: int = 12
    coordinate_spread: float = 0.65
    region_separation: float = 5.0
    communication_neighbors: int = 12
    count_dispersion: float = 12.0
    base_edge_density: float = 0.25
    rewiring_edge_density: float = 0.12
    perturbation_noise: float = 0.05
    include_activity_evidence: bool = False
    include_perturbation_evidence: bool = False
    causal_confounding: str = "none"  # none, measured, or unmeasured
    confounder_strength: float = 0.60


SIMULATION_CONFIG = SimulationConfig()

