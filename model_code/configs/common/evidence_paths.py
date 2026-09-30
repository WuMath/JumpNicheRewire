
from __future__ import annotations
from grn2_project.paths import *
DATA_ROOT = PROCESSED_DATA_ROOT / "evidence_benchmark"
RAW_ROOT = RAW_DATA_ROOT / "evidence_benchmark"
WORKSPACE_ROOT = BENCHMARK_RESULT_ROOT / "evidence_benchmark"
MODEL_DATA = RAW_DATA_ROOT / "ModelData"
REAL_DATA = RAW_DATA_ROOT / "real_data"
SERGIO_DATASET_IDS = tuple(f"sergio_grn_100g_seed_{seed}" for seed in range(20260902, 20260907))
SCCIGEN_DATASET_IDS = ("sccigen_human_breast_example1",)
VISIUM_DATASET_IDS = ("real_pseudo__visium_ffpe_mouse_brain",)
MERFISH_DATASET_IDS = ("real_pseudo__merfish_mouse_brain_s2r3",)
