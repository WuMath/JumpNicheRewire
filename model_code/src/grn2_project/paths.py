"""Canonical, location-independent paths for the organized repository."""

from pathlib import Path
from configs.paths import RECONSTRUCT_ROOT, PATHS, OLD_RAW


PROJECT_ROOT = RECONSTRUCT_ROOT
DATA_ROOT = PATHS.data_root
# Raw inputs are read-only. Conversion/extraction destinations are explicitly
# redirected to local paths by the new preprocessing profiles.
RAW_DATA_ROOT = OLD_RAW
PROCESSED_DATA_ROOT = DATA_ROOT / "processed"
RESULT_DATA_ROOT = DATA_ROOT / "result"
ARCHIVE_ROOT = PROJECT_ROOT / "archive"

MODEL_RESULT_ROOT = RESULT_DATA_ROOT / "model"
BENCHMARK_RESULT_ROOT = RESULT_DATA_ROOT / "benchmark"
EVALUATION_RESULT_ROOT = RESULT_DATA_ROOT / "evaluation"
REPORT_RESULT_ROOT = RESULT_DATA_ROOT / "reports"
