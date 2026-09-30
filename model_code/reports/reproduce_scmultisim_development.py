"""Read-only reproduction of the published scMultiSim one-dataset GRN table.

Run from the repository root with the jumpniche Python interpreter.  This
script never fits a model, changes the active profile, or writes a result.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import psutil

from independent_baselines.comparison import metrics


DATASET = "published_tree1_500_cells110_genes_sigma0.1_1"
STUDY = ROOT / "data/result/optimization_20260921"
DATA = ROOT / "data/processed/publication_20260920" / DATASET
FINAL = STUDY / "single_dataset_jnr_target032_hour_20260924/final_comparison"
PERFORMANCE = (
    "AP", "AUROC", "Early_precision", "Precision_at_10", "Recall_at_10",
    "F1_at_10", "Precision_at_50", "Recall_at_50", "F1_at_50",
    "Precision_at_100", "Recall_at_100", "F1_at_100",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prediction_source(method: str, candidate: str, historical: pd.DataFrame) -> tuple[Path, Path, str]:
    if method == "JumpNicheRewire":
        folder = STUDY / "single_dataset_jnr_positive_fusion_followup_20260924/02_runs" / candidate
        return folder / "global_grn.parquet", folder / "run_status.json", "prediction_sha256"
    if method == "CellOracle_Net_core":
        folder = STUDY / "single_dataset_extension/CellOracle" / candidate / DATASET
        return folder / "global_grn.parquet", folder / "run_status.json", "global_sha256"
    if method in {"CeSpGRN", "RegDiffusion"}:
        folder = STUDY / "single_dataset_extension" / method / candidate / DATASET
        return folder / "global_grn.parquet", folder / "run_status.json", "global_sha256"
    matching = historical.loc[
        historical.dataset_id.eq(DATASET) & historical.method.eq(method)
        & historical.candidate.eq(candidate) & historical.role.eq("candidate")
        & historical.state.eq("accepted")
    ]
    if len(matching) != 1 or not candidate.startswith(("r01__", "r02__")):
        raise ValueError(f"Historical candidate not uniquely accepted: {method}/{candidate}")
    origin = matching.iloc[0]["origin"]
    expected_prefix = "r01__" if origin == "round_01" else "r02__"
    if origin not in {"round_01", "round_02"} or not candidate.startswith(expected_prefix):
        raise ValueError(f"Historical source mismatch: {method}/{candidate}")
    native_name = candidate.removeprefix(expected_prefix)
    folder = STUDY / origin / "02_runs" / method / native_name / DATASET
    return folder / "global_grn.parquet", folder / "run_status.json", "prediction_sha256"


def main() -> None:
    available = psutil.virtual_memory().available
    if available < int(1.5 * 1024**3):
        raise MemoryError(f"Available RAM {available / 1024**3:.2f} GiB is below the 1.5 GiB guard")
    manifest = json.loads((DATA / "evidence_manifest.json").read_text(encoding="utf8"))
    for relative, expected in manifest["files"].items():
        path = DATA / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Published-data file missing or changed: {relative}")
    truth = pd.read_csv(DATA / "truth/truth_grn.csv", dtype={"tf": str, "target": str})
    if (len(truth) != 624 or int(truth.label.sum()) != 130
            or truth.duplicated(["tf", "target"]).any()):
        raise ValueError("The fixed 624-edge/130-positive truth universe changed")
    table = pd.read_csv(FINAL / "nine_method_metrics.csv")
    audit = json.loads((FINAL / "audit.json").read_text(encoding="utf8"))
    historical = pd.read_csv(STUDY / "round_02/03_development/run_audit.csv")
    if (len(table) != 9 or table.method.nunique() != 9
            or set(table.columns) != {"method", "candidate", *PERFORMANCE}
            or audit["selected"] != "prior__l1_0075"
            or audit["independent_confirmation"] is not False):
        raise ValueError("Final table/audit no longer matches the frozen development comparison")
    print(f"Dataset: {DATASET}; cells=500; candidates=624; positives=130")
    print("Read-only cached-prediction evaluation; same dataset was used for selection and scoring.")
    for row in table.itertuples(index=False):
        prediction_path, status_path, hash_key = prediction_source(row.method, row.candidate, historical)
        status = json.loads(status_path.read_text(encoding="utf8"))
        result = status.get("result", status)  # round_02 wraps the worker receipt
        if (status.get("state") != "completed" or result.get("state") != "completed"
                or result.get("dataset_id") != DATASET
                or result.get("method") != row.method
                or result.get("candidate") != row.candidate.removeprefix("r01__").removeprefix("r02__")
                or result.get("input_signature", manifest["signature"]) != manifest["signature"]
                or result.get(hash_key) != sha256(prediction_path)):
            raise ValueError(f"Prediction/status/hash mismatch: {row.method}")
        if (status.get("exit_code", status.get("worker_exit_code", 0)) != 0
                or result.get("truth_read", False)
                or result.get("source", {}).get("inference_truth_files_read", False)):
            raise ValueError(f"Worker exit or truth-use receipt invalid: {row.method}")
        input_hashes = result.get("input_hashes", result.get("source", {}).get("input_sha256", {}))
        if row.method == "CellOracle_Net_core":
            preflight = json.loads((status_path.parent / "preflight.json").read_text(encoding="utf8"))
            if preflight.get("inference_truth_files_read") is not False:
                raise ValueError("CellOracle preflight truth-use receipt invalid")
            input_hashes = preflight["input_sha256"]
        for relative, expected in input_hashes.items():
            if manifest["files"].get(relative) != expected:
                raise ValueError(f"Input hash mismatch: {row.method}/{relative}")
        prediction = pd.read_parquet(prediction_path)
        if (not {"tf", "target", "score"}.issubset(prediction)
                or prediction.duplicated(["tf", "target"]).any()
                or not np.isfinite(prediction.score).all()):
            raise ValueError(f"Invalid prediction file: {row.method}")
        joined = truth.merge(prediction[["tf", "target", "score"]],
                             on=["tf", "target"], how="left", validate="one_to_one")
        coverage = int(joined.score.notna().sum())
        joined["score"] = joined.score.fillna(0.0)
        observed = metrics(joined.label, joined.score, (10, 50, 100))
        for name in PERFORMANCE:
            if not np.isclose(observed[name], getattr(row, name), atol=1e-12, rtol=0):
                raise ValueError(f"Metric mismatch: {row.method}/{name}")
        print(f"{row.method:20s} AP={observed['AP']:.6f} AUROC={observed['AUROC']:.6f} "
              f"EP={observed['Early_precision']:.6f} covered={coverage}/624")
    print("PASS: all nine cached predictions reproduce all twelve reported metrics.")


if __name__ == "__main__":
    main()
