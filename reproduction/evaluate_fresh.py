"""Evaluate fresh JNR fits on the exact paper candidate sets and labels."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "model_code"), str(ROOT / "model_code" / "src")]

import numpy as np
import pandas as pd
from independent_baselines.comparison import metrics


def evaluate(dataset: str) -> dict:
    prediction = ROOT / "reproduction_runs" / "fits" / dataset / "global_grn.parquet"
    frame = pd.read_parquet(prediction)[["tf", "target", "score"]]
    if frame.duplicated(["tf", "target"]).any() or not np.isfinite(frame.score).all():
        raise ValueError(f"Invalid fresh scores: {prediction}")
    if dataset == "scmultisim":
        truth = pd.read_csv(ROOT / "model_inputs/scmultisim/truth/truth_grn.csv")
    else:
        truth = pd.read_csv(ROOT / "results/p22/support_edges.csv")[["tf", "target", "label"]]
    joined = truth.merge(frame, on=["tf", "target"], how="left", validate="one_to_one")
    if joined.score.isna().any():
        raise ValueError(f"Fresh fit lacks {int(joined.score.isna().sum())} evaluated edges")
    k = math.ceil(len(joined) * 0.1)
    result = metrics(joined.label.to_numpy(int), joined.score.to_numpy(float), (),
                     decision_k=k, f_betas=(.1,))
    row = {"dataset": dataset, "method": "JumpNicheRewire_fresh",
           "candidates": len(joined), "positives": int(joined.label.sum()),
           "decision_k": k, **result}
    out = ROOT / "reproduction_runs" / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    joined.to_csv(out / f"{dataset}_fresh_edges.csv", index=False)
    (out / f"{dataset}_fresh_metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("scmultisim", "p22"))
    args = parser.parse_args()
    print(json.dumps(evaluate(args.dataset), indent=2))
