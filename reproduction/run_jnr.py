"""Run one fresh JumpNicheRewire fit using the manuscript's saved parameters.

This is deliberately separate from the saved model_fits and score tables.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
sys.path[:0] = [str(ROOT / "model_code"), str(ROOT / "model_code" / "src")]

import numpy as np
import pandas as pd

DATASETS = ("scmultisim", "p22", "case_mouse_kidney", "case_human_breast_cancer")


def assign(saved: dict, current) -> None:
    for field in fields(current):
        if field.name not in saved:
            continue
        value = saved[field.name]
        old = getattr(current, field.name)
        if is_dataclass(old) and isinstance(value, dict):
            assign(value, old)
        elif isinstance(old, Path) or field.name in {"input_dir", "output_dir"}:
            setattr(current, field.name, Path(value))
        else:
            setattr(current, field.name, value)


def saved_config(dataset: str) -> dict:
    if dataset == "scmultisim":
        path = ROOT / "model_code" / "experiment_parameters" / "参数配置_scMultiSim.json"
        return json.loads(path.read_text(encoding="utf-8"))["methods"]["JumpNicheRewire"]["full_model_config"]
    return json.loads((ROOT / "model_fits" / dataset / "model_config.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--rebuilt-input", action="store_true",
                        help="Use reproduction_runs/prepared rather than model_inputs")
    parser.add_argument("--check", action="store_true", help="Validate the input and saved configuration without fitting")
    args = parser.parse_args()
    from configs.common.presets import ModelParameters
    from jump_niche_rewire import JumpNicheRewirePipeline
    from jump_niche_rewire.convergence import ConvergenceError, summarize_convergence

    inputs = (ROOT / "reproduction_runs" / "prepared" if args.rebuilt_input else ROOT / "model_inputs") / args.dataset
    if not (inputs / "expression.csv").is_file():
        raise FileNotFoundError(inputs)
    out = ROOT / "reproduction_runs" / "fits" / args.dataset
    model = ModelParameters()
    assign(saved_config(args.dataset), model)
    model.data.input_dir = inputs
    model.output.output_dir = out
    model.runtime.checkpoint_data_signature = hashlib.sha256((inputs / "expression.csv").read_bytes()).hexdigest()
    for file_name in ("coordinates.csv", "metadata.csv", "tf_list.csv", "lr_pairs.csv",
                      "prior_tf_gene.csv", "prior_lr_tf.csv"):
        if not (inputs / file_name).is_file():
            raise FileNotFoundError(inputs / file_name)
    if args.check:
        print(f"PASS: {args.dataset} input and saved model configuration load; fit not started")
        return
    if out.exists():
        raise FileExistsError(f"Fresh fit output already exists; keep it for audit: {out}")
    out.mkdir(parents=True)
    (out / "effective_config.json").write_text(
        json.dumps(__import__("dataclasses").asdict(model), default=str, indent=2), encoding="utf-8")
    started = time.perf_counter()
    state = {"dataset": args.dataset, "state": "running", "input_dir": str(inputs),
             "config_source": str(ROOT / "model_code" / "experiment_parameters" / "参数配置_scMultiSim.json")
             if args.dataset == "scmultisim" else str(ROOT / "model_fits" / args.dataset / "model_config.json")}
    (out / "run_status.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    try:
        failure = None
        try:
            result = JumpNicheRewirePipeline(model).run()
        except ConvergenceError as error:
            result, failure = error.result, error
        summary = summarize_convergence(
            result.base, result.rewiring, model.runtime.minimum_target_convergence_rate,
            communication_rewiring_disabled=model.rewiring.disable_communication_rewiring)
        if failure is not None or not summary.get("convergence_passed"):
            raise RuntimeError(f"Fit did not pass convergence check: {summary}")
        beta = result.rewiring.beta_ccc
        score = np.max(np.abs(beta), axis=0) if args.dataset.startswith("case_") else np.mean(np.abs(beta), axis=0)
        frame = pd.DataFrame({"tf": np.repeat(result.base.tf_names, len(result.base.target_names)),
                              "target": np.tile(result.base.target_names, len(result.base.tf_names)),
                              "score": score.ravel()})
        frame.to_parquet(out / "global_grn.parquet", index=False)
        np.savez_compressed(out / "mechanism_coefficients.npz", gamma=result.rewiring.gamma,
                            active_lr_indices=result.rewiring.active_lr_indices,
                            tf_names=np.asarray(result.base.tf_names, dtype=str),
                            target_names=np.asarray(result.base.target_names, dtype=str),
                            regions=np.asarray(result.rewiring.regions, dtype=str))
        state.update(state="completed", convergence=summary, edges=len(frame),
                     global_grn_sha256=hashlib.sha256((out / "global_grn.parquet").read_bytes()).hexdigest())
    except BaseException as error:
        state.update(state="failed", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        state["runtime_seconds"] = time.perf_counter() - started
        (out / "run_status.json").write_text(json.dumps(state, default=str, indent=2), encoding="utf-8")
    print(f"PASS: {args.dataset}, {state['edges']} fitted edges, {out}")


if __name__ == "__main__":
    main()
