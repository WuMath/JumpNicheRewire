"""Recheck saved numerical stopping criteria without fitting or reading truth."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from grn2_project.paths import REPORT_RESULT_ROOT

from .config import ModelStageConfig
from .stage2_model import _solver_source_signature
from .state import stable_hash, write_json_atomic


from configs.self_sim_data.audit_settings import ConvergenceAuditConfig


def audit_convergence(config=None, model_config=None):
    config = config or ConvergenceAuditConfig()
    model = model_config or ModelStageConfig()
    output_dir = config.output_dir or (REPORT_RESULT_ROOT / "convergence_v2" / model.profile)
    registry = pd.read_csv(model.registry_path)
    if set(registry["benchmark_profile"].astype(str)) != {model.profile}:
        raise ValueError(f"Registry {model.registry_path} is not isolated to profile {model.profile!r}")
    registry = registry[registry["split"].isin(config.allowed_splits)]
    source_signature = _solver_source_signature()
    rows = []
    for _, dataset in registry.iterrows():
        for parameter in model.parameter_grid():
            root = model.profile_benchmark_root / "02_runs/jumpniche" / parameter.parameter_id / dataset["dataset_id"]
            row = dict(dataset_id=dataset["dataset_id"], split=dataset["split"], parameter_id=parameter.parameter_id,
                       state="missing", accepted=False, run_dir=str(root))
            try:
                status = json.loads((root / "run_status.json").read_text(encoding="utf-8"))
                effective = json.loads((root / "effective_config.json").read_text(encoding="utf-8"))
                tables = root / effective["output"]["parquet_dir_name"]
                diagnostics = pd.read_parquet(tables / "convergence.parquet")
                history = pd.read_parquet(tables / "convergence_history.parquet")
                base = diagnostics[diagnostics.module == "base_grn"]
                rewire = diagnostics[diagnostics.module == "communication_rewiring"]
                joint = diagnostics[diagnostics.module == "joint_iteration"]
                base_ok = (base.primal_residual <= base.primal_tolerance) & (base.dual_residual <= base.dual_tolerance) & (base.linear_solve_ok == 1)
                fista_tolerance = effective["rewiring"]["fista_tolerance"]
                rewire_ok = (rewire.kkt_residual <= rewire.kkt_tolerance) & (rewire.relative_change <= fista_tolerance)
                if len(joint) != 1 or base.empty or rewire.empty:
                    raise ValueError("Missing or duplicate convergence module summaries")
                j = joint.iloc[0]
                joint_tolerance = effective["rewiring"]["iteration_tolerance"]
                joint_ok = bool(np.isfinite(j.fixed_point_residual) and j.fixed_point_residual <= joint_tolerance and j.beta_relative_change <= joint_tolerance and j.converged == 1)
                labels_consistent = bool(np.array_equal(base_ok.to_numpy(), base.converged.to_numpy() == 1) and np.array_equal(rewire_ok.to_numpy(), rewire.converged.to_numpy() == 1))
                signature_ok = status.get("solver_source_signature") == source_signature and status.get("parameter_signature") == stable_hash(parameter)
                complete_path = root / ".complete.json"
                marker_ok = complete_path.exists() and json.loads(complete_path.read_text(encoding="utf-8")).get("signature") == status.get("signature")
                accepted = bool(status.get("state") == "completed" and status.get("convergence_passed") is True
                                and base_ok.mean() >= model.minimum_convergence_rate and rewire_ok.mean() >= model.minimum_convergence_rate
                                and joint_ok and labels_consistent and signature_ok and marker_ok and len(history) > 0)
                row.update(state=status["state"], accepted=accepted, base_rate=float(base_ok.mean()), rewiring_rate=float(rewire_ok.mean()),
                           joint_converged=joint_ok, joint_iterations=int(j.iterations), joint_residual=float(j.fixed_point_residual),
                           joint_tolerance=joint_tolerance, beta_relative_change=float(j.beta_relative_change),
                           max_admm_primal_ratio=float(base.primal_ratio.max()), max_admm_dual_ratio=float(base.dual_ratio.max()),
                           max_fista_kkt_ratio=float(rewire.kkt_ratio.max()), base_max_iterations=int(base.iterations.max()),
                           final_fista_max_iterations=int(rewire.iterations.max()), history_rows=len(history),
                           labels_consistent=labels_consistent, signature_matches=signature_ok,
                           runtime_seconds=float(status["runtime_seconds"]))
            except (OSError, ValueError, KeyError, AttributeError) as error:
                row["audit_error"] = str(error)
            rows.append(row)
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "convergence_acceptance.csv", index=False, encoding="utf-8-sig")
    passed = int(frame.accepted.sum()) if len(frame) else 0
    summary = dict(benchmark_profile=model.profile, expected_runs=len(frame), accepted_runs=passed, all_passed=bool(len(frame) and passed == len(frame)),
                   source_signature=source_signature, test_truth_read=False,
                   runs=json.loads(frame.to_json(orient="records")))
    write_json_atomic(output_dir / "convergence_acceptance.json", summary)
    lines = ["# 收敛修复验收", "", f"通过：{passed}/{len(frame)}。这里只验收优化收敛，不代表预测性能或生物学有效性。", "",
             "阈值：每个内层模块的目标通过率至少90%；外层未阻尼固定点残差及系数变化≤1e-4。以每次effective_config.json记录为准。",
             "未读取test真值；未启动freeze/test；原data/result/benchmark结果保留。", "",
             "| 数据集 | 参数 | 基础GRN | 重连 | 外层迭代 | 未阻尼残差 | 验收 |", "|---|---|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['dataset_id']} | {row['parameter_id']} | {row.get('base_rate', 0):.1%} | {row.get('rewiring_rate', 0):.1%} | {row.get('joint_iterations', '-')} | {row.get('joint_residual', float('nan')):.3g} | {'PASS' if row['accepted'] else 'FAIL'} |")
    lines.extend(["", "完整逐次轨迹：每次运行下的parquet/convergence_history.parquet；Excel仅预览前1000行，终点诊断保留完整。",
                  "convergence_acceptance.csv/json包含逐次验收、残差/阈值比、代码签名和失败原因。",
                  "其他数据规模仍需单独验收；不能从pilot通过推断所有真实数据都会收敛。"])
    (output_dir / "convergence_acceptance.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[convergence audit] {passed}/{len(frame)} accepted -> {output_dir}")
    if not summary["all_passed"]:
        raise RuntimeError("Convergence audit failed; inspect convergence_acceptance.csv, do not freeze/test")
    return config.output_dir
