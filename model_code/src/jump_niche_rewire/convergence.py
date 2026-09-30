"""One fail-closed convergence contract used by the pipeline and benchmark."""
from __future__ import annotations
import numpy as np
import pandas as pd


def summarize_convergence(
    base,
    rewiring,
    minimum_rate=0.90,
    communication_rewiring_disabled=False,
):
    diagnostics = pd.concat([base.diagnostics, rewiring.diagnostics], ignore_index=True)
    applicable = (
        diagnostics.get("applicable", pd.Series(1, index=diagnostics.index))
        .fillna(1)
        .astype(bool)
    )
    applicable_diagnostics = diagnostics.loc[applicable]
    values = pd.to_numeric(
        applicable_diagnostics.get("converged"), errors="coerce"
    ).fillna(0)
    critical = diagnostics.loc[diagnostics["module"] != "joint_iteration"].copy()
    critical_values = pd.to_numeric(critical.get("converged"), errors="coerce").fillna(0)
    rates = critical.assign(value=critical_values).groupby("module")["value"].mean().to_dict()
    joint = diagnostics.loc[diagnostics["module"] == "joint_iteration"]
    joint_applicable = not bool(communication_rewiring_disabled)
    joint_ok = (
        bool(
            len(joint) == 1
            and pd.to_numeric(joint["converged"], errors="coerce").fillna(0).iloc[0] == 1
        )
        if joint_applicable
        else True
    )
    expected_modules = {"base_grn", "communication_rewiring"}
    modules_ok = expected_modules.issubset(rates) and all(np.isfinite(v) and v >= minimum_rate for v in rates.values())
    return dict(
        convergence_rate=float(critical_values.mean()) if len(critical_values) else 0.,
        all_diagnostic_rate=float(values.mean()) if len(values) else 0.,
        failed_diagnostics=int((critical_values < 1).sum()), diagnostics=len(critical),
        module_rates={str(k): float(v) for k, v in rates.items()},
        joint_applicable=joint_applicable,
        joint_converged=joint_ok if joint_applicable else None,
        convergence_passed=bool(modules_ok and joint_ok),
        convergence_policy=(
            "v3_per_module_joint_not_applicable_for_grn_only"
            if not joint_applicable
            else "v3_per_module_and_undamped_joint"
        ),
        minimum_convergence_rate=float(minimum_rate),
    )


class ConvergenceError(RuntimeError):
    def __init__(self, result, summary):
        self.result = result
        self.summary = summary
        super().__init__("Convergence gate failed; diagnostics and predictions were saved, but this is NOT an accepted run: " + str(summary))
