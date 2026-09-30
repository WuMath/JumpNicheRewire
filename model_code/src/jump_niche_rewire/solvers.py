"""Certified inner solvers; objective definitions and penalties are unchanged.

ADMM uses K=[I; D], s=rho*K.T*(z-z_old) (Boyd et al., 2011, 3.3).
Group FISTA checks the original-coordinate KKT residual, not step size alone.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg, splu


def _norm(x: np.ndarray) -> float:
    return float(np.sqrt(np.sum(np.asarray(x) ** 2)))


def graph_fused_admm(x, y, source, target, edge_weight, l1_weight, config):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not all(np.isfinite(a).all() for a in (x, y, edge_weight, l1_weight)):
        raise ValueError("Nonfinite input in graph-fused ADMM")
    if config.admm_rho <= 0 or config.admm_max_iterations < 1:
        raise ValueError("ADMM rho and iteration budget must be positive")
    if np.any(np.asarray(l1_weight) < 0) or config.fusion_penalty < 0 or np.any(np.asarray(edge_weight) < 0):
        raise ValueError("ADMM penalties must be nonnegative")
    if min(config.admm_absolute_tolerance, config.admm_relative_tolerance, config.cg_tolerance, config.admm_rho_update_interval) <= 0:
        raise ValueError("ADMM tolerances and rho update interval must be positive")
    n, k = x.shape
    if n == 0 or k == 0:
        raise ValueError("ADMM requires at least one cell and TF")
    source, target, edge_weight = map(np.asarray, (source, target, edge_weight))
    keep = (edge_weight > 0) & (source != target)
    source, target, edge_weight = source[keep], target[keep], edge_weight[keep]
    m = len(source)
    incidence = sparse.coo_matrix((np.r_[np.ones(m), -np.ones(m)],
                                  (np.r_[np.arange(m), np.arange(m)], np.r_[source, target])),
                                 shape=(m, n)).tocsr()
    laplacian = (incidence.T @ incidence).tocsr()
    degree = laplacian.diagonal()
    rho = float(config.admm_rho)
    beta = np.zeros((n, k))
    z, u = beta.copy(), beta.copy()
    d, v = np.zeros((m, k)), np.zeros((m, k))
    xy = x * y[:, None]
    direct = n * k <= config.admm_direct_max_variables
    data_hessian = sparse.block_diag([np.outer(row, row) for row in x], format="csc") if direct else None
    constraint_hessian = sparse.kron(sparse.eye(n, format="csc") + laplacian, sparse.eye(k), format="csc") if direct else None
    factor, factor_rho = None, None
    history = []
    converged = False
    for iteration in range(1, config.admm_max_iterations + 1):
        def matvec(vector):
            b = vector.reshape(n, k)
            return (x * np.sum(x * b, axis=1)[:, None] + rho * (b + laplacian @ b)).ravel()

        rhs = (xy + rho * (z - u + incidence.T @ (d - v))).ravel()
        cg_info = 0
        if direct:
            if factor_rho != rho:
                factor = splu(data_hessian + rho * constraint_hessian)
                factor_rho = rho
            beta = factor.solve(rhs).reshape(n, k)
        else:
            operator = LinearOperator((n * k, n * k), matvec=matvec, dtype=float)
            diagonal = (x * x + rho * (1 + degree[:, None])).ravel()
            preconditioner = LinearOperator(operator.shape, matvec=lambda a: a / diagonal, dtype=float)
            value, cg_info = cg(operator, rhs, x0=beta.ravel(), M=preconditioner,
                                rtol=config.cg_tolerance, atol=0, maxiter=config.cg_max_iterations)
            beta = value.reshape(n, k)
        if not np.isfinite(beta).all() or cg_info < 0:
            raise FloatingPointError("ADMM linear solve failed")
        linear_residual = _norm(matvec(beta.ravel()) - rhs) / max(_norm(rhs), 1e-12)
        linear_ok = cg_info == 0 and linear_residual <= max(10 * config.cg_tolerance, 1e-12)
        old_z, old_d = z, d
        b_plus_u = beta + u
        z = np.sign(b_plus_u) * np.maximum(np.abs(b_plus_u) - l1_weight[None, :] / rho, 0)
        db = incidence @ beta
        d_plus_v = db + v
        d = np.sign(d_plus_v) * np.maximum(np.abs(d_plus_v) - config.fusion_penalty * edge_weight[:, None] / rho, 0)
        u += beta - z
        v += db - d
        primal = np.sqrt(_norm(beta - z)**2 + _norm(db - d)**2)
        dual = rho * _norm(z - old_z + incidence.T @ (d - old_d))
        eps_primal = np.sqrt((n + m) * k) * config.admm_absolute_tolerance + config.admm_relative_tolerance * max(
            np.sqrt(_norm(beta)**2 + _norm(db)**2), np.sqrt(_norm(z)**2 + _norm(d)**2))
        eps_dual = np.sqrt(n * k) * config.admm_absolute_tolerance + config.admm_relative_tolerance * _norm(rho * (u + incidence.T @ v))
        objective = .5 * np.sum((np.sum(x * beta, axis=1) - y)**2) + np.sum(l1_weight[None, :] * np.abs(beta)) + config.fusion_penalty * np.sum(edge_weight[:, None] * np.abs(db))
        converged = bool(primal <= eps_primal and dual <= eps_dual and linear_ok)
        history.append(dict(iteration=iteration, objective=float(objective), primal_residual=float(primal),
                            dual_residual=float(dual), primal_tolerance=float(eps_primal), dual_tolerance=float(eps_dual),
                            primal_ratio=float(primal / eps_primal), dual_ratio=float(dual / eps_dual),
                            rho=rho, linear_residual=linear_residual, linear_solve_ok=int(linear_ok),
                            converged=int(converged)))
        if converged:
            break
        # Rescale BOTH scaled dual variables when rho changes; adaptation is finite.
        if config.admm_adaptive_rho and iteration % config.admm_rho_update_interval == 0 and iteration <= config.admm_rho_adapt_until:
            pr, dr = primal / max(eps_primal, 1e-15), dual / max(eps_dual, 1e-15)
            next_rho = rho
            if pr > config.admm_residual_balance * dr:
                next_rho = min(rho * 2, config.admm_rho_max)
            elif dr > config.admm_residual_balance * pr:
                next_rho = max(rho / 2, config.admm_rho_min)
            u *= rho / next_rho
            v *= rho / next_rho
            rho = next_rho
    summary = dict(history[-1])
    summary.update(iterations=iteration, stop_reason="converged" if converged else "iteration_limit",
                   linear_solver="sparse_direct" if direct else "preconditioned_cg", history=history)
    # Do not threshold certified numerical coefficients after checking residuals.
    return beta, summary


def group_fista(design, response, penalties, ridge, max_iterations, tolerance, initial=None):
    x, y, penalties = np.asarray(design, float), np.asarray(response, float), np.asarray(penalties, float)
    n, p = x.shape
    if n == 0 or y.shape != (n,):
        raise ValueError("FISTA requires matching nonempty samples")
    groups = len(penalties)
    if not all(np.isfinite(a).all() for a in (x, y, penalties)) or not np.isfinite(ridge):
        raise ValueError("Nonfinite group-FISTA inputs")
    if ridge < 0 or np.any(penalties < 0) or tolerance <= 0 or max_iterations < 1:
        raise ValueError("Invalid group-FISTA penalties, tolerance or budget")
    if p == 0:
        return np.zeros(p), dict(iterations=0, converged=1, objective=float(.5 * np.mean(y*y)),
                                 kkt_residual=0., kkt_tolerance=tolerance, relative_change=0., history=[], stop_reason="empty_design")
    if groups == 0 or p % groups:
        raise ValueError("Feature count must be divisible by group count")
    width = p // groups
    cross = x.T @ y / n
    gram = x.T @ x / n + ridge * np.eye(p) if p <= 1024 else None
    def gradient(b):
        return gram @ b - cross if gram is not None else x.T @ (x @ b - y) / n + ridge * b
    def smooth(b):
        return float(.5 * np.sum((x @ b - y)**2) / n + .5 * ridge * np.sum(b*b))
    def objective(b):
        return smooth(b) + float(np.sum(penalties * np.sqrt(np.sum(b.reshape(groups, width)**2, axis=1))))
    def prox(a, step):
        a = a.reshape(groups, width)
        norms = np.sqrt(np.sum(a*a, axis=1))
        scales = np.maximum(1 - step * penalties / np.maximum(norms, 1e-300), 0)
        return (a * scales[:, None]).ravel()
    def kkt(b):
        bg, gg = b.reshape(groups, width), gradient(b).reshape(groups, width)
        norms = np.sqrt(np.sum(bg*bg, axis=1))
        positive = norms > 0
        residual = np.maximum(np.sqrt(np.sum(gg*gg, axis=1)) - penalties, 0)
        residual[positive] = np.sqrt(np.sum((gg[positive] + penalties[positive, None] * bg[positive] / norms[positive, None])**2, axis=1))
        return _norm(residual)
    # A spectral estimate plus majorization backtracking avoids any SVD.
    direction = np.ones(p) / np.sqrt(p)
    lipschitz = max(ridge, 1e-8)
    for _ in range(20):
        product = gram @ direction if gram is not None else x.T @ (x @ direction) / n + ridge * direction
        size = _norm(product)
        if size <= 1e-15:
            break
        direction = product / size
        lipschitz = max(float(direction @ (gram @ direction if gram is not None else x.T @ (x @ direction) / n + ridge * direction)), 1e-8)
    b = np.zeros(p) if initial is None else np.asarray(initial, float).copy()
    if b.shape != (p,) or not np.isfinite(b).all():
        raise ValueError("Invalid FISTA warm start")
    point, momentum, previous_objective = b.copy(), 1., objective(b)
    threshold = tolerance * max(1., _norm(cross))
    history, restarts, converged = [], 0, False
    for iteration in range(1, max_iterations + 1):
        restarted = False
        for attempt in range(2):
            grad = gradient(point)
            smooth_point = smooth(point)
            for backtrack in range(60):
                candidate = prox(point - grad / lipschitz, 1 / lipschitz)
                change = candidate - point
                if smooth(candidate) <= smooth_point + float(grad @ change) + .5 * lipschitz * np.sum(change*change) + 1e-14:
                    break
                lipschitz *= 2
            else:
                raise FloatingPointError("FISTA backtracking failed")
            value = objective(candidate)
            if value <= previous_objective + 1e-14 or attempt == 1:
                break
            point, momentum, restarted = b.copy(), 1., True
            restarts += 1
        relative = _norm(candidate - b) / max(_norm(b), 1.)
        residual = kkt(candidate)
        converged = bool(residual <= threshold and relative <= tolerance)
        history.append(dict(iteration=iteration, objective=value, kkt_residual=residual, kkt_tolerance=threshold,
                            kkt_ratio=residual / threshold, relative_change=relative, lipschitz=lipschitz,
                            restarted=int(restarted), converged=int(converged)))
        new_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
        next_point = candidate + (momentum - 1) / new_momentum * (candidate - b)
        if float((point - candidate) @ (candidate - b)) > 0:
            next_point, new_momentum = candidate.copy(), 1.
            restarts += 1
        b, point, momentum, previous_objective = candidate, next_point, new_momentum, value
        if converged:
            break
    summary = dict(history[-1])
    summary.update(iterations=iteration, stop_reason="converged" if converged else "iteration_limit",
                   restarts=restarts, warm_start=int(initial is not None), history=history)
    return b, summary


def region_group_fista(
    interactions,
    response,
    region_index,
    n_regions,
    penalties,
    ridge,
    max_iterations,
    tolerance,
    initial=None,
):
    """Matrix-free group FISTA for region-specific interaction coefficients.

    ``interactions`` has one column per LR/TF pair.  A cell uses only the
    coefficient belonging to its region.  This is algebraically identical to
    the old dense ``cells x (pairs * regions)`` design, but never materializes
    its structural zeros.
    """

    z = np.asarray(interactions, dtype=float)
    y = np.asarray(response, dtype=float)
    region_index = np.asarray(region_index, dtype=np.int64)
    penalties = np.asarray(penalties, dtype=float)
    n, groups = z.shape
    n_regions = int(n_regions)
    p = groups * n_regions
    if n == 0 or y.shape != (n,) or region_index.shape != (n,):
        raise ValueError("Region FISTA requires aligned nonempty samples")
    if n_regions < 1 or np.any(region_index < 0) or np.any(region_index >= n_regions):
        raise ValueError("Region FISTA received invalid region indices")
    if penalties.shape != (groups,):
        raise ValueError("Region FISTA penalties must match interaction groups")
    if not all(np.isfinite(a).all() for a in (z, y, penalties)) or not np.isfinite(ridge):
        raise ValueError("Nonfinite region-group-FISTA inputs")
    if ridge < 0 or np.any(penalties < 0) or tolerance <= 0 or max_iterations < 1:
        raise ValueError("Invalid region-group-FISTA penalties, tolerance or budget")
    if groups == 0:
        return np.zeros(0), dict(
            iterations=0,
            converged=1,
            objective=float(0.5 * np.mean(y * y)),
            kkt_residual=0.0,
            kkt_tolerance=tolerance,
            relative_change=0.0,
            history=[],
            stop_reason="empty_design",
            matrix_free=1,
        )

    def predict(vector):
        coefficients = vector.reshape(groups, n_regions)
        return np.sum(z * coefficients[:, region_index].T, axis=1)

    def adjoint(vector):
        output = np.empty((groups, n_regions), dtype=float)
        for group in range(groups):
            output[group] = np.bincount(
                region_index,
                weights=z[:, group] * vector,
                minlength=n_regions,
            )
        return (output / n).ravel()

    cross = adjoint(y)

    def gradient(vector):
        return adjoint(predict(vector) - y) + ridge * vector

    def hessian_product(vector):
        return adjoint(predict(vector)) + ridge * vector

    def smooth(vector):
        residual = predict(vector) - y
        return float(0.5 * np.sum(residual * residual) / n + 0.5 * ridge * np.sum(vector * vector))

    def objective(vector):
        grouped = vector.reshape(groups, n_regions)
        return smooth(vector) + float(
            np.sum(penalties * np.sqrt(np.sum(grouped * grouped, axis=1)))
        )

    def prox(vector, step):
        grouped = vector.reshape(groups, n_regions)
        norms = np.sqrt(np.sum(grouped * grouped, axis=1))
        scales = np.maximum(
            1.0 - step * penalties / np.maximum(norms, 1e-300), 0.0
        )
        return (grouped * scales[:, None]).ravel()

    def kkt(vector):
        grouped = vector.reshape(groups, n_regions)
        grouped_gradient = gradient(vector).reshape(groups, n_regions)
        norms = np.sqrt(np.sum(grouped * grouped, axis=1))
        positive = norms > 0
        residual = np.maximum(
            np.sqrt(np.sum(grouped_gradient * grouped_gradient, axis=1)) - penalties,
            0,
        )
        residual[positive] = np.sqrt(
            np.sum(
                (
                    grouped_gradient[positive]
                    + penalties[positive, None]
                    * grouped[positive]
                    / norms[positive, None]
                )
                ** 2,
                axis=1,
            )
        )
        return _norm(residual)

    direction = np.ones(p, dtype=float) / np.sqrt(p)
    lipschitz = max(float(ridge), 1e-8)
    for _ in range(20):
        product = hessian_product(direction)
        size = _norm(product)
        if size <= 1e-15:
            break
        direction = product / size
        lipschitz = max(float(direction @ hessian_product(direction)), 1e-8)

    b = np.zeros(p) if initial is None else np.asarray(initial, dtype=float).copy()
    if b.shape != (p,) or not np.isfinite(b).all():
        raise ValueError("Invalid region FISTA warm start")
    point, momentum, previous_objective = b.copy(), 1.0, objective(b)
    threshold = tolerance * max(1.0, _norm(cross))
    history: list[dict[str, float | int]] = []
    restarts = 0
    converged = False
    for iteration in range(1, max_iterations + 1):
        restarted = False
        for attempt in range(2):
            grad = gradient(point)
            smooth_point = smooth(point)
            for _ in range(60):
                candidate = prox(point - grad / lipschitz, 1.0 / lipschitz)
                change = candidate - point
                if smooth(candidate) <= (
                    smooth_point
                    + float(grad @ change)
                    + 0.5 * lipschitz * np.sum(change * change)
                    + 1e-14
                ):
                    break
                lipschitz *= 2.0
            else:
                raise FloatingPointError("Region FISTA backtracking failed")
            value = objective(candidate)
            if value <= previous_objective + 1e-14 or attempt == 1:
                break
            point, momentum, restarted = b.copy(), 1.0, True
            restarts += 1
        relative = _norm(candidate - b) / max(_norm(b), 1.0)
        residual = kkt(candidate)
        converged = bool(residual <= threshold and relative <= tolerance)
        history.append(
            dict(
                iteration=iteration,
                objective=value,
                kkt_residual=residual,
                kkt_tolerance=threshold,
                kkt_ratio=residual / threshold,
                relative_change=relative,
                lipschitz=lipschitz,
                restarted=int(restarted),
                converged=int(converged),
            )
        )
        new_momentum = (1.0 + np.sqrt(1.0 + 4.0 * momentum**2)) / 2.0
        next_point = candidate + (momentum - 1.0) / new_momentum * (candidate - b)
        if float((point - candidate) @ (candidate - b)) > 0:
            next_point, new_momentum = candidate.copy(), 1.0
            restarts += 1
        b, point, momentum, previous_objective = (
            candidate,
            next_point,
            new_momentum,
            value,
        )
        if converged:
            break
    summary = dict(history[-1])
    summary.update(
        iterations=iteration,
        stop_reason="converged" if converged else "iteration_limit",
        restarts=restarts,
        warm_start=int(initial is not None),
        matrix_free=1,
        history=history,
    )
    return b, summary
