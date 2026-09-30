"""Simulation generator aligned to all six benchmark output levels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from grn2_project.paths import PROCESSED_DATA_ROOT
from .utils import ensure_directory, standardize


from configs.self_sim_data.simulation_settings import SimulationConfig, SIMULATION_CONFIG





def _negative_binomial(
    mean: np.ndarray,
    dispersion: float,
    random: np.random.Generator,
) -> np.ndarray:
    mean = np.clip(mean, 1e-4, 1e4)
    latent = random.gamma(shape=dispersion, scale=mean / dispersion)
    return random.poisson(latent).astype(float)


def _communication_truth(
    coordinates: np.ndarray,
    ligand: np.ndarray,
    receptor: np.ndarray,
    neighbor_count: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    n, pathway_count = ligand.shape
    tree = cKDTree(coordinates)
    k = min(neighbor_count, n - 1)
    distances, neighbors = tree.query(coordinates, k=k + 1)
    bandwidth = np.maximum(distances[:, -1], 1e-6)
    exposure = np.zeros((n, pathway_count), dtype=float)
    rows: list[dict[str, object]] = []
    for receiver in range(n):
        sender_indices = neighbors[receiver, 1:]
        sender_distance = distances[receiver, 1:]
        denominator = 2.0 * np.maximum(bandwidth[sender_indices] * bandwidth[receiver], 1e-12)
        base_kernel = np.exp(-(sender_distance**2) / denominator)
        base_kernel /= max(base_kernel.sum(), 1e-12)
        for pathway in range(pathway_count):
            contribution = base_kernel * ligand[sender_indices, pathway] * receptor[receiver, pathway]
            exposure[receiver, pathway] = contribution.sum()
            for sender, score in zip(sender_indices, contribution):
                if score > 0:
                    rows.append(
                        {
                            "sender_index": int(sender),
                            "receiver_index": receiver,
                            "lr_index": pathway,
                            "true_score": float(score),
                        }
                    )
    return exposure, pd.DataFrame(rows)


def generate_simulated_project(config: SimulationConfig) -> Path:
    random = np.random.default_rng(config.random_seed)
    root = ensure_directory(Path(config.output_dir))
    truth_dir = ensure_directory(root / config.truth_dir_name)
    region_names = [f"region_{index + 1}" for index in range(config.region_count)]
    n = config.region_count * config.cells_per_region
    region_index = np.repeat(np.arange(config.region_count), config.cells_per_region)
    cell_ids = np.array([f"cell_{index + 1:04d}" for index in range(n)])

    angles = np.linspace(0, 2 * np.pi, config.region_count, endpoint=False)
    centers = np.column_stack([np.cos(angles), np.sin(angles)]) * config.region_separation
    coordinates = centers[region_index] + random.normal(0.0, config.coordinate_spread, size=(n, 2))
    if config.causal_confounding not in {"none", "measured", "unmeasured"}:
        raise ValueError("causal_confounding must be none, measured, or unmeasured")
    coordinate_signal = coordinates[:, 0]
    coordinate_signal = (coordinate_signal - coordinate_signal.mean()) / max(coordinate_signal.std(), 1e-8)
    latent_confounder = 0.65 * coordinate_signal + random.normal(0.0, 0.75, size=n)
    latent_confounder = (latent_confounder - latent_confounder.mean()) / max(latent_confounder.std(), 1e-8)
    active_confounder = (
        latent_confounder if config.causal_confounding in {"measured", "unmeasured"} else np.zeros(n)
    )

    tf_names = [f"TF{index + 1}" for index in range(config.tf_count)]
    ligand_names = [f"L{index + 1}" for index in range(config.lr_count)]
    receptor_names = [f"R{index + 1}" for index in range(config.lr_count)]
    target_names = [f"G{index + 1}" for index in range(config.target_count)]
    noise_names = [f"N{index + 1}" for index in range(config.noise_gene_count)]
    genes = tf_names + ligand_names + receptor_names + target_names + noise_names
    expression = np.zeros((n, len(genes)), dtype=float)
    gene_index = {gene: position for position, gene in enumerate(genes)}

    tf_mean = np.full((n, config.tf_count), 2.5)
    for tf_index in range(config.tf_count):
        tf_mean[:, tf_index] += 5.0 * (region_index == tf_index % config.region_count)
    tf_mean *= np.exp(0.12 * config.confounder_strength * active_confounder[:, None])
    tf_counts = _negative_binomial(tf_mean, config.count_dispersion, random)
    expression[:, : config.tf_count] = tf_counts

    ligand_mean = np.full((n, config.lr_count), 0.8)
    receptor_mean = np.full((n, config.lr_count), 0.8)
    for lr_index in range(config.lr_count):
        sender_region = lr_index % config.region_count
        receiver_region = (lr_index + 1) % config.region_count
        ligand_mean[:, lr_index] += 9.0 * (region_index == sender_region)
        receptor_mean[:, lr_index] += 7.0 * (region_index == receiver_region)
    ligand_mean *= np.exp(0.35 * config.confounder_strength * active_confounder[:, None])
    receptor_mean *= np.exp(0.10 * config.confounder_strength * active_confounder[:, None])
    ligand_counts = _negative_binomial(ligand_mean, config.count_dispersion, random)
    receptor_counts = _negative_binomial(receptor_mean, config.count_dispersion, random)
    ligand_start = config.tf_count
    receptor_start = ligand_start + config.lr_count
    expression[:, ligand_start:receptor_start] = ligand_counts
    expression[:, receptor_start : receptor_start + config.lr_count] = receptor_counts

    ligand_proxy = ligand_counts / (ligand_counts + np.maximum(np.median(ligand_counts, axis=0), 1.0))
    receptor_proxy = receptor_counts / (receptor_counts + np.maximum(np.median(receptor_counts, axis=0), 1.0))
    exposure, contribution_truth = _communication_truth(
        coordinates,
        ligand_proxy,
        receptor_proxy,
        config.communication_neighbors,
    )
    tf_standardized, _, _ = standardize(np.log1p(tf_counts), axis=0)

    beta_true = np.zeros((config.region_count, config.tf_count, config.target_count), dtype=float)
    for region in range(config.region_count):
        mask = random.random((config.tf_count, config.target_count)) < config.base_edge_density
        signs = random.choice([-1.0, 1.0], size=mask.shape)
        beta_true[region] = mask * signs * random.uniform(0.12, 0.35, size=mask.shape)
        for target in range(config.target_count):
            beta_true[region, target % config.tf_count, target] = 0.22 + 0.04 * region

    gamma_true = np.zeros(
        (config.region_count, config.lr_count, config.tf_count, config.target_count), dtype=float
    )
    for region in range(config.region_count):
        mask = random.random((config.lr_count, config.tf_count, config.target_count)) < config.rewiring_edge_density
        signs = random.choice([-1.0, 1.0], size=mask.shape)
        gamma_true[region] = mask * signs * random.uniform(0.18, 0.45, size=mask.shape)
        lr_index = region % config.lr_count
        gamma_true[region, lr_index, region % config.tf_count, region % config.target_count] = 0.40

    target_linear = np.full((n, config.target_count), 1.25)
    confounder_loadings = random.choice([-1.0, 1.0], size=config.target_count) * random.uniform(
        0.25, 0.55, size=config.target_count
    )
    target_linear += config.confounder_strength * active_confounder[:, None] * confounder_loadings[None, :]
    for cell in range(n):
        region = region_index[cell]
        target_linear[cell] += tf_standardized[cell] @ beta_true[region]
        for lr_index in range(config.lr_count):
            target_linear[cell] += (
                exposure[cell, lr_index]
                * tf_standardized[cell]
                @ gamma_true[region, lr_index]
            )
    target_mean = np.exp(np.clip(target_linear, -1.5, 3.2))
    target_counts = _negative_binomial(target_mean, config.count_dispersion, random)
    target_start = receptor_start + config.lr_count
    expression[:, target_start : target_start + config.target_count] = target_counts
    noise_start = target_start + config.target_count
    noise_mean = random.uniform(0.5, 3.0, size=(n, config.noise_gene_count))
    expression[:, noise_start:] = _negative_binomial(noise_mean, config.count_dispersion, random)

    expression_frame = pd.DataFrame(expression, index=cell_ids, columns=genes)
    expression_frame.index.name = "cell_id"
    coordinates_frame = pd.DataFrame(coordinates, index=cell_ids, columns=["x", "y"])
    coordinates_frame.index.name = "cell_id"
    metadata = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "sample_id": "sample_1",
            "cell_type": [f"type_{index + 1}" for index in region_index],
            "region": [region_names[index] for index in region_index],
            "condition": np.where(region_index == 0, "control", "stimulated"),
        }
    )
    if config.causal_confounding == "measured":
        metadata["measured_confounder"] = latent_confounder
    lr_pairs = pd.DataFrame(
        {
            "lr_id": [f"LR{index + 1}" for index in range(config.lr_count)],
            "ligand": ligand_names,
            "receptor": receptor_names,
            "signal_type": ["secreted" if index % 3 else "contact" for index in range(config.lr_count)],
            "rho": [1.0 if index % 3 else 0.6 for index in range(config.lr_count)],
            "confidence": 1.0,
        }
    )

    prior_rows: list[dict[str, object]] = []
    for tf_index, tf in enumerate(tf_names):
        for target_index, target in enumerate(target_names):
            supported = float(np.any(beta_true[:, tf_index, target_index] != 0))
            prior_rows.append(
                {
                    "tf": tf,
                    "target": target,
                    "motif": np.clip(0.15 + 0.75 * supported + random.normal(0, 0.08), 0, 1),
                    "atac": np.nan,
                    "grn": np.clip(0.10 + 0.70 * supported + random.normal(0, 0.10), 0, 1),
                }
            )
    lr_tf_rows: list[dict[str, object]] = []
    for lr_index in range(config.lr_count):
        for tf_index, tf in enumerate(tf_names):
            supported = float(np.any(gamma_true[:, lr_index, tf_index, :] != 0))
            lr_tf_rows.append(
                {"lr_id": f"LR{lr_index + 1}", "tf": tf, "score": 0.15 + 0.75 * supported}
            )

    expression_frame.to_csv(root / "expression.csv")
    coordinates_frame.to_csv(root / "coordinates.csv")
    metadata.to_csv(root / "metadata.csv", index=False)
    pd.DataFrame({"tf": tf_names}).to_csv(root / "tf_list.csv", index=False)
    lr_pairs.to_csv(root / "lr_pairs.csv", index=False)
    pd.DataFrame(prior_rows).to_csv(root / "prior_tf_gene.csv", index=False)
    pd.DataFrame(lr_tf_rows).to_csv(root / "prior_lr_tf.csv", index=False)

    beta_cell = beta_true[region_index]
    gamma_cell = gamma_true[region_index]
    delta_cell = np.einsum("ia,iakg->ikg", exposure, gamma_cell)
    beta_ccc = beta_cell + delta_cell
    grn_rows: list[dict[str, object]] = []
    for cell in range(n):
        for tf_index, tf in enumerate(tf_names):
            for target_index, target in enumerate(target_names):
                value = float(beta_ccc[cell, tf_index, target_index])
                grn_rows.append(
                    {
                        "cell_id": cell_ids[cell],
                        "region": region_names[region_index[cell]],
                        "tf": tf,
                        "target": target,
                        "label": int(abs(value) > 1e-10),
                        "true_score": abs(value),
                        "signed_truth": value,
                        "beta0_true": float(beta_cell[cell, tf_index, target_index]),
                        "delta_beta_true": float(delta_cell[cell, tf_index, target_index]),
                    }
                )
    pd.DataFrame(grn_rows).to_csv(truth_dir / "truth_grn.csv", index=False)

    lr_gene_rows: list[dict[str, object]] = []
    for cell in range(n):
        region = region_index[cell]
        for lr_index in range(config.lr_count):
            for target_index, target in enumerate(target_names):
                score = float(
                    exposure[cell, lr_index]
                    * np.sum(tf_standardized[cell] * gamma_true[region, lr_index, :, target_index])
                )
                lr_gene_rows.append(
                    {
                        "receiver_id": cell_ids[cell],
                        "region": region_names[region],
                        "lr_id": f"LR{lr_index + 1}",
                        "target": target,
                        "label": int(abs(score) > 1e-10),
                        "true_score": abs(score),
                        "signed_truth": score,
                    }
                )
    pd.DataFrame(lr_gene_rows).to_csv(truth_dir / "truth_lr_gene.csv", index=False)

    receptor_rows: list[dict[str, object]] = []
    for region in range(config.region_count):
        for lr_index, receptor in enumerate(receptor_names):
            for tf_index, tf in enumerate(tf_names):
                for target_index, target in enumerate(target_names):
                    label = int(abs(gamma_true[region, lr_index, tf_index, target_index]) > 1e-10)
                    if label:
                        receptor_rows.append(
                            {
                                "region": region_names[region],
                                "receptor": receptor,
                                "tf": tf,
                                "target": target,
                                "label": 1,
                            }
                        )
    pd.DataFrame(receptor_rows).to_csv(truth_dir / "truth_receptors.csv", index=False)

    contribution_truth["sender_id"] = cell_ids[contribution_truth["sender_index"].to_numpy(dtype=int)]
    contribution_truth["receiver_id"] = cell_ids[contribution_truth["receiver_index"].to_numpy(dtype=int)]
    aggregate_truth = contribution_truth.groupby(["sender_id", "receiver_id"], as_index=False)["true_score"].sum()
    full_pairs = pd.MultiIndex.from_product([cell_ids, cell_ids], names=["sender_id", "receiver_id"]).to_frame(index=False)
    full_pairs = full_pairs[full_pairs["sender_id"] != full_pairs["receiver_id"]]
    full_pairs = full_pairs.merge(aggregate_truth, on=["sender_id", "receiver_id"], how="left").fillna({"true_score": 0.0})
    positive_values = full_pairs.loc[full_pairs["true_score"] > 0, "true_score"]
    threshold = float(positive_values.quantile(0.75)) if len(positive_values) else np.inf
    full_pairs["label"] = (full_pairs["true_score"] >= threshold).astype(int)
    full_pairs.to_csv(truth_dir / "truth_cci.csv", index=False)

    causal_rows: list[dict[str, object]] = []
    for region in range(config.region_count):
        for lr_index in range(config.lr_count):
            for tf_index, tf in enumerate(tf_names):
                for target_index, target in enumerate(target_names):
                    value = float(gamma_true[region, lr_index, tf_index, target_index])
                    if abs(value) > 1e-10:
                        causal_rows.append(
                            {
                                "region": region_names[region],
                                "lr_id": f"LR{lr_index + 1}",
                                "tf": tf,
                                "target": target,
                                "gamma_true": value,
                                "causal_true": value,
                            }
                        )
    causal_truth = pd.DataFrame(causal_rows)
    causal_truth.to_csv(truth_dir / "truth_causal.csv", index=False)
    pd.DataFrame(
        {
            "cell_id": cell_ids,
            "confounder_value": latent_confounder,
            "observed_in_metadata": config.causal_confounding == "measured",
            "causal_scenario": config.causal_confounding,
        }
    ).to_csv(truth_dir / "truth_confounder.csv", index=False)

    if config.include_activity_evidence:
        pd.DataFrame(ligand_proxy, index=cell_ids, columns=[f"LR{i + 1}" for i in range(config.lr_count)]).rename_axis("cell_id").to_csv(root / "ligand_activity.csv")
        pd.DataFrame(receptor_proxy, index=cell_ids, columns=[f"LR{i + 1}" for i in range(config.lr_count)]).rename_axis("cell_id").to_csv(root / "receptor_activity.csv")
        pd.DataFrame(tf_standardized, index=cell_ids, columns=tf_names).rename_axis("cell_id").to_csv(root / "tf_activity.csv")
    if config.include_perturbation_evidence and not causal_truth.empty:
        perturbations = causal_truth.copy()
        perturbations["delta_observed"] = perturbations["gamma_true"] + random.normal(
            0.0, config.perturbation_noise, size=len(perturbations)
        )
        perturbations["design_quality"] = "randomized_simulation"
        perturbations[["region", "lr_id", "tf", "target", "delta_observed", "design_quality"]].to_csv(
            root / "perturbations.csv", index=False
        )
    return root
