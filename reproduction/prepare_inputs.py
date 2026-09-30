"""Rebuild the four inputs used in the manuscript from bundled raw files.

Outputs go to reproduction_runs/prepared. Published model_inputs are read only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
for thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_var] = "1"
CODE = ROOT / "model_code"
sys.path[:0] = [str(CODE), str(CODE / "src")]

import numpy as np
import pandas as pd


OUT = ROOT / "reproduction_runs" / "prepared"
RAW = ROOT / "raw_data"


def scmultisim() -> Path:
    from publication_study.prepare import prepare_benchmark

    dest = OUT / "scmultisim"
    if dest.exists():
        compare("scmultisim")
        if not (OUT / "development_registry.csv").is_file():
            raise FileNotFoundError("Prepared scMultiSim input exists but development_registry.csv is missing")
        print("REUSE: validated existing scMultiSim input")
        return dest
    raw = RAW / "scmultisim"
    job = dict(dataset_id="published_tree1_500_cells110_genes_sigma0.1_1",
               export_dir=str(raw / "export"), split="development",
               stratum="tree1_sigma0.1", independence_group="tree1_sigma0.1")
    cfg = SimpleNamespace(processed_root=OUT, benchmark_maximum_targets=200)
    rows = prepare_benchmark(cfg, [job])
    generated = OUT / job["dataset_id"]
    generated.rename(dest)
    manifest = dest / "evidence_manifest.json"
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["dataset_dir"] = str(dest.resolve())
    record["truth_dir"] = str((dest / "truth").resolve())
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    rows[0]["dataset_dir"] = str(dest.resolve())
    rows[0]["truth_dir"] = str((dest / "truth").resolve())
    pd.DataFrame(rows).to_csv(OUT / "development_registry.csv", index=False)
    return dest


def p22(rscript: Path) -> Path:
    from real_support_benchmark.prepare import _p22, _cellchat_pairs

    dest = OUT / "p22"
    if dest.exists():
        compare("p22")
        print("REUSE: validated existing P22 input")
        return dest
    cfg = SimpleNamespace(external_raw=RAW / "p22", output_root=OUT / "p22_aux",
                          spots=500, seed=20260925, rscript=rscript)
    expression, coords, audit = _p22(cfg)
    catalog = pd.read_csv(RAW / "references" / "tf_list.csv", dtype=str).tf.dropna().astype(str).tolist()
    lr_catalog = _cellchat_pairs(cfg)
    detected = (expression > 0).mean()
    logged = np.log1p(expression.div(expression.sum(axis=1).clip(lower=1), axis=0) * 10000)
    variance = logged.var(ddof=0)
    eligible = detected[detected.ge(.02)].index
    tfs = sorted((g for g in catalog if g in eligible), key=lambda g: (-variance[g], g))[:20]
    targets = sorted((g for g in eligible if g not in tfs), key=lambda g: (-variance[g], g))[:120]
    if len(tfs) != 20 or len(targets) != 120:
        raise ValueError("P22 selected TF/target counts changed")
    pairs = lr_catalog[lr_catalog.ligand.isin(eligible) & lr_catalog.receptor.isin(eligible)].copy()
    pairs["rank"] = pairs.ligand.map(logged.mean()) * pairs.receptor.map(logged.mean())
    pairs = pairs.sort_values(["rank", "ligand", "receptor"], ascending=[False, True, True]).head(20)
    pairs["lr_id"] = pairs.ligand + "__" + pairs.receptor
    genes = sorted(set(tfs) | set(targets) | set(pairs.ligand) | set(pairs.receptor))
    dest.mkdir(parents=True)
    expression.loc[:, genes].to_csv(dest / "expression.csv")
    coords.to_csv(dest / "coordinates.csv")
    pd.DataFrame({"cell_id": expression.index, "sample_id": "gse205055_p22",
                  "cell_type": "unassigned", "region": "global"}).to_csv(dest / "metadata.csv", index=False)
    pd.DataFrame({"tf": tfs}).to_csv(dest / "tf_list.csv", index=False)
    pairs[["lr_id", "ligand", "receptor"]].to_csv(dest / "lr_pairs.csv", index=False)
    pd.DataFrame(columns=["tf", "target", "motif", "atac", "grn"]).to_csv(dest / "prior_tf_gene.csv", index=False)
    pd.DataFrame(columns=["lr_id", "tf", "score"]).to_csv(dest / "prior_lr_tf.csv", index=False)
    pd.DataFrame({"tf": tfs}).to_csv(dest / "evaluation_tfs.csv", index=False)
    pd.DataFrame({"target": targets}).to_csv(dest / "evaluation_targets.csv", index=False)
    audit.update(dataset="gse205055_p22", selected_spots=len(expression), input_genes=len(genes),
                 tfs=tfs, targets=len(targets), lr_pairs=len(pairs), seed=cfg.seed,
                 reference_sources_read=False)
    (dest / "prepare_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


def p22_labels() -> Path:
    from real_support_benchmark.references import p22 as make_labels

    if not (OUT / "p22" / "evaluation_tfs.csv").exists():
        raise FileNotFoundError("Run the p22 input step first")
    base = OUT / "p22_aux"
    target = base / "01_model_input" / "gse205055_p22"
    target.mkdir(parents=True, exist_ok=True)
    for source in (OUT / "p22").iterdir():
        if source.suffix != ".csv":
            continue
        destination = target / source.name
        if not destination.exists():
            destination.write_bytes(source.read_bytes())
        elif destination.read_bytes() != source.read_bytes():
            raise ValueError(f"P22 benchmark input differs from prepared input: {destination}")
    labels = base / "02_reference" / "gse205055_p22" / "support_edges.csv"
    audit = labels.with_name("reference_audit.json")
    if labels.exists():
        if not audit.is_file():
            raise FileNotFoundError(f"P22 labels exist without reference audit: {audit}")
        compare("p22-labels")
        print("REUSE: validated existing P22 support labels")
        return labels.parent
    cfg = SimpleNamespace(output_root=base, raw_root=RAW,
                          gtf_path=RAW / "mouse_annotations" / "Mus_musculus.GRCm38.86.gtf.gz",
                          fragment_path=RAW / "p22" / "GSM6758285_MouseBrain_20um_100barcodes_ATAC_fragments.tsv.gz",
                          motif_catalog=ROOT / "reference_data" / "motif_tf_gene.parquet",
                          task_limit_seconds=4 * 3600, worker_rss_limit=int(3.5 * 1024**3),
                          minimum_available_ram=int(1.5 * 1024**3),
                          minimum_disk_free=5 * 1024**3)
    return make_labels(cfg)


def case(name: str) -> Path:
    from configs.real_data.settings.prepare_visium import VisiumConfig
    from preprocess.prepare_visium import main as convert_visium
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    dest = OUT / name
    if dest.exists() and not (dest / "source_manifest.json").exists():
        raise FileExistsError(f"Incomplete Visium conversion requires review: {dest}")
    is_human = name == "case_human_breast_cancer"
    prefix = "Visium_FFPE_Human_Breast_Cancer" if is_human else "Visium_FFPE_Mouse_Kidney"
    raw = RAW / name
    cfg = VisiumConfig(
        matrix_h5=raw / (prefix + "_filtered_feature_bc_matrix.h5"),
        spatial_source=raw / (prefix + "_spatial.tar.gz"),
        reference_dir=RAW / ("references_human" if is_human else "references"),
        output_dir=dest, maximum_spots=None, maximum_genes=600, maximum_tfs=12,
        maximum_lr_pairs=40, minimum_detected_spots=8, random_seed=20260920,
        sample_id=prefix, cell_type_label="NA", region_label="global",
        condition_label="tumor" if is_human else "healthy")
    if not dest.exists():
        convert_visium(cfg)
    metadata = pd.read_csv(dest / "metadata.csv")
    if "annotation_source" in metadata:
        compare(name)
        print(f"REUSE: validated existing {name} input")
        return dest
    expression = pd.read_csv(dest / "expression.csv", index_col=0)
    normalized = np.log1p(expression.div(expression.sum(axis=1).replace(0, 1), axis=0) * 10000)
    pcs = PCA(n_components=10, random_state=20260920).fit_transform(normalized)
    clusters = KMeans(n_clusters=6, n_init=10, random_state=20260920).fit_predict(pcs)
    metadata["region"] = ["expression_cluster_" + str(i + 1) for i in clusters]
    metadata["cell_type"] = "NA"
    metadata["annotation_source"] = "unsupervised_expression_cluster_not_celltype_or_histology"
    metadata.to_csv(dest / "metadata.csv", index=False)
    if is_human:
        from configs.real_data.settings.prepare_lr_tf import LRTFConfig
        from preprocess.prepare_lr_tf import prepare_signaling_edges, build_lr_tf_prior

        signaling = RAW / "publication_20260920" / "references" / "human_signaling.tsv"
        prior_cfg = LRTFConfig(reference_dir=dest, maximum_frontier_nodes=1000, maximum_path_length=4)
        signal = prepare_signaling_edges(pd.read_csv(signaling, sep="\t"), prior_cfg)
        prior, paths, audit = build_lr_tf_prior(
            signal, pd.read_csv(dest / "lr_pairs.csv"), pd.read_csv(dest / "tf_list.csv"), prior_cfg)
        prior.to_csv(dest / "prior_lr_tf.csv", index=False)
        paths.to_parquet(dest / "reference_receptor_tf_paths.parquet", index=False)
        audit.to_csv(dest / "lr_tf_audit.csv", index=False)
    return dest


def compare(name: str) -> None:
    if name == "p22-labels":
        made = pd.read_csv(OUT / "p22_aux/02_reference/gse205055_p22/support_edges.csv")
        published = pd.read_csv(ROOT / "results/p22/support_edges.csv")
        motif_tfs = set(pd.read_parquet(ROOT / "reference_data/motif_tf_gene.parquet", columns=["tf"]).tf.astype(str))
        made = made.loc[made.tf.isin(motif_tfs) & made.has_mm10_promoter.eq(1)]
        made = made.sort_values(["tf", "target"]).reset_index(drop=True)
        published = published.sort_values(["tf", "target"]).reset_index(drop=True)
        if not made.equals(published):
            raise ValueError("Regenerated P22 support labels differ from the paper")
        print(f"PASS: P22 {len(made)} evaluated edges and {int(made.label.sum())} support labels agree")
        return
    current = OUT / name
    reported = ROOT / "model_inputs" / name
    for filename in ("expression.csv", "coordinates.csv", "metadata.csv", "tf_list.csv",
                     "lr_pairs.csv", "prior_tf_gene.csv", "prior_lr_tf.csv"):
        a, b = pd.read_csv(current / filename), pd.read_csv(reported / filename)
        if not a.equals(b):
            raise ValueError(f"Different processed table: {name}/{filename}")
    additional = (("truth/truth_grn.csv", "evaluation_targets.csv") if name == "scmultisim"
                  else ("evaluation_tfs.csv", "evaluation_targets.csv") if name == "p22"
                  else ())
    for filename in additional:
        if not pd.read_csv(current / filename).equals(pd.read_csv(reported / filename)):
            raise ValueError(f"Different processed table: {name}/{filename}")
    print(f"PASS: {name} processed inputs equal the manuscript inputs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("scmultisim", "p22", "p22-labels",
                                         "case_mouse_kidney", "case_human_breast_cancer", "compare"))
    parser.add_argument("--dataset", choices=("scmultisim", "p22", "p22-labels",
                                               "case_mouse_kidney", "case_human_breast_cancer"))
    parser.add_argument("--rscript", type=Path, default=Path("D:/SoftWare/R/R-4.4.1/bin/Rscript.exe"))
    args = parser.parse_args()
    if args.step == "compare":
        if args.dataset is None:
            parser.error("compare requires --dataset")
        compare(args.dataset)
        return
    result = (scmultisim() if args.step == "scmultisim" else
              p22(args.rscript) if args.step == "p22" else
              p22_labels() if args.step == "p22-labels" else case(args.step))
    print(result)


if __name__ == "__main__":
    main()
