"""Build a portable archive for the results actually reported in the Chinese paper."""
from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

ARTICLE = Path(__file__).resolve().parent
ROOT = ARTICLE.parents[2]
RECON = ROOT / "reconstruct"
RESULT = RECON / "data/result"
METHODS = ("CeSpGRN", "CLR", "GENIE3", "GRNBoost2",
           "JumpNicheRewire", "PPCOR", "RegDiffusion")
OUT = ARTICLE / "JumpNicheRewire_论文数据代码结果完整包.zip"

spec = importlib.util.spec_from_file_location(
    "scmulti_audit", RECON / "reports/reproduce_scmultisim_development.py")
scmulti = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(scmulti)


def put_file(archive: ZipFile, source: Path, name: str) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    archive.write(source, name)


def put_csv(archive: ZipFile, frame: pd.DataFrame, name: str) -> None:
    archive.writestr(name, frame.to_csv(index=False).encode("utf-8-sig"))


def joined_scores(truth: pd.DataFrame, prediction_path: Path) -> pd.DataFrame:
    pred = pd.read_parquet(prediction_path)[["tf", "target", "score"]]
    if pred.duplicated(["tf", "target"]).any():
        raise ValueError(f"Duplicate predictions: {prediction_path}")
    joined = truth[["tf", "target", "label"]].merge(
        pred, on=["tf", "target"], how="left", validate="one_to_one")
    joined["score"] = joined["score"].fillna(0.0)
    if not np.isfinite(joined.score).all():
        raise ValueError(f"Nonfinite prediction: {prediction_path}")
    return joined


with ZipFile(OUT, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
    for name in ("main_dmaa_revision_zh.tex", "references.bib",
                 "redraw_manuscript_figures.py", "figure_reproduction_audit.json"):
        put_file(archive, ARTICLE / name, f"paper/{name}")
    for name in ("mouse_kidney_communication_case_edge_only.pdf",
                 "human_breast_communication_case_edge_only.pdf"):
        put_file(archive, ARTICLE / "figures" / name, f"paper/figures/{name}")

    for source in (RECON / "src").rglob("*"):
        if source.is_file() and source.suffix in {".py", ".R", ".r"}:
            put_file(archive, source, "model_code/src/" + source.relative_to(RECON / "src").as_posix())
    for source in (RECON / "baselines").rglob("*"):
        if source.is_file() and source.suffix in {".py", ".R", ".r", ".md"}:
            put_file(archive, source, "model_code/baselines/" + source.relative_to(RECON / "baselines").as_posix())
    for source in (RECON / "configs").rglob("*"):
        if source.is_file() and source.suffix in {".json", ".py"}:
            put_file(archive, source, "model_code/configs/" + source.relative_to(RECON / "configs").as_posix())
    put_file(archive, RECON / "main_jumpniche.py", "model_code/main_jumpniche.py")
    for name in ("preprocess_jumpniche.py", "requirements.txt", "README.md"):
        put_file(archive, RECON / name, f"model_code/{name}")
    for name in ("reproduce_scmultisim_development.py",
                 "audit_real_support_extension.py",
                 "analyze_cached_case_communication.py"):
        put_file(archive, RECON / "reports" / name, f"model_code/reports/{name}")
    for name in ("参数配置_scMultiSim.json", "参数配置_GSE205055_P22.json"):
        put_file(archive, RECON / "outputs/three_dataset_benchmark_20260925" / name,
                 f"model_code/experiment_parameters/{name}")

    study = RESULT / "optimization_20260921"
    final = study / "single_dataset_jnr_target032_hour_20260924/final_comparison"
    original = pd.read_csv(final / "nine_method_metrics.csv")
    historical = pd.read_csv(study / "round_02/03_development/run_audit.csv")
    truth = pd.read_csv(scmulti.DATA / "truth/truth_grn.csv", dtype={"tf": str, "target": str})
    sim_rows = original[original.method.isin(METHODS)].copy()
    if len(sim_rows) != len(METHODS) or len(truth) != 624 or truth.label.sum() != 130:
        raise ValueError("scMultiSim source universe has changed")
    for row in sim_rows.itertuples(index=False):
        prediction_path, _, _ = scmulti.prediction_source(row.method, row.candidate, historical)
        joined = joined_scores(truth, prediction_path)
        put_csv(archive, joined, f"results/scmultisim/scores/{row.method}.csv")
    put_csv(archive, sim_rows, "results/scmultisim/source_metrics.csv")
    put_file(archive, scmulti.DATA / "evidence_manifest.json",
             "results/scmultisim/source_data_manifest.json")
    for source in scmulti.DATA.rglob("*"):
        if source.is_file():
            put_file(archive, source, "model_inputs/scmultisim/" + source.relative_to(scmulti.DATA).as_posix())
    sim_raw = RECON / "data/raw/publication_20260920/benchmark/published_tree1_500_cells110_genes_sigma0.1_1"
    for source in sim_raw.rglob("*"):
        if source.is_file():
            put_file(archive, source, "raw_data/scmultisim/" + source.relative_to(sim_raw).as_posix())

    real = RESULT / "real_support_benchmark_20260925"
    p22_truth = pd.read_csv(real / "02_reference/gse205055_p22/support_edges.csv")
    motif_tfs = set(pd.read_parquet(ROOT / "data/processed/references/motif_tf_gene.parquet",
                                    columns=["tf"]).tf.astype(str))
    p22_truth = p22_truth.loc[p22_truth.tf.isin(motif_tfs) &
                              p22_truth.has_mm10_promoter.eq(1)].copy()
    p22_source = pd.read_csv(real / "05_extension/04_evaluation/benchmark_metrics_10_methods.csv")
    p22_rows = p22_source[(p22_source.dataset == "gse205055_p22") &
                           (p22_source.evidence_layer == "label") &
                           (p22_source.method.isin(tuple(m for m in METHODS if m != "JumpNicheRewire") +
                                                 ("JumpNicheRewire_communication",)))].copy()
    if len(p22_rows) != len(METHODS) or len(p22_truth) != 840 or p22_truth.label.sum() != 100:
        raise ValueError("P22 source universe has changed")
    for method in METHODS:
        native = "JumpNicheRewire_communication" if method == "JumpNicheRewire" else method
        base = (real / "03_runs" if native in {"JumpNicheRewire_communication", "CLR",
                                                  "GENIE3", "GRNBoost2", "PPCOR"}
                else real / "05_extension/03_runs")
        prediction_path = base / "gse205055_p22" / native / "global_grn.parquet"
        joined = joined_scores(p22_truth, prediction_path)
        put_csv(archive, joined, f"results/p22/scores/{method}.csv")
    put_csv(archive, p22_rows, "results/p22/source_metrics.csv")
    put_csv(archive, p22_truth, "results/p22/support_edges.csv")
    put_file(archive, ROOT / "data/processed/references/motif_tf_gene.parquet",
             "reference_data/motif_tf_gene.parquet")
    p22_rna = ROOT / "data/raw/Data0925"
    for name in ("GSM6753043_MouseBrain_20um_100barcodes_ATAC_matrix.tsv.gz",
                 "GSM6753043_MouseBrain_20um_100barcodes_ATAC_spatial.tar.gz",
                 "CellChatDB.mouse.rda"):
        put_file(archive, p22_rna / name, f"raw_data/p22/{name}")
    p22_input = real / "01_model_input/gse205055_p22"
    for source in p22_input.rglob("*"):
        if source.is_file():
            put_file(archive, source, "model_inputs/p22/" + source.relative_to(p22_input).as_posix())
    p22_fit = real / "03_runs/gse205055_p22/JumpNicheRewire_communication"
    for source in p22_fit.rglob("*"):
        if source.is_file() and source.suffix != ".log":
            put_file(archive, source, "model_fits/p22/" + source.relative_to(p22_fit).as_posix())

    cases = RESULT / "case_communication_20260926"
    for case in ("case_mouse_kidney", "case_human_breast_cancer"):
        for name in ("analysis_summary.json", "complete_edge_delta_summary.csv",
                     "selected_edge_spatial_values.csv",
                     "selected_edge_region_summary.csv", "expression_umap.csv",
                     "top_global_grn_edges.csv"):
            put_file(archive, cases / case / name, f"results/cases/{case}/{name}")
        case_input = RECON / "data/processed/publication_20260920" / case
        for source in case_input.rglob("*"):
            if source.is_file():
                put_file(archive, source, f"model_inputs/{case}/" + source.relative_to(case_input).as_posix())
        case_fit = RESULT / "publication_20260920/02_runs" / case / "JumpNicheRewire"
        for source in case_fit.rglob("*"):
            if source.is_file() and source.suffix != ".log":
                put_file(archive, source, f"model_fits/{case}/" + source.relative_to(case_fit).as_posix())
        raw_case_name = case.removeprefix("case_")
        raw_case = RECON / "data/raw/publication_20260920/cases" / raw_case_name
        for source in raw_case.iterdir():
            if source.is_file() and source.name != "source_files.json":
                put_file(archive, source, f"raw_data/{case}/{source.name}")

    put_file(archive, ARTICLE / "reproduce_reported_results.py", "reproduce_reported_results.py")
    put_file(archive, ARTICLE / "reproduce_case_from_fits.py", "reproduce_case_from_fits.py")
    put_file(archive, ARTICLE / "make_reproduction_bundle.py", "make_reproduction_bundle.py")
    put_file(archive, ARTICLE / "REPRODUCIBILITY_CN.md", "README_复现教程.md")
    if (ARTICLE / "main_dmaa_revision_zh.pdf").is_file():
        put_file(archive, ARTICLE / "main_dmaa_revision_zh.pdf", "paper/main_dmaa_revision_zh.pdf")

with ZipFile(OUT, "a", compression=ZIP_DEFLATED, compresslevel=6) as archive:
    entries = []
    for item in archive.infolist():
        digest = hashlib.sha256()
        with archive.open(item) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        entries.append({"path": item.filename, "bytes": item.file_size,
                        "sha256": digest.hexdigest()})
    archive.writestr("MANIFEST_SHA256.json",
                     json.dumps({"files": entries}, ensure_ascii=False, indent=2).encode("utf-8"))

print(OUT)
print(f"Archive size: {OUT.stat().st_size / 1024**2:.2f} MiB")
