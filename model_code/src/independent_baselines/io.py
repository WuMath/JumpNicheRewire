"""文件契约：拟合只读白名单输入，评价才读取 truth。"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd

def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".writing")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temp.replace(path)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()

def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()

def mapped_path(value, config):
    value = str(value)
    if config.path_prefix_from:
        old = config.path_prefix_from.replace("\\", "/").rstrip("/")
        normalized = value.replace("\\", "/")
        if normalized == old or normalized.startswith(old + "/"):
            value = config.path_prefix_to.rstrip("/\\") + normalized[len(old):]
    return Path(value)

def selected(config, evaluation=False):
    if config.suite == "perturbation" and not config.perturbation_protocol_validated:
        raise ValueError("扰动协议未验收：先修复对照组特征选择、guide/对照和生物学重复；不要只改开关。")
    path = config.formal_registry if config.suite == "core_final" else (
        config.perturbation_registry if config.suite == "perturbation" else config.evidence_registry)
    if config.suite == "scmultisim":
        if config.split not in {"pilot", "validation", "test"}:
            raise ValueError("scmultisim split must be pilot/validation/test")
        if config.split == "test" and (not config.unlock_test or not config.scmultisim_frozen.is_file()):
            raise PermissionError("scMultiSim test requires validation freeze and unlock_test=True")
        path = config.scmultisim_root / config.split / "dataset_registry.csv"
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if frame.dataset_id.duplicated().any():
        raise ValueError("注册表中 dataset_id 不唯一")
    if config.suite == "scmultisim":
        if not frame["split"].eq(config.split).all() or not frame.evaluation_scope.eq("scmultisim_grn_cci").all():
            raise ValueError("scMultiSim registry scope/split mismatch")
        wanted = config.dataset_ids
    elif config.suite == "core_final":
        if config.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train/validation/test")
        if config.split == "test" and not config.unlock_test:
            raise PermissionError("先冻结方法参数，再在 DatasetConfig 设置 unlock_test=True")
        frame = frame[frame["split"].eq(config.split) & frame.scenario.isin(config.core_scenarios)]
        if "benchmark_profile" in frame and not frame.benchmark_profile.eq("final").all():
            raise ValueError("不能混合 pilot/final 注册表")
        wanted = config.dataset_ids
    else:
        wanted = config.dataset_ids
        if wanted is None:
            names = {"sergio": config.sergio_ids, "sccigen": config.sccigen_ids, "visium": config.visium_ids}
            if config.suite not in names:
                raise ValueError("扰动必须明确填写 dataset_ids；未知 suite 不能自动全量运行")
            wanted = names[config.suite]
    if wanted is not None:
        missing = set(wanted) - set(frame.dataset_id)
        if missing or not wanted:
            raise ValueError(f"所选数据不存在或为空: {sorted(missing)}")
        frame = frame.set_index("dataset_id").loc[list(wanted)].reset_index()
    if frame.empty or frame.dataset_id.str.contains("merfish", case=False).any():
        raise ValueError("选中数据为空，或包含已排除的 MERFISH")
    rows = []
    for _, row in frame.iterrows():
        root = mapped_path(row.dataset_dir, config)
        if config.suite == "scmultisim" and config.split == "test":
            frozen = json.loads(config.scmultisim_frozen.read_text(encoding="utf-8"))["protocol"]
            audit = json.loads((root / "preparation_audit.json").read_text(encoding="utf-8"))
            if digest(audit["protocol"]) != digest(frozen["simulator"]):
                raise PermissionError("scMultiSim test data does not match frozen simulator protocol")
            expected = {f"scmultisim_{frozen['simulator']['scenario']}_test_seed_{s}"
                        for s in frozen["seed_plan"]["test"]}
            if row.dataset_id not in expected:
                raise PermissionError("Dataset not in the prespecified test seed plan")
        item = dict(dataset_id=row.dataset_id, dataset_dir=str(root), species=row.get("species","synthetic"),
                    scenario=row.get("scenario",config.suite), split=row.get("split","external"),
                    suite=config.suite)
        if evaluation:
            item["truth_dir"] = str(mapped_path(row.get("truth_dir", root / "truth"), config))
            item["truth_kind"] = row.get("truth_kind","exact")
        rows.append(item)
    return rows

def inputs(row, config):
    root = Path(row["dataset_dir"])
    # 不枚举数据目录，不读取注册表的真值 signature，不复制 truth。
    expression_file = root / "baseline_input/expression_cells_by_genes.csv"
    if not expression_file.is_file():
        expression_file = root / "expression.csv"
    expression = pd.read_csv(expression_file, index_col=0)
    expression.index = expression.index.astype(str)
    expression.columns = expression.columns.astype(str)
    if expression.index.has_duplicates or expression.columns.has_duplicates:
        raise ValueError("细胞 ID 或基因名不唯一；请先预处理")
    values = expression.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("表达矩阵必须为有限非负值；不要用 z-score 输入")
    metadata = pd.read_csv(root / "metadata.csv", dtype=str, keep_default_na=False).set_index("cell_id")
    coords = pd.read_csv(root / "coordinates.csv", index_col=0)
    coords.index = coords.index.astype(str)
    if metadata.index.has_duplicates or coords.index.has_duplicates:
        raise ValueError("metadata/coordinates 中 ID 重复")
    metadata = metadata.loc[expression.index]
    coords = coords.loc[expression.index]
    if not np.isfinite(coords.to_numpy(dtype=float)).all():
        raise ValueError("坐标非有限值")
    if config.gene_panel_file is not None:
        genes = pd.read_csv(config.gene_panel_file, dtype=str)["gene"].tolist()
        if len(genes) != len(set(genes)) or set(genes) - set(expression.columns):
            raise ValueError("gene_panel 包含重复或不存在的基因")
        expression = expression.loc[:, genes]
        values = expression.to_numpy(dtype=np.float32)
    tf_file = root / "tf_list.csv"
    tfs = pd.read_csv(tf_file, dtype=str).iloc[:,0].dropna().tolist()
    tfs = [tf for tf in dict.fromkeys(tfs) if tf in expression.columns]
    lr_file = root / "lr_pairs.csv"
    lr = pd.read_csv(lr_file, dtype=str, keep_default_na=False)
    if not {"ligand","receptor"}.issubset(lr.columns):
        raise ValueError("lr_pairs.csv 必须包含 ligand,receptor")
    if "lr_id" not in lr:
        lr["lr_id"] = lr.ligand + "__" + lr.receptor
    lr = lr.drop_duplicates(["ligand","receptor"])
    if config.expression_scale == "raw_counts":
        totals = values.sum(axis=1, keepdims=True)
        logged = np.log1p(values * config.target_sum / np.maximum(totals, 1))
    elif config.expression_scale == "log1p_normalized":
        logged = values.copy()
    else:
        raise ValueError("expression_scale 必须明确为 raw_counts 或 log1p_normalized")
    return dict(raw=expression, log=pd.DataFrame(logged,index=expression.index,columns=expression.columns),
                metadata=metadata, coordinates=coords, tfs=tfs, lr=lr,
                files=[expression_file,root/"metadata.csv",root/"coordinates.csv",tf_file,lr_file])

def ann_data(data, config, raw=False):
    import anndata as ad
    frame = data["raw"] if raw else data["log"]
    obj = ad.AnnData(frame.to_numpy(dtype=np.float32), obs=data["metadata"].copy(),
                     var=pd.DataFrame(index=frame.columns.copy()))
    obj.obsm["spatial"] = data["coordinates"].to_numpy(dtype=float)
    if config.cell_type_column in obj.obs:
        obj.obs[config.cell_type_column] = obj.obs[config.cell_type_column].astype("category")
    return obj

def simple_lr(data):
    lr = data["lr"].copy()
    # 复合物绝不通过拆字符串伪造单基因关系；当前公共输入是简单基因对。
    keep = lr.ligand.isin(data["raw"].columns) & lr.receptor.isin(data["raw"].columns)
    if "signal_type" in lr:
        keep &= lr.signal_type.ne("null_placeholder")
    return lr.loc[keep].copy()

def require_types(data, config):
    key = config.cell_type_column
    if key not in data["metadata"] or data["metadata"][key].eq("").any():
        raise ValueError(f"需要完整 metadata.{key}；不要编造细胞类型")
    if data["metadata"][key].nunique() < 2:
        raise ValueError("至少需要两个有效细胞类型；单一类型不能用于此比较")

def safe_name(value):
    if value in {".",".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"不安全的输出名称: {value}")
    return value

class TableWriter:
    """按批写完整 Parquet；Excel 仅预览，不先创建海量字典。"""
    def __init__(self, output, method, preview_rows=10000):
        self.output, self.method = Path(output), method
        if not 0 <= preview_rows <= 100000:
            raise ValueError("preview_rows 必须在 0 到 100000；完整表保存在 Parquet")
        self.preview_rows = preview_rows
        self.tables = {}
        (self.output / "parquet").mkdir(parents=True, exist_ok=True)

    def write(self, name, chunks):
        import pyarrow as pa
        import pyarrow.parquet as pq
        path = self.output / "parquet" / f"{safe_name(name)}.parquet"
        temp = path.with_name(path.name + ".writing")
        writer, total, previews, remaining = None, 0, [], self.preview_rows
        try:
            for frame in chunks:
                frame = frame.copy()
                if "score" not in frame:
                    raise ValueError("统一预测表必须包含 score")
                if "score" in frame:
                    scores = pd.to_numeric(frame.score, errors="raise")
                    if not np.isfinite(scores).all() or (scores < 0).any():
                        raise ValueError("统一 score 必须有限且非负，越大越强")
                    frame["score"] = scores.astype(float)
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temp, table.schema, compression="snappy")
                writer.write_table(table)
                total += len(frame)
                if remaining:
                    previews.append(frame.head(remaining))
                    remaining -= min(remaining, len(frame))
            if writer is None:
                raise ValueError(f"{name}: no schema/output; unsupported task must not be fabricated")
        finally:
            if writer is not None:
                writer.close()
        temp.replace(path)
        preview = pd.concat(previews, ignore_index=True) if previews else pd.DataFrame()
        self.tables[name] = dict(rows=total,path=str(path),preview=preview)

    def finish(self):
        workbook = self.output / f"{self.method}_results.xlsx"
        temp = workbook.with_name(workbook.stem + ".writing.xlsx")
        with pd.ExcelWriter(temp, engine="openpyxl") as writer:
            pd.DataFrame([dict(key="method",value=self.method)]).to_excel(writer,sheet_name="run_info",index=False)
            pd.DataFrame([dict(table=k,rows=v["rows"],parquet=v["path"],preview_rows=len(v["preview"]))
                          for k,v in self.tables.items()]).to_excel(writer,sheet_name="manifest",index=False)
            for name, value in self.tables.items():
                value["preview"].to_excel(writer,sheet_name=name[:31],index=False)
        temp.replace(workbook)
        return workbook
