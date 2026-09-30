"""独立方法调度：配置对象直传子进程；没有隐藏环境变量/请求文件。"""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
import contextlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from .io import selected, inputs, json_write, digest, sha256, TableWriter, safe_name

PACKAGES = {
    "genie3": ("numpy","pandas","scipy","arboreto","distributed","pyarrow","openpyxl","psutil"),
    "grnboost2": ("numpy","pandas","scipy","arboreto","distributed","pyarrow","openpyxl","psutil"),
    "cespgrn": ("torch","numpy","scipy","networkx","scikit-learn","pyarrow","openpyxl","psutil"),
    "spagrn": ("spagrn","ctxcore","anndata","pyarrow","openpyxl","psutil"),
    "liana": ("liana","anndata","pyarrow","openpyxl","psutil"),
    "cellphonedb": ("cellphonedb","anndata","pyarrow","openpyxl","psutil"),
    "imiracle": ("torch","torch-geometric","anndata","pyarrow","openpyxl","psutil"),
    "clarify": ("torch","torch-geometric","node2vec","networkx","pyarrow","openpyxl","psutil"),
}

def environment(name, config):
    packages = {}
    for package in PACKAGES[name]:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            raise ImportError(f"缺少 {package}；请先进入此方法的独立环境，见 baselines/README.md")
    if hasattr(config, "repository"):
        repo = Path(config.repository)
        if not repo.is_dir():
            raise FileNotFoundError(f"请按教程下载官方源码到 {repo}")
        commit = subprocess.run(["git","-C",str(repo),"rev-parse","HEAD"],
                                capture_output=True,text=True,check=True).stdout.strip()
        if commit != config.expected_commit:
            raise ValueError(f"官方代码版本不匹配：当前 {commit}，配置 {config.expected_commit}")
        dirty = subprocess.run(["git","-C",str(repo),"diff","--name-only","HEAD"],
                               capture_output=True,text=True,check=True).stdout.strip()
        if dirty:
            raise ValueError(f"官方源码有未记录改动，不可标为原版：{dirty}")
        packages["official_commit"] = commit
    if hasattr(config, "database") and not Path(config.database).is_file():
        raise FileNotFoundError(f"请手动准备资源文件：{config.database}")
    if hasattr(config, "motif_annotations") and not Path(config.motif_annotations).is_file():
        raise FileNotFoundError(config.motif_annotations)
    return packages

def protocol(name, config, data_config, versions):
    settings = asdict(config)
    for key in ("mode","overwrite","preview_rows"):
        settings.pop(key,None)
    preprocessing = {key:getattr(data_config,key) for key in
                     ("suite","expression_scale","target_sum","cell_type_column","gene_panel_file")}
    resources = {}
    for field in ("database","motif_annotations"):
        if hasattr(config,field):
            resources[field] = sha256(getattr(config,field))
    if data_config.gene_panel_file:
        resources["gene_panel"] = sha256(data_config.gene_panel_file)
    code = {name:sha256(Path(__file__).parent/name) for name in ("io.py","runtime.py","adapters.py")}
    return dict(method=name,parameters=settings,preprocessing=preprocessing,
                resources=resources,versions=versions,adapter_code=code)

class PeakRSS:
    def __init__(self):
        self.peak = 0
        self.stop_event = threading.Event()
    def start(self):
        import psutil
        process = psutil.Process()
        def sample():
            while not self.stop_event.is_set():
                try:
                    total = process.memory_info().rss
                    for child in process.children(recursive=True):
                        try:
                            total += child.memory_info().rss
                        except psutil.Error:
                            pass
                    self.peak = max(self.peak,total)
                except psutil.Error:
                    pass
                self.stop_event.wait(0.2)
        self.thread = threading.Thread(target=sample,daemon=True)
        self.thread.start()
    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2)

def _dataset_main(name, config, data_config, row, lock):
    # 仅控制数值库线程，不存放数据路径。
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
    from .adapters import fit
    output = data_config.output_root / data_config.suite / "02_methods" / config.method / row["dataset_id"]
    output.mkdir(parents=True,exist_ok=True)
    state = dict(method=config.method,dataset_id=row["dataset_id"],state="running",
                 started_utc=datetime.now(timezone.utc).isoformat(),worker_pid=os.getpid(),
                 truth_read_during_inference=False,protocol_signature=digest(lock),
                 convergence_passed=None,convergence_note="官方方法诊断见 raw/log；未伪造统一收敛率")
    status_file = output / "run_status.json"
    json_write(status_file,state)
    monitor = PeakRSS()
    monitor.start()
    started = time.perf_counter()
    try:
        with (output / "console.log").open("w",encoding="utf-8") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                data = inputs(row,data_config)
                input_hashes = {str(path):sha256(path) for path in data["files"]}
                state["input_files"] = input_hashes
                state["cells"],state["genes"] = data["raw"].shape
                signature = digest(dict(protocol=lock,inputs=input_hashes))
                state["signature"] = signature
                json_write(output / "effective_config.json",dict(method=asdict(config),data=asdict(data_config),
                                                                 dataset=row,versions=lock["versions"]))
                writer = TableWriter(output,config.method,config.preview_rows)
                diagnostics = fit(name,config,data_config,row,data,output,writer)
                state.update(diagnostics or {})
                workbook = writer.finish()
                state.update(state="completed",workbook_path=str(workbook),
                             tasks=list(writer.tables),output_hashes={
                                 table:sha256(info["path"]) for table,info in writer.tables.items()})
    except KeyboardInterrupt:
        state.update(state="interrupted",error_message="用户停止",traceback=traceback.format_exc())
    except Exception as error:
        state.update(state="failed",error_type=type(error).__name__,
                     error_message=str(error),traceback=traceback.format_exc())
    finally:
        monitor.stop()
        state.update(runtime_seconds=time.perf_counter()-started,peak_rss_bytes=monitor.peak,
                     finished_utc=datetime.now(timezone.utc).isoformat())
        json_write(status_file,state)
    print(f"[{state['state']}] {row['dataset_id']} -> {output}",flush=True)

def launch(name, config, data_config):
    rows = selected(data_config)
    safe_name(config.method)
    print(f"[dataset config] {Path(sys.argv[0]).parent / 'baseline_settings/datasets.py'}")
    for row in rows:
        safe_name(row["dataset_id"])
        print(f"[input] {row['dataset_id']}: {row['dataset_dir']}")
    print(f"[output] {data_config.output_root / data_config.suite}")
    try:
        versions = environment(name,config)
    except Exception as error:
        print(f"[environment missing] {error}",file=sys.stderr)
        return 1
    current = protocol(name,config,data_config,versions)
    lock_file = data_config.output_root / data_config.suite / "00_protocols" / f"{config.method}.json"
    if config.mode == "check":
        for row in rows:
            data = inputs(row,data_config)
            print(row["dataset_id"],"cells x genes =",data["raw"].shape,"TFs =",len(data["tfs"]))
        print("[check passed] 未拟合、未读取真值。")
        return 0
    if config.mode == "freeze":
        if data_config.suite in {"core_final", "scmultisim"} and data_config.split == "test":
            raise PermissionError("不能在 test 阶段创建/覆盖冻结协议；先回到 validation 固定参数。")
        if lock_file.exists() and json.loads(lock_file.read_text(encoding="utf-8")) != json.loads(json.dumps(current,default=str)):
            raise FileExistsError("已有不同冻结协议；请更换 output_root，不覆盖旧协议。")
        json_write(lock_file,current)
        print(f"[frozen] {lock_file}")
        return 0
    if config.mode != "run":
        raise ValueError("mode must be check/freeze/run")
    if not lock_file.exists():
        raise PermissionError("先把 MethodConfig.mode 改为 freeze 并执行一次，然后改为 run。")
    lock = json.loads(lock_file.read_text(encoding="utf-8"))
    if digest(lock) != digest(current):
        raise PermissionError("参数、预处理、依赖或代码与冻结协议不一致；不能继续原 test。")
    statuses = []
    # 只固定 Python 集合遍历的随机种子，不通过环境变量传递路径或请求。
    # spawn 子进程启动前设置，才能约束官方代码里的 set -> list。
    os.environ["PYTHONHASHSEED"] = str(config.seed)
    for row in rows:
        output = data_config.output_root / data_config.suite / "02_methods" / config.method / row["dataset_id"]
        status_path = output / "run_status.json"
        if status_path.exists() and not config.overwrite:
            previous = json.loads(status_path.read_text(encoding="utf-8"))
            unchanged = (previous.get("state") == "completed" and
                         previous.get("protocol_signature") == digest(lock) and
                         bool(previous.get("input_files")) and
                         all(Path(path).is_file() and sha256(path)==h for path,h in previous.get("input_files",{}).items()) and
                         bool(previous.get("output_hashes")) and
                         all((output/"parquet"/f"{t}.parquet").is_file() and
                             sha256(output/"parquet"/f"{t}.parquet")==h for t,h in previous.get("output_hashes",{}).items()))
            if unchanged:
                print(f"[unchanged] {row['dataset_id']}")
                statuses.append(previous)
                continue
            if previous.get("state") == "completed":
                raise FileExistsError(f"完成结果已变化，拒绝覆盖：{output}；请选择新 output_root。")
        json_write(status_path,dict(method=config.method,dataset_id=row["dataset_id"],state="queued"))
        process = mp.get_context("spawn").Process(target=_dataset_main,args=(name,config,data_config,row,lock))
        try:
            process.start()
            process.join()
            state = json.loads(status_path.read_text(encoding="utf-8"))
            if state.get("state") in {"queued","running"}:
                state.update(state="interrupted",worker_return_code=process.exitcode)
                json_write(status_path,state)
        except KeyboardInterrupt:
            if process.is_alive():
                process.terminate()
                process.join()
            json_write(status_path,dict(method=config.method,dataset_id=row["dataset_id"],state="interrupted"))
            raise
        except Exception as error:
            if process.is_alive():
                process.terminate()
                process.join()
            state=dict(method=config.method,dataset_id=row["dataset_id"],state="failed",
                       error_message=str(error),traceback=traceback.format_exc())
            json_write(status_path,state)
        finally:
            if process.pid is not None and not process.is_alive():
                process.close()
        statuses.append(state)
    import pandas as pd
    inventory = data_config.output_root / data_config.suite / f"{config.method}_run_inventory.csv"
    pd.json_normalize(statuses).to_csv(inventory,index=False)
    print(f"[inventory] {inventory}")
    return 0 if all(s["state"]=="completed" for s in statuses) else 1
