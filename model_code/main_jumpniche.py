"""唯一模型/benchmark/评价入口：所有实验参数都在 configs 的类中修改。"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import os
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
for folder in (ROOT, ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

# 以下只控制数值库线程，避免 Windows 的 MKL/OpenMP 冲突。
# 它们不保存数据路径、模型参数文件路径或子进程请求。
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"


def write_run_receipt(config, kind, action, state, **extra):
    """保存入口日志；preprocess_jumpniche.py 也复用这个日志函数。"""
    from jumpniche_app.safety import write_json

    try:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        log_file = config.run.output_root / "00_logs" / f"{stamp}_{kind}_{action}.json"
        write_json(log_file, {
            "profile": f"{config.category}.{config.name}",
            "action": action, "state": state,
            "python_executable": sys.executable, "pid": os.getpid(),
            "configuration": config, **extra,
        })
    except Exception as error:
        print(f"[receipt not written] {error}", file=sys.stderr)


def main(dataset_config=None, dataset_row=None, dataset_signature=None):
    """直接执行 main()；括号内三个参数仅供自动隔离数据集时内部使用。"""

    # 每个独立进程直接收到已经读好的配置和数据行。
    # 不读取环境变量，不读取 request.pkl，也不会重新读取你正在编辑的 active.py。
    if dataset_config is not None:
        from evidence_benchmark.run_model import _run_one_dataset

        print(f"[dataset] {dataset_row.dataset_id}", flush=True)
        print(f"[input] {dataset_row.dataset_dir}", flush=True)
        output_dir = dataset_config.output_root / str(dataset_row.evidence_track) / str(dataset_row.dataset_id)
        print(f"[output] {output_dir}", flush=True)
        state = _run_one_dataset(dataset_config, dataset_row, dataset_signature)
        raise SystemExit(0 if state.get("state") == "completed" else 1)

    # 从这里往下是你执行 python main_jumpniche.py 时的正常流程。
    config = None
    action = None
    try:
        if len(sys.argv) != 1:
            raise ValueError("不使用命令行参数。请修改 configs/active.py 和对应的数据配置类。")
        from configs.active import ACTIVE
        from jumpniche_app.configuration import load_profile
        from jumpniche_app.model_runner import run

        config = load_profile(ACTIVE.profile)
        action = ACTIVE.main_action
        print(f"[entry] main / {ACTIVE.profile} / {action}", flush=True)
        print(f"[config file] {ROOT / 'configs' / (ACTIVE.profile.replace('.', '/') + '.py')}", flush=True)
        print(f"[result root] {config.run.output_root}", flush=True)
        result = run(config, action)
        if result is not None:
            print(f"[result] {result}", flush=True)
        if action not in {"check", "status"}:
            write_run_receipt(config, "main", action, "completed", result=str(result))
        return 0
    except KeyboardInterrupt:
        print("[interrupted] 已停止本次运行；已保存的数据集结果和检查点保留。", file=sys.stderr)
        if config is not None and action not in {"check", "status"}:
            write_run_receipt(config, "main", action, "interrupted")
        return 130
    except Exception as error:
        traceback.print_exc()
        if config is not None and action not in {"check", "status"}:
            write_run_receipt(config, "main", action, "failed", error_type=type(error).__name__, traceback=traceback.format_exc())
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
