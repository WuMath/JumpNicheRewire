"""唯一数据处理入口：修改 configs/active.py 后直接运行。"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
for folder in (ROOT, ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from main_jumpniche import write_run_receipt


def main():
    """读取预处理配置，然后调用对应的数据处理类。"""
    config = None
    action = None
    try:
        if len(sys.argv) != 1:
            raise ValueError("不使用命令行参数。请修改 configs/active.py 和对应的数据配置类。")
        from configs.active import ACTIVE
        from jumpniche_app.configuration import load_profile
        from jumpniche_app.preprocessors import run

        config = load_profile(ACTIVE.profile)
        action = ACTIVE.preprocess_action
        print(f"[entry] preprocess / {ACTIVE.profile} / {action}", flush=True)
        print(f"[config file] {ROOT / 'configs' / (ACTIVE.profile.replace('.', '/') + '.py')}", flush=True)
        result = run(config, action)
        if result is not None:
            print(f"[result] {result}", flush=True)
        if action != "check":
            write_run_receipt(config, "preprocess", action, "completed", result=str(result))
        return 0
    except KeyboardInterrupt:
        print("[interrupted] 预处理已停止，已有输入和模型结果未删除。", file=sys.stderr)
        if config is not None and action != "check":
            write_run_receipt(config, "preprocess", action, "interrupted")
        return 130
    except Exception as error:
        traceback.print_exc()
        if config is not None and action != "check":
            write_run_receipt(config, "preprocess", action, "failed", error_type=type(error).__name__, traceback=traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
