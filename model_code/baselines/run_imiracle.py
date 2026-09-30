"""iMIRACLE-LRComponent: 独立运行；所有参数在 baselines/baseline_settings/imiracle.py。"""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baseline_settings.datasets import DATA
from baseline_settings.imiracle import CONFIG
from independent_baselines.runtime import launch

if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("不使用命令行参数；请修改配置类。")
    raise SystemExit(launch("imiracle", CONFIG, DATA))
