"""唯一独立方法比较入口；读取完整 Parquet/CSV，不读取 Excel 预览。"""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from baseline_settings.datasets import DATA
from baseline_settings.evaluation import EVALUATION
from independent_baselines.comparison import evaluate

if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("请在 baseline_settings 中修改类字段，不传命令行参数")
    print(evaluate(DATA,EVALUATION))
