"""共享默认值；各数据配置可继承/覆盖，算法实现不在这里。"""

from __future__ import annotations
from dataclasses import dataclass, field
from configs.common.model import ProjectConfig, BaseGRNConfig, SpatialConfig, OutputConfig


@dataclass
class SmoothGRNConfig(BaseGRNConfig):
    l1_penalty: float = 0.02
    fusion_penalty: float = 0.15


@dataclass
class FormalSpatialConfig(SpatialConfig):
    spatial_permutations: int = 999


@dataclass
class CompactOutputConfig(OutputConfig):
    # 程序实际保留的表写入 Parquet，Excel 仅作小规模预览。
    # top_grn_rows/top_lr_gene_rows 等候选截断限制依然生效。
    maximum_excel_rows_per_table: int = 10000


@dataclass
class ModelParameters(ProjectConfig):
    base_grn: BaseGRNConfig = field(default_factory=SmoothGRNConfig)
    spatial: SpatialConfig = field(default_factory=FormalSpatialConfig)
    output: OutputConfig = field(default_factory=CompactOutputConfig)
