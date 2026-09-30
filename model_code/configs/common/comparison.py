"""各配置共享的评价/外部结果接口；可在数据配置中覆盖这些对象。"""

from __future__ import annotations
from dataclasses import dataclass
from configs.common.evidence_settings import (
    EvidenceEvaluationConfig, EvidenceJobConfig, ExternalImportConfig, ExternalResultSpec,
)


@dataclass
class EvaluationParameters(EvidenceEvaluationConfig):
    bootstrap_iterations: int = 2000
    confidence_level: float = 0.95


@dataclass
class JobParameters(EvidenceJobConfig):
    pass


@dataclass
class ExternalParameters(ExternalImportConfig):
    # 每个官方方法独立运行后，在这里填写其文件路径和列名映射。
    specifications: tuple[ExternalResultSpec, ...] = ()
