"""Finite one-hour JNR tuning around the accepted positive-fusion winner."""
from dataclasses import dataclass
from pathlib import Path
from .single_dataset_jnr_tuning import OneHourJNR, ROOT


@dataclass(frozen=True)
class Target032HourJNR(OneHourJNR):
    output_root: Path = ROOT/'data/result/optimization_20260921/single_dataset_jnr_target032_hour_20260924'

    def candidates(self):
        base = dict(l1_penalty=.075, fusion_penalty=.0005, group_penalty=.01,
                    ridge_penalty=.0001, aggregation='mean_abs')
        variants = {
            'l1_00725': dict(l1_penalty=.0725),
            'l1_00735': dict(l1_penalty=.0735),
            'l1_0074': dict(l1_penalty=.074),
            'l1_00745': dict(l1_penalty=.0745),
            'l1_00755': dict(l1_penalty=.0755),
            'l1_0076': dict(l1_penalty=.076),
            'l1_00765': dict(l1_penalty=.0765),
            'l1_00775': dict(l1_penalty=.0775),
            'fusion_0006': dict(fusion_penalty=.0006),
            'fusion_0007': dict(fusion_penalty=.0007),
            'fusion_0008': dict(fusion_penalty=.0008),
            'fusion_001': dict(fusion_penalty=.001),
            'fusion_0015': dict(fusion_penalty=.0015),
            'group_0003': dict(group_penalty=.003),
            'group_0005': dict(group_penalty=.005),
            'group_0007': dict(group_penalty=.007),
            'group_0008': dict(group_penalty=.008),
            'group_0012': dict(group_penalty=.012),
            'group_0015': dict(group_penalty=.015),
            'ridge_00005': dict(ridge_penalty=.00005),
            'ridge_0002': dict(ridge_penalty=.0002),
            'coupled_0006_0007': dict(fusion_penalty=.0006, group_penalty=.007),
        }
        candidates = {name: dict(base, **change) for name, change in variants.items()}
        if len(candidates) != 22 or any(p['fusion_penalty'] < .0005 for p in candidates.values()):
            raise ValueError('Target-0.32 grid invalid')
        return candidates
