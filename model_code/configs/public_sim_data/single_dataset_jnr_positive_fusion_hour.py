"""One-hour JNR search restricted to strictly positive spatial fusion penalty."""
from dataclasses import dataclass
from pathlib import Path
from .single_dataset_jnr_tuning import OneHourJNR, ROOT


@dataclass(frozen=True)
class PositiveFusionHourJNR(OneHourJNR):
    output_root: Path = ROOT/'data/result/optimization_20260921/single_dataset_jnr_positive_fusion_hour_20260924'

    def candidates(self):
        base = dict(l1_penalty=.1, fusion_penalty=.0005, group_penalty=.01,
                    ridge_penalty=.0001, aggregation='mean_abs')
        variants = {
            'l1_009': dict(l1_penalty=.09),
            'l1_0095': dict(l1_penalty=.095),
            'l1_0105': dict(l1_penalty=.105),
            'l1_011': dict(l1_penalty=.11),
            'l1_012': dict(l1_penalty=.12),
            'l1_008': dict(l1_penalty=.08),
            'fusion_0006': dict(fusion_penalty=.0006),
            'fusion_00075': dict(fusion_penalty=.00075),
            'fusion_00125': dict(fusion_penalty=.00125),
            'fusion_0015': dict(fusion_penalty=.0015),
            'fusion_0025': dict(fusion_penalty=.0025),
            'group_0007': dict(group_penalty=.007),
            'group_0008': dict(group_penalty=.008),
            'group_0009': dict(group_penalty=.009),
            'group_0011': dict(group_penalty=.011),
            'group_0012': dict(group_penalty=.012),
            'group_0015': dict(group_penalty=.015),
            'coupled_009_0008': dict(l1_penalty=.09, group_penalty=.008),
            'coupled_011_0012': dict(l1_penalty=.11, group_penalty=.012),
            'coupled_012_0015': dict(l1_penalty=.12, group_penalty=.015),
            'ridge_00005': dict(ridge_penalty=.00005),
            'ridge_0002': dict(ridge_penalty=.0002),
        }
        candidates = {name: dict(base, **change) for name, change in variants.items()}
        if len(candidates) != 22 or any(p['fusion_penalty'] < .0005 for p in candidates.values()):
            raise ValueError('Positive-fusion grid invalid')
        return candidates
