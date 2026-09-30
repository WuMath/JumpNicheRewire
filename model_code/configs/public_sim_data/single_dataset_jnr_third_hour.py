"""Third fixed one-hour JNR-only search on the same development dataset."""
from dataclasses import dataclass
from pathlib import Path
from .single_dataset_jnr_tuning import OneHourJNR, ROOT


@dataclass(frozen=True)
class ThirdHourJNR(OneHourJNR):
    output_root: Path = ROOT/'data/result/optimization_20260921/single_dataset_jnr_third_hour_20260923'

    def candidates(self):
        base = dict(l1_penalty=.1, fusion_penalty=.005, group_penalty=.01,
                    ridge_penalty=.0001, aggregation='mean_abs')
        variants = {
            'fusion_0000': dict(fusion_penalty=0.0),
            'fusion_00005': dict(fusion_penalty=.0005),
            'fusion_0001': dict(fusion_penalty=.001),
            'fusion_0002': dict(fusion_penalty=.002),
            'fusion_0003': dict(fusion_penalty=.003),
            'fusion_0004': dict(fusion_penalty=.004),
            'fusion_0006': dict(fusion_penalty=.006),
            'fusion_00075': dict(fusion_penalty=.0075),
            'fusion_0009': dict(fusion_penalty=.009),
            'l1_004': dict(l1_penalty=.04),
            'l1_006': dict(l1_penalty=.06),
            'l1_008': dict(l1_penalty=.08),
            'l1_012': dict(l1_penalty=.12),
            'l1_015': dict(l1_penalty=.15),
            'group_0001': dict(group_penalty=.001),
            'group_0003': dict(group_penalty=.003),
            'group_0005': dict(group_penalty=.005),
            'group_0015': dict(group_penalty=.015),
            'group_0025': dict(group_penalty=.025),
            'ridge_00001': dict(ridge_penalty=.00001),
            'ridge_00005': dict(ridge_penalty=.00005),
            'ridge_001': dict(ridge_penalty=.001),
        }
        return {name: dict(base, **change) for name, change in variants.items()}
