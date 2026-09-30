"""Second fixed, one-hour JNR-only parameter search on the same development set."""
from dataclasses import dataclass
from pathlib import Path
from .single_dataset_jnr_tuning import OneHourJNR, ROOT


@dataclass(frozen=True)
class SecondHourJNR(OneHourJNR):
    output_root: Path = ROOT/'data/result/optimization_20260921/single_dataset_jnr_second_hour_20260923'

    def candidates(self):
        base = dict(l1_penalty=.1,fusion_penalty=.02,group_penalty=.01,
                    ridge_penalty=.0001,aggregation='mean_abs')
        variants = {
            'fusion_0005': dict(fusion_penalty=.005),
            'fusion_001': dict(fusion_penalty=.01),
            'fusion_0015': dict(fusion_penalty=.015),
            'fusion_0025': dict(fusion_penalty=.025),
            'fusion_003': dict(fusion_penalty=.03),
            'fusion_004': dict(fusion_penalty=.04),
            'l1_006': dict(l1_penalty=.06),
            'l1_008': dict(l1_penalty=.08),
            'l1_012': dict(l1_penalty=.12),
            'l1_014': dict(l1_penalty=.14),
            'l1_018': dict(l1_penalty=.18),
            'group_0002': dict(group_penalty=.002),
            'group_0005': dict(group_penalty=.005),
            'group_002': dict(group_penalty=.02),
            'group_003': dict(group_penalty=.03),
            'ridge_00001': dict(ridge_penalty=.00001),
            'ridge_0001': dict(ridge_penalty=.001),
            'ridge_001': dict(ridge_penalty=.01),
            'coupled_a': dict(l1_penalty=.08,group_penalty=.02),
            'coupled_b': dict(l1_penalty=.12,group_penalty=.005),
            'coupled_c': dict(l1_penalty=.15,group_penalty=.02),
            'coupled_d': dict(l1_penalty=.07,group_penalty=.005),
        }
        return {name:dict(base,**update) for name,update in variants.items()}
