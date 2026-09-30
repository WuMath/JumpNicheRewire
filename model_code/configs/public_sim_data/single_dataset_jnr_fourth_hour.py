"""Fourth fixed one-hour JNR-only search on the same development dataset."""
from dataclasses import dataclass
from pathlib import Path
from .single_dataset_jnr_tuning import OneHourJNR, ROOT


@dataclass(frozen=True)
class FourthHourJNR(OneHourJNR):
    output_root: Path = ROOT/'data/result/optimization_20260921/single_dataset_jnr_fourth_hour_20260924'

    def candidates(self):
        base = dict(l1_penalty=.1, fusion_penalty=0., group_penalty=.01,
                    ridge_penalty=.0001, aggregation='mean_abs')
        variants = {
            'l1_011': dict(l1_penalty=.11),
            'l1_012': dict(l1_penalty=.12),
            'l1_013': dict(l1_penalty=.13),
            'l1_014': dict(l1_penalty=.14),
            'l1_015': dict(l1_penalty=.15),
            'l1_017': dict(l1_penalty=.17),
            'l1_020': dict(l1_penalty=.20),
            'l1_025': dict(l1_penalty=.25),
            'l1_030': dict(l1_penalty=.30),
            'group_0007': dict(group_penalty=.007),
            'group_0008': dict(group_penalty=.008),
            'group_0009': dict(group_penalty=.009),
            'group_0011': dict(group_penalty=.011),
            'group_0012': dict(group_penalty=.012),
            'group_0014': dict(group_penalty=.014),
            'group_0018': dict(group_penalty=.018),
            'group_0020': dict(group_penalty=.020),
            'ridge_00002': dict(ridge_penalty=.00002),
            'ridge_00005': dict(ridge_penalty=.00005),
            'ridge_0002': dict(ridge_penalty=.0002),
            'coupled_012_0012': dict(l1_penalty=.12, group_penalty=.012),
            'coupled_015_0012': dict(l1_penalty=.15, group_penalty=.012),
        }
        return {name: dict(base, **change) for name, change in variants.items()}
