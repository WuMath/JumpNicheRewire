"""Second timed search under the fixed positive spatial-fusion constraint."""
from dataclasses import dataclass
from pathlib import Path
from .single_dataset_jnr_tuning import OneHourJNR, ROOT


@dataclass(frozen=True)
class PositiveFusionFollowupJNR(OneHourJNR):
    output_root: Path = ROOT/'data/result/optimization_20260921/single_dataset_jnr_positive_fusion_followup_20260924'

    def candidates(self):
        base = dict(l1_penalty=.08, fusion_penalty=.0005, group_penalty=.01,
                    ridge_penalty=.0001, aggregation='mean_abs')
        variants = {
            'l1_0075': dict(l1_penalty=.075),
            'l1_007': dict(l1_penalty=.07),
            'l1_0065': dict(l1_penalty=.065),
            'l1_006': dict(l1_penalty=.06),
            'l1_005': dict(l1_penalty=.05),
            'l1_004': dict(l1_penalty=.04),
            'l1_00825': dict(l1_penalty=.0825),
            'l1_0085': dict(l1_penalty=.085),
            'l1_00875': dict(l1_penalty=.0875),
            'fusion_0006': dict(fusion_penalty=.0006),
            'fusion_00075': dict(fusion_penalty=.00075),
            'fusion_001': dict(fusion_penalty=.001),
            'fusion_00125': dict(fusion_penalty=.00125),
            'fusion_0015': dict(fusion_penalty=.0015),
            'group_0005': dict(group_penalty=.005),
            'group_00075': dict(group_penalty=.0075),
            'group_00125': dict(group_penalty=.0125),
            'group_0015': dict(group_penalty=.015),
            'ridge_00005': dict(ridge_penalty=.00005),
            'ridge_0002': dict(ridge_penalty=.0002),
            'coupled_0075_00075': dict(l1_penalty=.075, group_penalty=.0075),
            'coupled_0085_fusion_00075': dict(l1_penalty=.085, fusion_penalty=.00075),
        }
        candidates = {name: dict(base, **change) for name, change in variants.items()}
        if len(candidates) != 22 or any(p['fusion_penalty'] < .0005 for p in candidates.values()):
            raise ValueError('Positive-fusion follow-up grid invalid')
        return candidates
