"""两个入口共用此选择器。只修改类字段，不传命令行参数。"""

from __future__ import annotations
from dataclasses import dataclass
from configs.benchmark_selected import SELECTED


@dataclass
class ActiveConfig:
    # public_sim_data.sergio / public_sim_data.sccigen / public_sim_data.scmultisim
    # real_data.visium / real_data.h5ad
    # perturbation.spatial_perturbseq
    # self_sim_data.demo / self_sim_data.pilot / self_sim_data.core_final
    profile: str = "benchmark_selected"
    # main: check / run / status / evaluate / jobs / import_external
    # self_sim_data.pilot/core_final 还支持 tune / freeze / test / audit
    main_action: str = SELECTED.action
    # preprocess: check / prepare / reuse
    # 具体配置另支持 simulate/install/catalog/references/convert；见 README。
    preprocess_action: str = "check"


ACTIVE = ActiveConfig()
