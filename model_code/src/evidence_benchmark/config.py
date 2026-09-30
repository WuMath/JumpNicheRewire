"""Compatibility imports; all settings are under configs/."""
from configs.common.evidence_paths import *
from configs.public_sim_data.settings import *
from configs.real_data.pseudotruth_settings import *
from configs.perturbation.settings import *
from configs.common.evidence_settings import *
CATALOG_CONFIG = CatalogConfig()

INSTALL_CONFIG = InstallConfig()

SERGIO_CONFIG = SergioConfig()

SCCIGEN_CONFIG = SCCIGenConfig()

REAL_CONFIG = RealPseudoTruthConfig()

PERTURBATION_CONFIG = PerturbationConfig()

STEREO_CONVERSION_CONFIG = StereoGEFConversionConfig()

MODEL_CONFIG = EvidenceModelConfig()

EVALUATION_CONFIG = EvidenceEvaluationConfig()

EXTERNAL_IMPORT_CONFIG = ExternalImportConfig()

JOB_CONFIG = EvidenceJobConfig()

