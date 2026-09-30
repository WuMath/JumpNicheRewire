from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from grn2_project.paths import *

@dataclass
class ReferenceConfig:
    tf_target_file: Path = PROJECT_ROOT / "data" / "raw" / "ModelData" / "Mouse-TF-Target.tsv"
    ligand_receptor_file: Path = PROJECT_ROOT / "data" / "raw" / "ModelData" / "Mouse-ligand-receptor.tsv"
    # Versioned clean output; the earlier ``references`` directory may contain
    # unresolved UniProt accessions and is retained only as an audit copy.
    output_dir: Path = PROJECT_ROOT / "data" / "processed" / "references"
    organism: str = "mouse"
    taxonomy_id: int = 10090
    # True: remove unresolved A0A.../P12345 UniProt accessions from gene-symbol fields.
    # False is only for identifier-mapping audits; do not use that output as model input.
    exclude_unmapped_uniprot_accessions: bool = True


CONFIG = ReferenceConfig()

