"""Restartable benchmark orchestration for JumpNicheRewire.

The package deliberately separates data preparation, method execution, and
analysis.  Each stage writes its own immutable manifests and completion marker.
"""

__version__ = "0.1.0"

