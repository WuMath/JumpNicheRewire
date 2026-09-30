"""Read-only software checks; no public data or truth is loaded."""
import importlib.metadata
import sys


def environment_probe():
    import numpy, pandas, scipy, pyarrow, arboreto
    # Import the actual future worker as well, before registering any model jobs.
    from .worker import run_worker
    return dict(state='completed', python=sys.executable,
        versions={p: importlib.metadata.version(p) for p in ('numpy','pandas','scipy','pyarrow','arboreto')})
