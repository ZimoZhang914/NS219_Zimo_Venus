"""Microglia annotation pipeline (VINE-seq, AD).

Heavy dependencies (torch, scanpy) are imported lazily — ``Config`` and
``load_config`` are usable on machines without them installed.
"""

from .config import Config, load_config

__version__ = "0.1.0"

__all__ = ["Config", "load_config", "run_pipeline"]


def __getattr__(name):
    # PEP 562 — lazy attribute access at module level
    if name == "run_pipeline":
        from .pipeline import run_pipeline
        return run_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
