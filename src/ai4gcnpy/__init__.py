from .core import _run_extraction, _run_builder


gcn_extractor = _run_extraction
gcn_builder = _run_builder

__all__ = ["gcn_extractor", "gcn_builder"]