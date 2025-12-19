from .core import _run_extraction, _run_builder, _run_graphrag


gcn_extractor = _run_extraction
gcn_builder = _run_builder
gcn_graphrag = _run_graphrag

__all__ = ["gcn_extractor", "gcn_builder", "gcn_graphrag"]