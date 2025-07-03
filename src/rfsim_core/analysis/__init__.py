# src/rfsim_core/analysis/__init__.py
"""
Defines the public interface for the analysis services package.

This package encapsulates all major pre-simulation analysis tools (like DC and Topology)
and their formal result contracts. By exposing these through a single, clean __init__.py,
we provide a clear and stable API for other parts of the simulation core to use.
"""
from .results import DCAnalysisResults, TopologyAnalysisResults
from .tools import DCAnalyzer, TopologyAnalyzer
from .exceptions import DCAnalysisError, TopologyAnalysisError

__all__ = [
    # Formal Result Contracts
    "DCAnalysisResults",
    "TopologyAnalysisResults",
    # Cache-Aware Analysis Services
    "DCAnalyzer",
    "TopologyAnalyzer",
    # Exceptions
    "DCAnalysisError",
    "TopologyAnalysisError",
]