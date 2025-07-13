# src/rfsim_core/analysis/results.py
"""
Defines the formal, type-safe data contracts for the results of major analysis services.

This module is central to the "Explicit Contracts" architectural mandate. By defining
immutable, frozen dataclasses, we replace the previous use of raw, untyped dictionaries
for passing structured data. This provides several key advantages:

1.  **Type Safety:** The structure of the result is enforced by the Python type system,
    preventing an entire class of runtime errors caused by typos in dictionary keys or
    unexpected data types.

2.  **Self-Documentation:** The dataclass definition itself serves as clear, unambiguous
    documentation of what an analysis produces. Consumers can see exactly what fields
    are available and what their types are.

3.  **Immutability and Correctness:** Using `frozen=True` guarantees that a result object,
    once created and cached, cannot be accidentally modified by downstream code. This is a
    powerful tool for ensuring "Correctness by Construction" and preventing difficult-to-debug
    side effects.

4.  **Decoupling:** By making result objects self-contained (e.g., including all necessary
    metadata like `ground_supernode_name`), we decouple consumers of the result from the
    specific `DCAnalyzer` instance that created it.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Set

import numpy as np
import networkx as nx

from ..units import Quantity


@dataclass(frozen=True)
class DCAnalysisResults:
    """
    The formal, type-safe result of a DC analysis for a single circuit level.
    This object is the explicit contract between the DCAnalyzer and its consumers.
    It is immutable (frozen) to ensure correctness after creation.

    This dataclass is designed to be a fully self-contained record. 
    The inclusion of `ground_supernode_name` ensures that consumers of this 
    result do not need to refer back to the DCAnalyzer instance that created it, 
    achieving complete decoupling.
    """
    y_ports_dc: Optional[Quantity]
    dc_port_names_ordered: List[str]
    dc_port_mapping: Dict[str, Optional[int]]
    dc_supernode_mapping: Dict[str, str]
    ground_supernode_name: Optional[str]


@dataclass(frozen=True)
class TopologyAnalysisResults:
    """
    The formal, type-safe result of a topological analysis for a single circuit level.
    This object is the explicit contract between the TopologyAnalyzer and its consumers.
    It is immutable (frozen) to ensure correctness after creation.
    """
    structurally_open_components: Set[str]
    ac_graph: nx.Graph
    active_nets: Set[str]
    external_port_connectivity: List[Tuple[str, str]]