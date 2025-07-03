# src/rfsim_core/data_structures.py
# Required for forward references in type hints (e.g., 'ComponentBase')
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING
from pathlib import Path

logger = logging.getLogger(__name__)

# Use TYPE_CHECKING to avoid circular imports for type hints at runtime.
if TYPE_CHECKING:
    from .parameters import ParameterManager
    from .components.base import ComponentBase
    from .parser.raw_data import ParsedCircuitNode


@dataclass(frozen=True)
class Net:
    """
    Represents an electrical node (net) in the final, synthesized circuit model.
    This object is immutable after creation by the CircuitBuilder.
    """
    name: str
    is_ground: bool = False
    is_external: bool = False
    # The 'index' field for MNA is intentionally omitted here. It is transient,
    # context-dependent state that belongs to the MnaAssembler, not the
    # fundamental circuit data model.

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, Net):
            return NotImplemented
        return self.name == other.name


@dataclass(frozen=True)
class Circuit:
    """
    Represents the final, synthesized hierarchical circuit model.

    This object is the definitive, simulation-ready representation of the circuit,
    produced by the CircuitBuilder from the parser's Intermediate Representation (IR).
    It is a data container and holds no imperative logic.
    """
    name: str
    hierarchical_id: str
    source_file_path: Path
    ground_net_name: str

    # The collection of all nets/nodes in this circuit's scope.
    nets: Dict[str, Net]

    # The collection of simulation-ready component instances (R, L, C, SubcircuitInstance).
    sim_components: Dict[str, ComponentBase]

    # The external interface of this circuit. The key is the port name.
    external_ports: Dict[str, Net]

    # The single, global parameter manager for the entire simulation.
    parameter_manager: ParameterManager

    # A non-negotiable link back to the root of the raw IR tree that was used
    # to synthesize this Circuit object. This is essential for validation and diagnostics.
    raw_ir_root: ParsedCircuitNode