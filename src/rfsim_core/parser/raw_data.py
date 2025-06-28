# src/rfsim_core/parser/raw_data.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

# The classes in this module define the Intermediate Representation (IR).
# This IR establishes the formal, type-safe, non-negotiable contract between
# the NetlistParser and the CircuitBuilder. Using frozen dataclasses abolishes
# the use of raw dictionaries for passing structured data, which is a foundational
# tenet of the definitive architecture for correctness and maintainability.

@dataclass(frozen=True)
class ParsedLeafComponentData:
    """IR for a leaf-level component (e.g., R, L, C, or a future plugin component)."""
    instance_id: str
    component_type: str
    raw_ports_dict: Dict[str | int, str]
    raw_parameters_dict: Dict[str, Any]
    source_yaml_path: Path

@dataclass(frozen=True)
class ParsedSubcircuitData:
    """IR for a subcircuit instance."""
    instance_id: str
    component_type: str  # Always "Subcircuit"
    definition_file_path: Path
    sub_circuit_definition_node: ParsedCircuitNode
    raw_port_mapping: Dict[str, str]
    raw_parameter_overrides: Dict[str, Any]
    source_yaml_path: Path

ParsedComponentData = Union[ParsedLeafComponentData, ParsedSubcircuitData]

@dataclass(frozen=True)
class ParsedCircuitNode:
    """
    Top-level IR node representing a single parsed YAML file.
    This forms a tree structure for hierarchical designs.
    """
    circuit_name: str
    ground_net_name: str
    source_yaml_path: Path
    components: List[ParsedComponentData]
    raw_parameters_dict: Dict[str, Any]
    raw_external_ports_list: List[Dict[str, Any]]
    raw_sweep_config: Optional[Dict[str, Any]] = None