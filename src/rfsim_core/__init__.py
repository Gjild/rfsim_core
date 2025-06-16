# --- Modify: src/rfsim_core/__init__.py ---
import logging
from .log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
logger.info("RFSim Core package initialized.")

from .units import ureg, pint, Quantity, ADMITTANCE_DIMENSIONALITY, IMPEDANCE_DIMENSIONALITY
from .data_structures import Circuit, Component, Net, Port
from .parser import NetlistParser, ParsingError, SchemaValidationError
from .parameters import ParameterManager, ParameterError
# --- Import constants module itself, but don't export by default ---
from . import constants
from .components import (
    ComponentBase,
    Resistor,
    Capacitor,
    Inductor,
    COMPONENT_REGISTRY,
    ComponentError,
    # LARGE_ADMITTANCE_SIEMENS # Removed from here
    DCBehaviorType
)
from .circuit_builder import CircuitBuilder, CircuitBuildError
# Import new validation classes
from .validation import (
    SemanticValidator,
    SemanticValidationError,
    ValidationIssue,
    ValidationIssueLevel,
    SemanticIssueCode
)
from .analysis_tools import (
    DCAnalyzer,
    TopologyAnalyzer,
    DCAnalysisError,
    TopologyAnalysisError
)
from .simulation import (
    SimulationError, MnaInputError, SingularMatrixError, MnaAssembler,
    run_sweep, run_simulation
)


__all__ = [
    # Units
    "ureg", "pint", "Quantity",
    # Canonical dimensionalities
    "ADMITTANCE_DIMENSIONALITY", "IMPEDANCE_DIMENSIONALITY",
    # Data Structures
    "Circuit", "Component", "Net", "Port",
    # Parser
    "NetlistParser", "ParsingError", "SchemaValidationError",
    # Parameters
    "ParameterManager", "ParameterError",
    # Components Base & Elements
    "ComponentBase", "Resistor", "Capacitor", "Inductor",
    "COMPONENT_REGISTRY", "ComponentError", # LARGE_ADMITTANCE_SIEMENS no longer exported here
    "DCBehaviorType",
    # Builder
    "CircuitBuilder", "CircuitBuildError",
    # Validation
    "SemanticValidator", "SemanticValidationError", "ValidationIssue",
    "ValidationIssueLevel", "SemanticIssueCode",
    # Analysis Tools
    "DCAnalyzer", "TopologyAnalyzer", "DCAnalysisError", "TopologyAnalysisError",
    # Simulation
    "SimulationError", "MnaInputError", "SingularMatrixError",
    "run_sweep", "run_simulation"
    # "constants" # Not exported by default, users import directly if needed
]