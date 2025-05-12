# --- src/rfsim_core/__init__.py ---
import logging
from .log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
logger.info("RFSim Core package initialized.")

from .units import ureg, pint, Quantity
from .data_structures import Circuit, Component, Net, Port
from .parser import NetlistParser, ParsingError, SchemaValidationError
from .parameters import ParameterManager, ParameterError
from .components import (
    ComponentBase,
    Resistor,
    Capacitor,
    Inductor,
    COMPONENT_REGISTRY,
    ComponentError,
    LARGE_ADMITTANCE_SIEMENS 
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
from .simulation import (
    SimulationError, MnaInputError, SingularMatrixError,
    run_sweep, run_simulation
)


__all__ = [
    # Units
    "ureg", "pint", "Quantity",
    # Data Structures
    "Circuit", "Component", "Net", "Port",
    # Parser
    "NetlistParser", "ParsingError", "SchemaValidationError",
    # Parameters
    "ParameterManager", "ParameterError",
    # Components Base & Elements
    "ComponentBase", "Resistor", "Capacitor", "Inductor",
    "COMPONENT_REGISTRY", "ComponentError", "LARGE_ADMITTANCE_SIEMENS",
    # Builder
    "CircuitBuilder", "CircuitBuildError",
    # Validation
    "SemanticValidator", "SemanticValidationError", "ValidationIssue", 
    "ValidationIssueLevel", "SemanticIssueCode",
    # Simulation
    "SimulationError", "MnaInputError", "SingularMatrixError",
    "run_sweep", "run_simulation"
]