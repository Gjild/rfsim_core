# src/rfsim_core/__init__.py
import logging
from .log_config import setup_logging

# Configure logging when the package is imported
setup_logging()

# Example: Get a logger for this specific module
logger = logging.getLogger(__name__)
logger.info("RFSim Core package initialized.")

# Expose key classes and functions for easier access
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
)
from .circuit_builder import CircuitBuilder, CircuitBuildError
from .simulation import run_simulation, SimulationError, MnaInputError, SingularMatrixError

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
    "COMPONENT_REGISTRY", "ComponentError",
    # Builder
    "CircuitBuilder", "CircuitBuildError",
    # Simulation
    "SimulationError", "MnaInputError", "SingularMatrixError"
]