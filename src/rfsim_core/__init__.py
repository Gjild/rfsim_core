# src/rfsim_core/__init__.py
import logging
from .log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
logger.info("RFSim Core package initialized.")

from .units import ureg, pint, Quantity, ADMITTANCE_DIMENSIONALITY, IMPEDANCE_DIMENSIONALITY
from .data_structures import Circuit
from .parser import NetlistParser
from .circuit_builder import CircuitBuilder
from .simulation import run_sweep, run_simulation
from .errors import RFSimError, CircuitBuildError, SimulationRunError

__all__ = [
    # Units
    "ureg", "pint", "Quantity",
    # Canonical dimensionalities
    "ADMITTANCE_DIMENSIONALITY", "IMPEDANCE_DIMENSIONALITY",
    # Data Structures
    "Circuit",
    # Parser
    "NetlistParser",
    # Builder
    "CircuitBuilder",
    # Simulation
    "run_sweep", "run_simulation",
    # Top-Level Errors (Actionable Diagnostics)
    "RFSimError", "CircuitBuildError", "SimulationRunError",
]