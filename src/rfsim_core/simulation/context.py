# src/rfsim_core/simulation/context.py
"""
Defines the `SimulationContext`, a cornerstone of the service-oriented architecture.
"""
from dataclasses import dataclass
import numpy as np

from ..data_structures import Circuit
from ..cache.service import SimulationCache


@dataclass(frozen=True)
class SimulationContext:
    """
    An immutable container for the complete state of a single simulation run.

    This object represents the "what" of the simulation—the complete, unambiguous
    source of truth for all inputs (the circuit, the frequencies) and shared
    resources (the cache). It is passed to the stateless SimulationEngine, which
    operates on it.

    Its immutability, enforced by `frozen=True`, is a powerful feature that
    guarantees a simulation's initial conditions cannot be altered mid-execution.
    This fulfills a key aspect of the "Correctness by Construction" mandate by
    eliminating an entire class of potential side-effect bugs.
    """
    top_level_circuit: Circuit
    freq_array_hz: np.ndarray
    cache: SimulationCache