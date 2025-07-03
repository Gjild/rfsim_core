# src/rfsim_core/simulation/results.py
"""
Defines the formal, explicit, and type-safe data contracts for simulation results.

This module is central to the "Explicit Contracts" architectural mandate. By defining
immutable, frozen dataclasses, we replace the previous use of raw, untyped tuples
and dictionaries for passing structured data between the simulation engine and its
consumers (including the cache).
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np

# Import the formal contract from the analysis package to be used here.
# This creates a clear dependency: simulation results depend on analysis results.
from ..analysis.results import DCAnalysisResults


@dataclass(frozen=True)
class SubcircuitSimResults:
    """
    The formal, type-safe, cached result of a single-level subcircuit simulation.

    This object is the explicit contract between the recursive simulation populating
    the cache and the logic that consumes those cached results. Its attributes
    are type-safe and self-documenting. It is an *internal* contract used by the
    SimulationEngine.
    """
    y_parameters: np.ndarray
    dc_results: Optional[DCAnalysisResults]


@dataclass(frozen=True)
class SimulationResult:
    """
    The final, user-facing result of a top-level simulation sweep.

    This object provides the explicit, type-safe, public contract for the output of
    the main `run_sweep` function. By using a formal dataclass instead of a raw
    tuple, the API becomes self-documenting, easier to use correctly, and more
    resilient to future changes. Consumers of the simulation result will access
    data via named attributes (e.g., `result.y_parameters`), which is more robust
    and readable than index-based access (e.g., `result[1]`).

    Attributes:
        frequencies_hz: A 1D NumPy array of the frequency points (in Hz) at which
                        the simulation was run.
        y_parameters: A 3D NumPy array of the complex-valued N-port Y-parameters.
                      Shape is (num_frequencies, num_ports, num_ports).
        top_level_dc_results: An optional `DCAnalysisResults` object containing the
                              detailed results of the DC analysis performed on the
                              top-level circuit. It is `None` if the DC analysis
                              could not be completed or was not applicable.
    """
    frequencies_hz: np.ndarray
    y_parameters: np.ndarray
    top_level_dc_results: Optional[DCAnalysisResults]