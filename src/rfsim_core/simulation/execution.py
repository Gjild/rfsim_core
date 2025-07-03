# src/rfsim_core/simulation/execution.py
"""
Provides the primary public API functions for running simulations.

**Architectural Refactoring (Phase 10):**

This module has been fundamentally refactored into a thin API wrapper that implements
the **Facade** design pattern. Its purpose is to provide a clean, simple, and powerful
entry point for the end-user, while completely hiding the internal complexity of the
underlying service-oriented architecture (`SimulationContext`, `SimulationEngine`,
`SimulationCache`, etc.).

The key responsibilities of this module are:
1.  **Expose `run_sweep`:** The primary, user-facing function for running a full
    hierarchical frequency sweep.
2.  **Encapsulate Setup:** It handles the instantiation and setup of the internal
    `SimulationContext` and `SimulationEngine` on behalf of the user.
3.  **Manage Cache Lifecycle:** It provides an elegant way for users to inject and
    reuse a `SimulationCache` object for performance-critical workflows, while
    defaulting to a new cache for simple use cases.
4.  **Enforce Explicit Contracts:** It returns a formal, type-safe `SimulationResult`
    object, ensuring a stable and self-documenting API.
5.  **Provide Top-Level Error Handling:** It wraps the entire simulation process in
    robust error handling, catching any `Diagnosable` exceptions from the core and
    presenting them to the user as a single, actionable `SimulationRunError`.

All complex, imperative simulation logic has been **relocated** from this file to the
new `simulation/engine.py` module, fulfilling the architectural mandate to separate
the public API from the internal implementation.
"""
import logging
import numpy as np
from typing import Optional, Tuple

# --- Core Data Structures & Services ---
from ..data_structures import Circuit
from ..cache import SimulationCache
from ..validation import SemanticValidator

# --- New Simulation Engine and Context ---
from .context import SimulationContext
from .engine import SimulationEngine

# --- Explicit Result and Exception Contracts ---
from .results import SimulationResult
from ..analysis.results import DCAnalysisResults  # For type hinting run_simulation
from ..errors import SimulationRunError, Diagnosable, format_diagnostic_report
from ..validation.exceptions import SemanticValidationError

logger = logging.getLogger(__name__)

def run_sweep(
    circuit: Circuit,
    freq_array_hz: np.ndarray,
    cache: Optional[SimulationCache] = None
) -> Tuple[SimulationResult, SimulationCache]:
    """
    The primary public API for running a hierarchical frequency sweep simulation.

    This function acts as a Facade, providing a simple entry point to the powerful
    underlying simulation engine and caching services.

    Args:
        circuit: The simulation-ready top-level Circuit object, as produced by the
                 `CircuitBuilder`.
        freq_array_hz: A 1D NumPy array of frequencies (in Hz) for the sweep.
        cache: An optional `SimulationCache` instance. If provided, its persistent,
               process-level cache will be reused, which can significantly speed up
               subsequent runs with similar subcircuits. If None (the default), a
               new, clean cache is created for this run.

    Returns:
        A tuple containing:
        - result: A formal `SimulationResult` object containing the final simulation
                  data (frequencies, Y-parameters, and DC results).
        - used_cache: The cache instance that was used for the run. This can be
                      captured and passed back into a subsequent call to `run_sweep`
                      to leverage the persistent process-level cache for performance.

    Raises:
        SimulationRunError: A user-friendly, diagnosable error if the simulation
                            fails at any stage (validation, analysis, or solve).
                            The original exception is chained for debugging.
    """
    # 1. Manage the cache lifecycle. Use the provided cache or create a new one.
    # This fulfills the goal of exposing the cache for advanced use cases.
    effective_cache = cache if cache is not None else SimulationCache()

    try:
        # 2. Perform pre-simulation semantic validation. This is a critical guard
        #    clause to prevent running simulations on logically invalid circuits.
        logger.info(f"--- Starting simulation sweep for '{circuit.name}' ---")
        validator = SemanticValidator(circuit)
        issues = validator.validate()
        # The SemanticValidationError is a 'Diagnosable' type, so it will be
        # caught and formatted correctly by the `except` block below.
        if any(issue.level == "ERROR" for issue in issues):
            raise SemanticValidationError(issues)

        # 3. Encapsulate the simulation's state and dependencies into the immutable context.
        context = SimulationContext(
            top_level_circuit=circuit,
            freq_array_hz=freq_array_hz,
            cache=effective_cache
        )

        # 4. Instantiate the stateless engine with the context. The engine contains
        #    all the "how-to" logic.
        engine = SimulationEngine(context)

        # 5. Execute the sweep. The engine handles all hierarchical complexity,
        #    caching, and analysis internally.
        y_mats, dc_res = engine.execute_sweep()

        # 6. Package the raw numerical results into the formal, user-facing contract.
        result = SimulationResult(
            frequencies_hz=freq_array_hz,
            y_parameters=y_mats,
            top_level_dc_results=dc_res
        )
        logger.info(f"Simulation sweep successful. Cache stats: {effective_cache.get_stats()}")

        # 7. Return the formal result and the cache instance, fulfilling the API contract.
        return result, effective_cache

    except Diagnosable as e:
        # 8a. Catch any well-defined, diagnosable error from the build or run process.
        #     Extract its detailed report and wrap it in the top-level user-facing error.
        logger.error(f"A diagnosable error occurred during simulation: {e}")
        raise SimulationRunError(e.get_diagnostic_report()) from e

    except Exception as e:
        # 8b. Catch all other unexpected errors for a graceful fallback report.
        #     This prevents unhandled exceptions from crashing the user's application.
        logger.critical(f"An unexpected internal error occurred during simulation: {e}", exc_info=True)
        report = format_diagnostic_report(
            error_type=f"An Unexpected Simulation Error Occurred ({type(e).__name__})",
            details=f"The simulator encountered an unexpected internal error: {e}",
            suggestion="This may be a bug. Review the traceback and consider filing a bug report.",
            context={}
        )
        raise SimulationRunError(report) from e


def run_simulation(
    circuit: Circuit,
    freq_hz: float,
    cache: Optional[SimulationCache] = None
) -> Tuple[SimulationResult, SimulationCache]:
    """
    A convenience wrapper around `run_sweep` for simulating at a single frequency point.

    This function returns a `SimulationResult` object containing data for the single
    frequency point, preserving the explicit contract of the primary `run_sweep` API.
    To access the Y-matrix, use `result.y_parameters[0]`.

    Args:
        circuit: The simulation-ready top-level Circuit object.
        freq_hz: The single frequency (in Hz) to simulate.
        cache: An optional `SimulationCache` instance to reuse.

    Returns:
        A tuple containing:
        - result: A formal `SimulationResult` object. Its `y_parameters` attribute
                  will have a shape of (1, num_ports, num_ports).
        - used_cache: The cache instance used for the run.
    """
    freq_array = np.array([freq_hz], dtype=float)

    # Call the primary API and return its results directly, preserving the formal
    # and explicit API contract. This function no longer betrays the architectural
    # philosophy by unpacking the result into a raw, anonymous tuple.
    sim_result, used_cache = run_sweep(circuit, freq_array, cache)

    return sim_result, used_cache