# src/rfsim_core/simulation/execution.py
import logging
import numpy as np
import scipy.sparse as sp
from typing import Tuple

from .mna import MnaAssembler, MnaInputError
from .solver import solve_mna, SingularMatrixError
from ..data_structures import Circuit
from .. import ureg, Quantity, ComponentError

logger = logging.getLogger(__name__)

class SimulationError(Exception):
    """General error during simulation execution."""
    pass

def run_simulation(circuit: Circuit, freq_hz: float) -> np.ndarray:
     """Runs a single-frequency simulation."""
     # Wrapper around run_sweep for a single frequency
     freq_array = np.array([freq_hz])
     frequencies, results = run_sweep(circuit, freq_array)
     return results[0, :, :] # Return the first (only) result

def run_sweep(circuit: Circuit, freq_array_hz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs a frequency sweep simulation and calculates the N-port Y-matrices.

    Args:
        circuit: The Circuit object, processed by CircuitBuilder.
        freq_array_hz: A NumPy array of simulation frequencies in Hz.

    Returns:
        A tuple containing:
            - frequencies_hz: The input frequency array used (np.ndarray).
            - y_matrices: 3D NumPy array (NumFreqs x NumPorts x NumPorts)
                          containing the Y-matrix for each frequency.

    Raises:
        SimulationError: If simulation fails for any frequency point.
        MnaInputError: If frequency array is invalid or circuit not ready.
        SingularMatrixError: If the MNA matrix is singular at any frequency.
    """
    if not isinstance(freq_array_hz, np.ndarray) or freq_array_hz.ndim != 1:
         raise MnaInputError("Frequency sweep must be provided as a 1D NumPy array.")
    if len(freq_array_hz) == 0:
         raise MnaInputError("Frequency sweep array cannot be empty.")
    # Check if circuit is built
    if not isinstance(circuit, Circuit) or not hasattr(circuit, 'sim_components'):
         raise MnaInputError("Circuit object must be processed by CircuitBuilder first.")

    num_freqs = len(freq_array_hz)
    logger.info(f"--- Starting frequency sweep for '{circuit.name}' ({num_freqs} points) ---")

    # Initialize result array - need number of ports first
    # Get port info from a temporary assembler instance (or refactor assembler)
    print(np.any(freq_array_hz > 0))
    try:
         # Use first valid AC frequency (>0) to initialize assembler for port info
         first_ac_freq = freq_array_hz[freq_array_hz > 0][0]
         temp_assembler = MnaAssembler(circuit, first_ac_freq) # Need F>0 for constructor
         num_ports = len(temp_assembler.port_indices)
         port_node_indices = temp_assembler.port_indices # Cache port indices
         port_names = temp_assembler.port_names
    except IndexError: # No AC frequencies found
         if not np.any(freq_array_hz > 0):
             raise MnaInputError("Cannot run AC sweep simulation with only F=0 Hz points.")
    except MnaInputError as e:
         logger.error(f"Failed to initialize assembler for port info: {e}")
         raise

    if num_ports == 0:
        logger.warning("Circuit has no external ports. Returning empty results.")
        return freq_array_hz, np.array([], dtype=np.complex128).reshape(num_freqs, 0, 0)

    y_matrices = np.empty((num_freqs, num_ports, num_ports), dtype=np.complex128)
    node_count = temp_assembler.node_count # Assumes node count doesn't change

    # --- Frequency Loop ---
    for idx, freq in enumerate(freq_array_hz):
        logger.info(f"  Simulating point {idx+1}/{num_freqs} at {freq:.4e} Hz...")

        # Handle F=0: Skip AC MNA solution for F=0 points in this phase
        if freq <= 0:
            logger.warning(f"Skipping MNA solve for F={freq} Hz. DC analysis needed.")
            # Fill results with NaN or some indicator?
            y_matrices[idx, :, :] = np.nan + 0j # Indicate skipped point
            continue

        try:
            # 1. Assemble MNA matrix (for this frequency)
            # Re-create assembler for each freq - no caching yet
            # Use cached port info from temp_assembler if needed, but MnaAssembler recalculates it
            assembler = MnaAssembler(circuit, freq)
            # Consistency check (optional): Ensure node count matches temp_assembler
            if assembler.node_count != node_count:
                raise SimulationError(f"Node count mismatch at freq={freq} Hz!")
            Yn_full = assembler.assemble()

            # 2. Calculate Z-matrix column by column
            z_matrix = np.zeros((num_ports, num_ports), dtype=np.complex128)
            for k in range(num_ports): # k is the index of the port being excited
                port_k_node_idx = port_node_indices[k]

                # Create excitation vector I: 1A at port k, 0 otherwise
                I_full = np.zeros(node_count, dtype=np.complex128)
                I_full[port_k_node_idx] = 1.0 + 0.0j

                # Solve MNA system: Yn * V = I
                V_full = solve_mna(Yn_full, I_full)

                # Extract voltages at all port nodes for this excitation
                V_ports_k = V_full[port_node_indices]
                z_matrix[:, k] = V_ports_k

            # 3. Calculate Y-matrix from Z-matrix
            try:
                y_matrix_freq = np.linalg.inv(z_matrix)
            except np.linalg.LinAlgError as e:
                 logger.error(f"Failed to invert Z-matrix at {freq} Hz: {e}. Matrix might be singular.")
                 # Fill with NaN and continue sweep? Or raise immediately? Let's fill.
                 y_matrices[idx, :, :] = np.nan + 0j
                 logger.warning(f"Setting Y-matrix to NaN for {freq} Hz due to Z-inversion failure.")
                 continue # Go to next frequency point

            # Store the result for this frequency
            y_matrices[idx, :, :] = y_matrix_freq
            logger.debug(f"  Y-matrix calculated for {freq:.4e} Hz.")

        except (MnaInputError, SingularMatrixError, ComponentError) as e:
             logger.error(f"Simulation failed at {freq} Hz: {e}", exc_info=False) # Less verbose logging in sweep
             # Mark results as NaN for this frequency
             y_matrices[idx, :, :] = np.nan + 0j
             logger.warning(f"Setting Y-matrix to NaN for {freq} Hz due to simulation error.")
             # Optionally: could add a flag to stop sweep on first error
             continue # Continue sweep unless configured otherwise
        except Exception as e:
             logger.error(f"Unexpected error during simulation at {freq} Hz: {e}", exc_info=True)
             y_matrices[idx, :, :] = np.nan + 0j
             logger.warning(f"Setting Y-matrix to NaN for {freq} Hz due to unexpected error.")
             continue

    logger.info(f"--- Frequency sweep finished for '{circuit.name}' ---")
    return freq_array_hz, y_matrices