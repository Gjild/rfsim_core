# --- src/rfsim_core/simulation/execution.py ---
import logging
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from typing import Tuple

from .mna import MnaAssembler, MnaInputError
from .solver import factorize_mna_matrix, solve_mna_system, SingularMatrixError
from ..data_structures import Circuit
from .. import ureg, Quantity, ComponentError, circuit_builder # Added circuit_builder for type hint

logger = logging.getLogger(__name__)

class SimulationError(Exception):
    """General error during simulation execution."""
    pass

def run_simulation(circuit: Circuit, freq_hz: float) -> np.ndarray:
     """
     Runs a single-frequency AC simulation (F > 0) and returns the
     intrinsic N-port Y-matrix.

     Args:
         circuit: The Circuit object, processed by CircuitBuilder.
         freq_hz: The single simulation frequency in Hz (must be > 0).

     Returns:
         The N-port intrinsic Y-matrix (NumPorts x NumPorts) at the specified frequency.

     Raises:
         SimulationError, MnaInputError, SingularMatrixError.
     """
     if freq_hz < 0: # Allow F=0
         raise MnaInputError(f"Single frequency simulation requires freq_hz >= 0. Got {freq_hz}.")
     if freq_hz == 0:
         logger.warning("Running run_simulation at F=0. Result uses component F=0 admittance, NOT rigorous DC analysis.")

     freq_array = np.array([freq_hz])
     # Pass the already built circuit
     frequencies, results = run_sweep(circuit, freq_array)
     # result shape is (1, num_ports, num_ports)
     if results.shape[0] > 0:
        return results[0, :, :]
     else: # Handle case where sweep might return empty results (e.g., no ports)
        num_ports = len(circuit.external_ports) # Get port count from circuit data
        return np.empty((num_ports, num_ports), dtype=np.complex128)

def run_sweep(circuit: Circuit, freq_array_hz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs a frequency sweep AC simulation (all frequencies must be > 0) and
    calculates the intrinsic N-port Y-matrices using Schur complement reduction.

    Args:
        circuit: The Circuit object, processed by CircuitBuilder.
        freq_array_hz: A 1D NumPy array of simulation frequencies in Hz (all must be > 0).

    Returns:
        A tuple containing:
            - frequencies_hz: The input frequency array used (np.ndarray).
            - y_matrices: 3D NumPy array (NumFreqs x NumPorts x NumPorts)
                          containing the intrinsic Y-matrix for each frequency.

    Raises:
        SimulationError, MnaInputError, SingularMatrixError, ComponentError.
    """
    if not isinstance(freq_array_hz, np.ndarray) or freq_array_hz.ndim != 1:
         raise MnaInputError("Frequency sweep must be provided as a 1D NumPy array.")
    if len(freq_array_hz) == 0:
         logger.warning("Frequency sweep array is empty. Returning empty results.")
         # Need to know num_ports to return correct shape empty array
         # Get from circuit data structure BEFORE assembler is created
         num_ports_from_circuit = len(circuit.external_ports)
         return freq_array_hz, np.array([], dtype=np.complex128).reshape(0, num_ports_from_circuit, num_ports_from_circuit)

    # Allow F=0 in sweep
    if np.any(freq_array_hz < 0):
         raise MnaInputError(f"All frequencies in the sweep must be >= 0 Hz. Found min value: {np.min(freq_array_hz)} Hz.")
    if np.any(freq_array_hz == 0):
        logger.warning("Frequency sweep includes F=0. Results at F=0 use component F=0 admittance, NOT rigorous DC analysis.")

    # Expect a simulation-ready circuit. Check if it looks like a built circuit (has sim_components attribute).
    if not isinstance(circuit, Circuit) or not hasattr(circuit, 'sim_components'):
         raise MnaInputError("Input circuit object does not appear to be simulation-ready (missing 'sim_components'). Did you run CircuitBuilder.build_circuit()?")

    num_freqs = len(freq_array_hz)
    logger.info(f"--- Starting frequency sweep for '{circuit.name}' ({num_freqs} points from {np.min(freq_array_hz):.3e} to {np.max(freq_array_hz):.3e} Hz) ---")
    logger.info("Calculating intrinsic Y-matrix using Schur complement.")

    # 1. Initialize MNA Assembler (once)
    try:
         assembler = MnaAssembler(circuit) # Frequency independent init
         num_ports = len(assembler.port_indices)
         node_count = assembler.node_count
         ext_indices_reduced = assembler.external_node_indices_reduced # Indices of ports in the reduced system
         int_indices_reduced = assembler.internal_node_indices_reduced # Indices of internal nodes in the reduced system
         num_internal_nodes = len(int_indices_reduced)
         reduced_dim = node_count - 1

         # Validate indices
         if len(ext_indices_reduced) != num_ports:
              raise SimulationError(f"Internal Error: Mismatch between external port count ({num_ports}) and reduced external indices ({len(ext_indices_reduced)}).")
         if len(ext_indices_reduced) + len(int_indices_reduced) != reduced_dim:
             raise SimulationError(f"Internal Error: Sum of external ({len(ext_indices_reduced)}) and internal ({len(int_indices_reduced)}) reduced indices does not match reduced dimension ({reduced_dim}).")

    except MnaInputError as e:
         logger.error(f"Failed to initialize MNA Assembler: {e}")
         raise

    if num_ports == 0 and reduced_dim == 0:
        logger.warning(f"Circuit '{circuit.name}' has no external ports and no internal nodes (after ground removal). Returning empty results.")
        return freq_array_hz, np.array([], dtype=np.complex128).reshape(num_freqs, 0, 0)
    elif reduced_dim == 0 and num_ports > 0:
         logger.error(f"Circuit '{circuit.name}' has {num_ports} ports but reduced system dimension is 0.")
         raise SimulationError(f"Circuit '{circuit.name}' has {num_ports} ports but a 0-dimensional reduced system. Check connections.")
    elif num_ports == 0:
         logger.warning(f"Circuit '{circuit.name}' has no external ports. Simulation will run but result matrix will be 0x0.")
         y_matrices = np.empty((num_freqs, 0, 0), dtype=np.complex128)
    else: # Normal case: num_ports > 0 and reduced_dim > 0
        y_matrices = np.empty((num_freqs, num_ports, num_ports), dtype=np.complex128)

    # --- Frequency Loop ---
    for idx, freq in enumerate(freq_array_hz):
        # Add specific F=0 log message
        freq_str = f"{freq:.4e} Hz"
        if freq == 0:
            freq_str = "0 Hz (DC - component value eval only)"
        logger.info(f"  Simulating point {idx+1}/{num_freqs} at {freq_str}...")
        # Initialize default result for this frequency point
        Y_intrinsic_current = np.full((num_ports, num_ports), np.nan + 0j, dtype=np.complex128)

        try:
            # 2. Assemble MNA matrix (uses cached sparsity)
            Yn_full = assembler.assemble(freq) # Pass single frequency value

            # 3. Reduce matrix (remove ground row/col)
            if node_count <= 1:
                 if num_ports > 0:
                      raise SimulationError(f"Circuit reduced to single node (ground) but {num_ports} ports are defined.")
                 else: # No ports, only ground: empty result is correct
                      Y_intrinsic_current = np.empty((0,0), dtype=np.complex128)
                      # Assign to y_matrices correctly based on num_ports
                      if num_ports == 0: y_matrices[idx] = Y_intrinsic_current # Store empty 0x0 matrix
                      continue # Skip to next frequency

            Yn_reduced = Yn_full[1:, 1:].tocsc() # Ensure CSC

            if Yn_reduced.shape[0] != reduced_dim or Yn_reduced.shape[1] != reduced_dim:
                raise SimulationError(f"Internal Error: Reduced matrix shape {Yn_reduced.shape} does not match expected dimension {reduced_dim}.")

            # 4. Partition the Reduced Matrix (unchanged logic)
            Y_EE = Yn_reduced[np.ix_(ext_indices_reduced, ext_indices_reduced)]
            if num_internal_nodes > 0 and num_ports > 0:
                 Y_EI = Yn_reduced[np.ix_(ext_indices_reduced, int_indices_reduced)]
                 Y_IE = Yn_reduced[np.ix_(int_indices_reduced, ext_indices_reduced)]
            else:
                 Y_EI = sp.csc_matrix((num_ports, num_internal_nodes), dtype=np.complex128)
                 Y_IE = sp.csc_matrix((num_internal_nodes, num_ports), dtype=np.complex128)

            if num_internal_nodes > 0:
                 Y_II = Yn_reduced[np.ix_(int_indices_reduced, int_indices_reduced)]
            else:
                 Y_II = sp.csc_matrix((0, 0), dtype=np.complex128)

            # 5. Calculate Schur Complement (unchanged logic)
            if num_internal_nodes == 0:
                 if num_ports > 0:
                    Y_intrinsic_current = Y_EE.toarray()
                 else:
                    Y_intrinsic_current = np.empty((0,0), dtype=np.complex128)
            else:
                 try:
                    lu_II = factorize_mna_matrix(Y_II)
                 except SingularMatrixError as fact_err:
                     logger.error(f"Factorization of internal matrix Y_II failed at {freq_str}: {fact_err}")
                     # Keep Y_intrinsic_current as NaN (default) and continue
                     if num_ports > 0: y_matrices[idx] = Y_intrinsic_current
                     continue

                 try:
                     if Y_IE.nnz > 0:
                         X = lu_II.solve(Y_IE.toarray())
                     else:
                         X = np.zeros((num_internal_nodes, num_ports), dtype=np.complex128)

                     if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                         raise SingularMatrixError("Solution X for internal nodes contains NaN/Inf.")

                 except (SingularMatrixError, RuntimeError, ValueError) as solve_err:
                      logger.error(f"Failed to solve internal node system Y_II*X = Y_IE at {freq_str}: {solve_err}")
                      # Keep Y_intrinsic_current as NaN (default) and continue
                      if num_ports > 0: y_matrices[idx] = Y_intrinsic_current
                      continue

                 if num_ports > 0:
                     Y_intrinsic_current = Y_EE.toarray() - (Y_EI @ X)
                 else:
                     Y_intrinsic_current = np.empty((0,0), dtype=np.complex128)

            # 6. Store the resulting intrinsic Y-matrix
            if num_ports >= 0: # Check >=0 to handle 0x0 case
                 if Y_intrinsic_current.shape != (num_ports, num_ports):
                      raise SimulationError(f"Internal Error: Calculated Y_intrinsic shape {Y_intrinsic_current.shape} mismatch with port count {num_ports}.")
                 y_matrices[idx] = np.asarray(Y_intrinsic_current, dtype=np.complex128)
                 logger.debug(f"  Intrinsic Y-matrix calculated for {freq_str}.")

        except (MnaInputError, SingularMatrixError, ComponentError, SimulationError) as e:
             logger.error(f"Simulation failed at {freq_str}: {e}", exc_info=False)
             # Store NaN matrix (already default)
             if num_ports > 0: y_matrices[idx] = Y_intrinsic_current
             continue # Continue sweep
        except Exception as e:
             logger.error(f"Unexpected error during simulation at {freq_str}: {e}", exc_info=True)
             if num_ports > 0: y_matrices[idx] = Y_intrinsic_current
             continue

    logger.info(f"--- Frequency sweep finished for '{circuit.name}' ---")
    if np.any(np.isnan(y_matrices)):
         logger.warning(f"Sweep for '{circuit.name}' completed, but one or more frequency points failed (results contain NaN).")

    return freq_array_hz, y_matrices