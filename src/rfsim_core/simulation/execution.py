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
     if not freq_hz > 0:
         raise MnaInputError(f"Single frequency simulation requires freq_hz > 0. Got {freq_hz}.")
     freq_array = np.array([freq_hz])
     frequencies, results = run_sweep(circuit, freq_array)
     # result shape is (1, num_ports, num_ports)
     return results[0, :, :]

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
         raise MnaInputError("Frequency sweep array cannot be empty.")
    if np.any(freq_array_hz <= 0):
         raise MnaInputError(f"All frequencies in the sweep must be > 0 Hz for AC analysis. Found min value: {np.min(freq_array_hz)} Hz.")
    if not isinstance(circuit, Circuit) or not hasattr(circuit, 'sim_components'):
         raise MnaInputError("Circuit object must be processed by CircuitBuilder first.")

    num_freqs = len(freq_array_hz)
    logger.info(f"--- Starting frequency sweep for '{circuit.name}' ({num_freqs} points from {freq_array_hz[0]:.3e} to {freq_array_hz[-1]:.3e} Hz) ---")
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
        # This case implies ports are defined but maybe only connect to ground?
         logger.error(f"Circuit '{circuit.name}' has {num_ports} ports but reduced system dimension is 0.")
         raise SimulationError(f"Circuit '{circuit.name}' has {num_ports} ports but a 0-dimensional reduced system. Check connections.")
    elif num_ports == 0:
         # Circuit has internal nodes but no ports. Simulation is possible but yields no Y-matrix.
         logger.warning(f"Circuit '{circuit.name}' has no external ports. Simulation will run but result matrix will be 0x0.")
         # We can proceed, but the result will be empty. Allocate correctly.
         y_matrices = np.empty((num_freqs, 0, 0), dtype=np.complex128)

    else: # Normal case: num_ports > 0 and reduced_dim > 0
        y_matrices = np.empty((num_freqs, num_ports, num_ports), dtype=np.complex128)


    # --- Frequency Loop ---
    for idx, freq in enumerate(freq_array_hz):
        logger.info(f"  Simulating point {idx+1}/{num_freqs} at {freq:.4e} Hz...")
        Y_intrinsic = np.full((num_ports, num_ports), np.nan + 0j, dtype=np.complex128) # Default to NaN

        try:
            # 2. Assemble MNA matrix (only components, uses cached sparsity)
            Yn_full = assembler.assemble(freq)

            # 3. Reduce matrix (remove ground row/col)
            # Ground is assumed index 0. node_count is the full size.
            if node_count <= 1:
                 # Only ground node exists, or error in indexing
                 if num_ports > 0: # Ports defined but no circuit?
                      raise SimulationError(f"Circuit reduced to single node (ground) but {num_ports} ports are defined.")
                 else: # No ports, only ground: empty result is correct
                      Y_intrinsic = np.empty((0,0), dtype=np.complex128)
                      if num_ports == 0: y_matrices[idx, :, :] = Y_intrinsic # Store empty 0x0 matrix
                      continue # Skip to next frequency

            Yn_reduced = Yn_full[1:, 1:].tocsc() # Ensure CSC

            # Check dimensions after reduction
            if Yn_reduced.shape[0] != reduced_dim or Yn_reduced.shape[1] != reduced_dim:
                raise SimulationError(f"Internal Error: Reduced matrix shape {Yn_reduced.shape} does not match expected dimension {reduced_dim}.")

            # 4. Partition the Reduced Matrix
            # Use np.ix_ for safe sparse slicing based on pre-calculated reduced indices
            Y_EE = Yn_reduced[np.ix_(ext_indices_reduced, ext_indices_reduced)]
            # Handle cases with no internal nodes or no external ports safely
            if num_internal_nodes > 0 and num_ports > 0:
                 Y_EI = Yn_reduced[np.ix_(ext_indices_reduced, int_indices_reduced)]
                 Y_IE = Yn_reduced[np.ix_(int_indices_reduced, ext_indices_reduced)]
            else:
                 # Create empty sparse matrices with correct shape if one dimension is 0
                 Y_EI = sp.csc_matrix((num_ports, num_internal_nodes), dtype=np.complex128)
                 Y_IE = sp.csc_matrix((num_internal_nodes, num_ports), dtype=np.complex128)

            if num_internal_nodes > 0:
                 Y_II = Yn_reduced[np.ix_(int_indices_reduced, int_indices_reduced)]
            else: # No internal nodes
                 Y_II = sp.csc_matrix((0, 0), dtype=np.complex128)

            # 5. Calculate Schur Complement
            if num_internal_nodes == 0:
                 # No internal nodes, Y_intrinsic is just Y_EE
                 if num_ports > 0:
                    Y_intrinsic = Y_EE.toarray() # Convert final result to dense
                 else: # No ports, no internal nodes
                    Y_intrinsic = np.empty((0,0), dtype=np.complex128)

            else: # Internal nodes exist
                 # Factorize Y_II
                 lu_II = factorize_mna_matrix(Y_II) # Handles singularity check inside

                 # Solve Y_II * X = Y_IE
                 # lu_II.solve can handle sparse or dense RHS. Y_IE is sparse.
                 # Using dense RHS is often faster for typical sizes involved here.
                 try:
                     # Convert Y_IE to dense for solve. Check if Y_IE is actually non-zero first.
                     if Y_IE.nnz > 0:
                         X = lu_II.solve(Y_IE.toarray()) # Solve returns dense result
                     else: # If Y_IE is all zeros, X is all zeros
                         X = np.zeros((num_internal_nodes, num_ports), dtype=np.complex128)

                     # Check solution for NaNs/Infs which might indicate near-singularity not caught by factorize
                     if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                         raise SingularMatrixError("Solution X for internal nodes contains NaN/Inf.")

                 except (SingularMatrixError, RuntimeError, ValueError) as solve_err:
                      logger.error(f"Failed to solve internal node system Y_II*X = Y_IE at {freq:.4e} Hz: {solve_err}")
                      # Set Y_intrinsic to NaN and continue to next frequency
                      Y_intrinsic = np.full((num_ports, num_ports), np.nan + 0j, dtype=np.complex128)
                      if num_ports > 0: y_matrices[idx, :, :] = Y_intrinsic
                      continue


                 # Calculate Y_intrinsic = Y_EE - Y_EI @ X
                 # Y_EI is sparse, X is dense. Result is dense.
                 if num_ports > 0:
                     Y_intrinsic = Y_EE.toarray() - (Y_EI @ X)
                 else: # Should not happen if num_internal_nodes > 0 implies num_ports > 0? No.
                     Y_intrinsic = np.empty((0,0), dtype=np.complex128) # No ports -> empty matrix


            # 6. Store the resulting intrinsic Y-matrix (ensure it's dense complex)
            if num_ports > 0:
                 if Y_intrinsic.shape != (num_ports, num_ports):
                      raise SimulationError(f"Internal Error: Calculated Y_intrinsic shape {Y_intrinsic.shape} mismatch with port count {num_ports}.")
                 y_matrices[idx, :, :] = np.asarray(Y_intrinsic, dtype=np.complex128)
                 logger.debug(f"  Intrinsic Y-matrix calculated for {freq:.4e} Hz.")
            # else: 0x0 matrix already handled if num_ports == 0 initially

        except (MnaInputError, SingularMatrixError, ComponentError, SimulationError) as e:
             logger.error(f"Simulation failed at {freq:.4e} Hz: {e}", exc_info=False)
             # Store NaN matrix
             if num_ports > 0: y_matrices[idx, :, :] = np.full((num_ports, num_ports), np.nan + 0j)
             continue # Continue sweep
        except Exception as e:
             logger.error(f"Unexpected error during simulation at {freq:.4e} Hz: {e}", exc_info=True)
             if num_ports > 0: y_matrices[idx, :, :] = np.full((num_ports, num_ports), np.nan + 0j)
             continue

    logger.info(f"--- Frequency sweep finished for '{circuit.name}' ---")
    if np.any(np.isnan(y_matrices)):
         logger.warning(f"Sweep for '{circuit.name}' completed, but one or more frequency points failed (results contain NaN).")

    return freq_array_hz, y_matrices