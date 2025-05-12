# --- src/rfsim_core/simulation/execution.py ---
import logging
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from typing import Tuple

from .mna import MnaAssembler, MnaInputError
from .solver import factorize_mna_matrix, solve_mna_system, SingularMatrixError
from ..data_structures import Circuit
from .. import ureg, Quantity, ComponentError
from ..validation import SemanticValidator, SemanticValidationError, ValidationIssue, ValidationIssueLevel

logger = logging.getLogger(__name__)

class SimulationError(Exception):
    """General error during simulation execution."""
    pass

def run_simulation(circuit: Circuit, freq_hz: float) -> np.ndarray:
    """
    Runs a single-frequency AC simulation (F >= 0) and returns the
    intrinsic N-port Y-matrix.
    Note: For F=0, this performs component value evaluation at DC, not a full
    topological DC analysis (which is Phase 7).

    Args:
        circuit: The Circuit object, processed by CircuitBuilder and ready for simulation.
        freq_hz: The single simulation frequency in Hz (must be >= 0).

    Returns:
        The N-port intrinsic Y-matrix (NumPorts x NumPorts) at the specified frequency.

    Raises:
        SimulationError, MnaInputError, SingularMatrixError, SemanticValidationError.
    """
    if freq_hz < 0:
        raise MnaInputError(f"Single frequency simulation requires freq_hz >= 0. Got {freq_hz}.")
    if freq_hz == 0:
        logger.warning("Running run_simulation at F=0. Result uses component F=0 admittance, NOT rigorous DC analysis (Phase 7).")

    freq_array = np.array([freq_hz])
    # Pass the already built circuit
    frequencies, results = run_sweep(circuit, freq_array)
    # result shape is (1, num_ports, num_ports)
    if results.shape[0] > 0:
        return results[0, :, :]
    else: 
        num_ports = len(circuit.external_ports) 
        return np.empty((num_ports, num_ports), dtype=np.complex128)

def run_sweep(circuit: Circuit, freq_array_hz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs a frequency sweep AC simulation (all frequencies must be >= 0) and
    calculates the intrinsic N-port Y-matrices using Schur complement reduction.
    Note: For F=0 points in the sweep, this performs component value evaluation
    at DC, not a full topological DC analysis (which is Phase 7).

    Args:
        circuit: The Circuit object, processed by CircuitBuilder and ready for simulation.
        freq_array_hz: A 1D NumPy array of simulation frequencies in Hz (all must be >= 0).

    Returns:
        A tuple containing:
            - frequencies_hz: The input frequency array used (np.ndarray).
            - y_matrices: 3D NumPy array (NumFreqs x NumPorts x NumPorts)
                          containing the intrinsic Y-matrix for each frequency.

    Raises:
        SimulationError, MnaInputError, SingularMatrixError, ComponentError, SemanticValidationError.
    """
    if not isinstance(freq_array_hz, np.ndarray) or freq_array_hz.ndim != 1:
            raise MnaInputError("Frequency sweep must be provided as a 1D NumPy array.")
    if len(freq_array_hz) == 0:
            logger.warning("Frequency sweep array is empty. Returning empty results.")
            num_ports_from_circuit = len(circuit.external_ports)
            return freq_array_hz, np.array([], dtype=np.complex128).reshape(0, num_ports_from_circuit, num_ports_from_circuit)

    if np.any(freq_array_hz < 0):
            raise MnaInputError(f"All frequencies in the sweep must be >= 0 Hz. Found min value: {np.min(freq_array_hz)} Hz.")
    if np.any(freq_array_hz == 0):
        logger.warning("Frequency sweep includes F=0. Results at F=0 use component F=0 admittance, NOT rigorous DC analysis (Phase 7).")

    if not isinstance(circuit, Circuit) or not hasattr(circuit, 'sim_components'):
            raise MnaInputError("Input circuit object does not appear to be simulation-ready (missing 'sim_components'). Ensure CircuitBuilder.build_circuit() was called.")

    # --- Semantic Validation ---
    logger.info(f"Performing semantic validation for circuit '{circuit.name}' prior to sweep.")
    validator = SemanticValidator(circuit)
    all_issues = validator.validate()
    
    warnings_and_info = [issue for issue in all_issues if issue.level in (ValidationIssueLevel.WARNING, ValidationIssueLevel.INFO)]
    errors = [issue for issue in all_issues if issue.level == ValidationIssueLevel.ERROR]
    
    for issue_item in warnings_and_info:
        log_level = logging.WARNING if issue_item.level == ValidationIssueLevel.WARNING else logging.INFO
        # Using issue_item.__str__() might be too verbose for a standard log line.
        # Let's use the planned structured logging.
        logger.log(log_level, 
                    f"Semantic Validation [{issue_item.level.name} - {issue_item.code}]: {issue_item.message} "
                    f"(Component: {issue_item.component_id or 'N/A'}, Net: {issue_item.net_name or 'N/A'}, "
                    f"Param: {issue_item.parameter_name or 'N/A'}, Details: {issue_item.details or ''})")
    
    if errors:
        error_messages_list = []
        for err in errors:
            error_messages_list.append(
                f"  - [{err.code}]: {err.message} (Component: {err.component_id or 'N/A'}, "
                f"Net: {err.net_name or 'N/A'}, Param: {err.parameter_name or 'N/A'}, Details: {err.details or ''})"
            )
        summary_message = (f"Semantic validation failed for circuit '{circuit.name}' "
                            f"with {len(errors)} error(s):\n" + "\n".join(error_messages_list))
        logger.error(summary_message)
        # Pass the original list of error ValidationIssue objects to the exception
        raise SemanticValidationError(errors, summary_message)
    
    logger.info(f"Semantic validation passed for circuit '{circuit.name}'. Proceeding with sweep.")
    # --- End Semantic Validation ---

    num_freqs = len(freq_array_hz)
    logger.info(f"--- Starting frequency sweep for '{circuit.name}' ({num_freqs} points from {np.min(freq_array_hz):.3e} to {np.max(freq_array_hz):.3e} Hz) ---")
    logger.info("Calculating intrinsic Y-matrix using Schur complement.")

    # Initialize MNA Assembler (once)
    try:
            assembler = MnaAssembler(circuit) 
            num_ports = len(assembler.port_indices)
            node_count = assembler.node_count
            ext_indices_reduced = assembler.external_node_indices_reduced 
            int_indices_reduced = assembler.internal_node_indices_reduced 
            num_internal_nodes = len(int_indices_reduced)
            reduced_dim = node_count - 1 

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
    else: 
        y_matrices = np.empty((num_freqs, num_ports, num_ports), dtype=np.complex128)

    for idx, freq in enumerate(freq_array_hz):
        freq_str = f"{freq:.4e} Hz"
        if freq == 0:
            freq_str = "0 Hz (DC - component value eval only)"
        logger.info(f"  Simulating point {idx+1}/{num_freqs} at {freq_str}...")
        Y_intrinsic_current = np.full((num_ports, num_ports), np.nan + 0j, dtype=np.complex128)

        try:
            Yn_full = assembler.assemble(freq) 

            if node_count <= 1:
                    if num_ports > 0:
                            raise SimulationError(f"Circuit reduced to single node (ground) but {num_ports} ports are defined.")
                    else: 
                            Y_intrinsic_current = np.empty((0,0), dtype=np.complex128)
                            if num_ports == 0: y_matrices[idx] = Y_intrinsic_current 
                            continue 

            Yn_reduced = Yn_full[1:, 1:].tocsc() 

            if Yn_reduced.shape[0] != reduced_dim or Yn_reduced.shape[1] != reduced_dim:
                raise SimulationError(f"Internal Error: Reduced matrix shape {Yn_reduced.shape} does not match expected dimension {reduced_dim}.")

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
                            if num_ports > 0: y_matrices[idx] = Y_intrinsic_current
                            continue

                    if num_ports > 0:
                        Y_intrinsic_current = Y_EE.toarray() - (Y_EI @ X)
                    else:
                        Y_intrinsic_current = np.empty((0,0), dtype=np.complex128)

            if num_ports >= 0: 
                    if Y_intrinsic_current.shape != (num_ports, num_ports):
                            raise SimulationError(f"Internal Error: Calculated Y_intrinsic shape {Y_intrinsic_current.shape} mismatch with port count {num_ports}.")
                    y_matrices[idx] = np.asarray(Y_intrinsic_current, dtype=np.complex128)
                    logger.debug(f"  Intrinsic Y-matrix calculated for {freq_str}.")

        except (MnaInputError, SingularMatrixError, ComponentError, SimulationError) as e:
                logger.error(f"Simulation failed at {freq_str}: {e}", exc_info=False)
                if num_ports > 0: y_matrices[idx] = Y_intrinsic_current
                continue 
        except Exception as e:
                logger.error(f"Unexpected error during simulation at {freq_str}: {e}", exc_info=True)
                if num_ports > 0: y_matrices[idx] = Y_intrinsic_current
                continue
    
    logger.info(f"--- Frequency sweep finished for '{circuit.name}' ---")
    if np.any(np.isnan(y_matrices)):
            logger.warning(f"Sweep for '{circuit.name}' completed, but one or more frequency points failed (results contain NaN).")

    return freq_array_hz, y_matrices