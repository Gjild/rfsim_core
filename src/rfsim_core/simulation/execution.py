# --- Modify: src/rfsim_core/simulation/execution.py ---
import logging
import numpy as np
import scipy.sparse as sp
# No direct use of splinalg here, but solver.py uses it.
# import scipy.sparse.linalg as splinalg 
from typing import Tuple, Dict, Any, Optional, Set, List

# Import MNA assembler and solver components
from .mna import MnaAssembler, MnaInputError
from .solver import factorize_mna_matrix, solve_mna_system, SingularMatrixError

# Import core data structures and error types
from ..data_structures import Circuit
from .. import ureg, Quantity, ComponentError # ureg is not directly used here but good practice

# Import validation components
from ..validation import SemanticValidator, SemanticValidationError, ValidationIssue, ValidationIssueLevel

# --- ADD New Imports for Phase 7 ---
from ..analysis_tools import DCAnalyzer, TopologyAnalyzer, DCAnalysisError, TopologyAnalysisError
from ..components.base import DCBehaviorType # Potentially for logging/debug
# --- Import constant from correct location ---
from ..constants import LARGE_ADMITTANCE_SIEMENS

logger = logging.getLogger(__name__)

class SimulationError(Exception):
    """General error during simulation execution."""
    pass

# --- run_simulation (remains unchanged for now, as per Phase 6 compatibility) ---
def run_simulation(circuit: Circuit, freq_hz: float) -> np.ndarray:
    """
    Runs a single-frequency AC simulation.
    Note: This is a simplified version. Full run_sweep is recommended.
    This version does not perform the rigorous DC analysis or topological filtering from Phase 7.
    """
    if not isinstance(circuit, Circuit) or not hasattr(circuit, 'sim_components'):
        raise MnaInputError("Input circuit object does not appear to be simulation-ready. Ensure CircuitBuilder.build_circuit() was called.")
    if not isinstance(freq_hz, (float, int)) or freq_hz <= 0: # AC only, F>0
        raise MnaInputError(f"Single frequency simulation requires freq_hz > 0. Got {freq_hz}")

    logger.info(f"--- Starting single-frequency simulation for '{circuit.name}' at {freq_hz:.4e} Hz ---")

    # Semantic Validation
    logger.info(f"Performing semantic validation for circuit '{circuit.name}'...")
    validator = SemanticValidator(circuit)
    all_issues = validator.validate()
    # Basic handling: log warnings/info, raise on error
    warnings_and_info = [issue for issue in all_issues if issue.level in (ValidationIssueLevel.WARNING, ValidationIssueLevel.INFO)]
    errors = [issue for issue in all_issues if issue.level == ValidationIssueLevel.ERROR]
    for issue_item in warnings_and_info:
        log_level = logging.WARNING if issue_item.level == ValidationIssueLevel.WARNING else logging.INFO
        logger.log(log_level, f"Semantic Validation [{issue_item.level.name} - {issue_item.code}]: {issue_item.message} "
                              f"(Component: {issue_item.component_id or 'N/A'}, Net: {issue_item.net_name or 'N/A'}, "
                              f"Param: {issue_item.parameter_name or 'N/A'}, Details: {issue_item.details or ''})")
    if errors:
        error_messages_list = [f"  - [{err.code}]: {err.message} (Component: {err.component_id or 'N/A'}, "
                               f"Net: {err.net_name or 'N/A'}, Param: {err.parameter_name or 'N/A'}, Details: {err.details or ''})" for err in errors]
        summary_message = (f"Semantic validation failed for circuit '{circuit.name}' "
                           f"with {len(errors)} error(s):\n" + "\n".join(error_messages_list))
        logger.error(summary_message)
        raise SemanticValidationError(errors, summary_message)
    logger.info(f"Semantic validation passed for circuit '{circuit.name}'.")


    try:
        assembler = MnaAssembler(circuit) # Uses full circuit, no active_nets_override
        num_ports = len(assembler.port_names)
        
        if assembler.node_count <= 1:
            if num_ports > 0:
                raise SimulationError("Circuit reduced to a single node (or no nodes) but external ports are defined.")
            logger.info("Circuit has one or zero nodes. Returning empty Y-matrix.")
            return np.empty((0, 0), dtype=np.complex128)

        Yn_full = assembler.assemble(freq_hz)

        # Schur Complement Logic (from Phase 6)
        if num_ports == 0:
            logger.info("No external ports defined. Returning empty Y-matrix.")
            return np.empty((0, 0), dtype=np.complex128)

        Yn_reduced = Yn_full[1:, 1:].tocsc() # Exclude ground node (assumed index 0)
        ext_indices_reduced = assembler.external_node_indices_reduced
        int_indices_reduced = assembler.internal_node_indices_reduced
        num_internal_nodes = len(int_indices_reduced)

        Y_EE = Yn_reduced[np.ix_(ext_indices_reduced, ext_indices_reduced)]
        Y_EI = Yn_reduced[np.ix_(ext_indices_reduced, int_indices_reduced)] if num_internal_nodes > 0 else sp.csc_matrix((num_ports, 0), dtype=complex)
        Y_IE = Yn_reduced[np.ix_(int_indices_reduced, ext_indices_reduced)] if num_internal_nodes > 0 else sp.csc_matrix((0, num_ports), dtype=complex)
        Y_II = Yn_reduced[np.ix_(int_indices_reduced, int_indices_reduced)] if num_internal_nodes > 0 else sp.csc_matrix((0, 0), dtype=complex)

        if num_internal_nodes == 0:
            Y_intrinsic_current = Y_EE.toarray()
        else:
            lu_II = factorize_mna_matrix(Y_II)
            if Y_IE.nnz > 0: # Only solve if Y_IE is not all zeros
                X = lu_II.solve(Y_IE.toarray()) # Y_IE is (num_internal, num_ports)
                if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                    raise SingularMatrixError("Solution X for internal nodes contains NaN/Inf values.")
            else: # Y_IE is all zeros
                X = np.zeros((num_internal_nodes, num_ports), dtype=complex)
            
            Y_intrinsic_current = Y_EE.toarray() - (Y_EI @ X)
        
        logger.info(f"--- Single-frequency simulation finished for '{circuit.name}' at {freq_hz:.4e} Hz ---")
        return Y_intrinsic_current

    except (MnaInputError, SingularMatrixError, ComponentError) as e:
        logger.error(f"Simulation failed for '{circuit.name}' at {freq_hz:.4e} Hz: {e}", exc_info=True)
        raise SimulationError(f"Simulation run failed: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error during simulation for '{circuit.name}' at {freq_hz:.4e} Hz: {e}", exc_info=True)
        raise SimulationError(f"Unexpected simulation error: {e}") from e


# --- run_sweep incorporating Phase 7 logic ---
def run_sweep(circuit: Circuit, freq_array_hz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, Any]]]:
    """
    Runs a frequency sweep AC simulation (all frequencies must be >= 0) including
    rigorous DC (F=0) analysis and topological pre-filtering.

    Args:
        circuit: The Circuit object, processed by CircuitBuilder and ready for simulation.
        freq_array_hz: A 1D NumPy array of simulation frequencies in Hz (all must be >= 0).
                       An F=0 point will trigger use of the DC analysis results.

    Returns:
        A tuple containing:
            - frequencies_hz: The input frequency array used (np.ndarray).
            - y_matrices: 3D NumPy array (NumFreqs x NumPorts x NumPorts)
                          containing the intrinsic Y-matrix for each frequency.
                          The F=0 point will contain results mapped from DC analysis if successful,
                          otherwise NaNs. Ports shorted to ground at DC will have diagonal
                          entries set to LARGE_ADMITTANCE_SIEMENS. Ports vanishing at DC
                          will have zero entries.
            - dc_analysis_results: Dictionary returned by DCAnalyzer.analyze() if successful,
                                   otherwise None.

    Raises:
        SimulationError, MnaInputError, SingularMatrixError, ComponentError,
        SemanticValidationError, DCAnalysisError, TopologyAnalysisError.
    """
    if not isinstance(freq_array_hz, np.ndarray) or freq_array_hz.ndim != 1:
            raise MnaInputError("Frequency sweep must be provided as a 1D NumPy array.")
    if len(freq_array_hz) == 0:
            logger.warning("Frequency sweep array is empty. Returning empty results.")
            num_ports_from_circuit = len(circuit.external_ports)
            return freq_array_hz, np.array([], dtype=np.complex128).reshape(0, num_ports_from_circuit, num_ports_from_circuit), None

    if np.any(freq_array_hz < 0):
            raise MnaInputError(f"All frequencies in the sweep must be >= 0 Hz. Found min value: {np.min(freq_array_hz)} Hz.")

    if not isinstance(circuit, Circuit) or not hasattr(circuit, 'sim_components'):
            raise MnaInputError("Input circuit object does not appear to be simulation-ready (missing 'sim_components'). Ensure CircuitBuilder.build_circuit() was called.")

    logger.info(f"--- Starting frequency sweep for '{circuit.name}' ({len(freq_array_hz)} points) ---")

    # --- 0. Semantic Validation (Existing - Assumed OK from Phase 6) ---
    logger.info(f"Performing semantic validation for circuit '{circuit.name}'...")
    validator = SemanticValidator(circuit)
    all_issues = validator.validate()
    warnings_and_info = [issue for issue in all_issues if issue.level in (ValidationIssueLevel.WARNING, ValidationIssueLevel.INFO)]
    errors = [issue for issue in all_issues if issue.level == ValidationIssueLevel.ERROR]
    for issue_item in warnings_and_info:
        log_level = logging.WARNING if issue_item.level == ValidationIssueLevel.WARNING else logging.INFO
        logger.log(log_level, f"Semantic Validation [{issue_item.level.name} - {issue_item.code}]: {issue_item.message} "
                              f"(Component: {issue_item.component_id or 'N/A'}, Net: {issue_item.net_name or 'N/A'}, "
                              f"Param: {issue_item.parameter_name or 'N/A'}, Details: {issue_item.details or ''})")
    if errors:
        error_messages_list = [f"  - [{err.code}]: {err.message} (Component: {err.component_id or 'N/A'}, "
                               f"Net: {err.net_name or 'N/A'}, Param: {err.parameter_name or 'N/A'}, Details: {err.details or ''})" for err in errors]
        summary_message = (f"Semantic validation failed for circuit '{circuit.name}' "
                           f"with {len(errors)} error(s):\n" + "\n".join(error_messages_list))
        logger.error(summary_message)
        raise SemanticValidationError(errors, summary_message)
    logger.info(f"Semantic validation passed for circuit '{circuit.name}'.")

    # --- 1. Rigorous DC Analysis (F=0) ---
    dc_analysis_results: Optional[Dict[str, Any]] = None
    dc_analyzer_instance: Optional[DCAnalyzer] = None
    logger.info(f"Performing rigorous DC analysis for circuit '{circuit.name}'...")
    try:
        dc_analyzer_instance = DCAnalyzer(circuit)
        dc_analysis_results = dc_analyzer_instance.analyze()
        logger.info("Rigorous DC analysis complete.")
        if dc_analysis_results:
            y_ports_dc_shape = getattr(dc_analysis_results.get('Y_ports_dc'), 'magnitude', np.array([])).shape
            logger.debug(f"DC Results: Y_ports_dc shape: {y_ports_dc_shape}, "
                         f"DC Port Names (Original Ext. Port Name reps): {dc_analysis_results.get('dc_port_names_ordered', 'N/A')}, "
                         f"DC Port Mapping (OrigExtPortName -> Y_ports_dc index | None): {dc_analysis_results.get('dc_port_mapping', 'N/A')}")
    except DCAnalysisError as e:
        logger.error(f"Rigorous DC analysis failed for '{circuit.name}': {e}", exc_info=True)
    except ComponentError as e: 
        logger.error(f"Component error during rigorous DC analysis for '{circuit.name}': {e}", exc_info=True)
    except Exception as e: 
        logger.error(f"Unexpected error during rigorous DC analysis for '{circuit.name}': {e}", exc_info=True)

    # --- 2. Topological Floating Node Removal (Pre-Sweep for F>0 AC) ---
    active_nets_for_ac: Optional[Set[str]] = None
    topology_analyzer: Optional[TopologyAnalyzer] = None
    logger.info(f"Performing AC topological floating node analysis for circuit '{circuit.name}'...")
    try:
        topology_analyzer = TopologyAnalyzer(circuit)
        active_nets_for_ac = topology_analyzer.get_active_nets()
    except TopologyAnalysisError as e:
        logger.error(f"AC Topological floating node analysis internal failure: {e}", exc_info=True)
        raise SimulationError(f"AC Topological floating node analysis internal failure: {e}") from e
    except ComponentError as e: 
        logger.error(f"Component error during AC Topological floating node analysis setup: {e}", exc_info=True)
        raise SimulationError(f"Component error during AC Topological floating node analysis setup: {e}") from e
    except Exception as e: # Catch unexpected errors from TopologyAnalyzer instantiation or get_active_nets call itself
        logger.error(f"Unexpected error during TopologyAnalyzer operation for circuit '{circuit.name}': {e}", exc_info=True)
        raise SimulationError(f"Unexpected error during TopologyAnalyzer operation: {e}") from e

    # Perform validation checks based on the results from TopologyAnalyzer
    # These checks can raise SimulationError directly.
    if not active_nets_for_ac:
        if circuit.external_ports:
            raise SimulationError(f"No nets are active for AC simulation, but circuit has external ports. This indicates all ports are floating.")
        else: 
            logger.warning(f"No nets are active for AC simulation (and no external ports defined). AC results will be empty/NaN.")
    else: # Some nets are active
        # Check 1: All defined external port nets must be present in active_nets_for_ac.
        # (This also implicitly checks that port nets exist in the circuit's net list if TopologyAnalyzer builds graph from circuit.nets)
        for port_name in circuit.external_ports.keys():
            if port_name not in active_nets_for_ac:
                raise SimulationError(f"External port '{port_name}' is topologically floating for AC analysis (not in active_nets_for_ac). Cannot proceed.")
        
        # Check 2: If external ports exist (and are confirmed active from above), ground net must also be active.
        if circuit.external_ports: # This implies circuit.ground_net_name must be valid.
            if circuit.ground_net_name not in active_nets_for_ac:
                 raise SimulationError(f"Circuit ground net '{circuit.ground_net_name}' is not in the set of active nets for AC simulation, "
                                       f"but external ports exist. This is invalid as ground is not part of the analyzable AC circuit.")
            
            # Check 3: If external ports and ground are active, check connectivity between them.
            # Ensure topology_analyzer was successfully instantiated before calling its methods.
            if topology_analyzer is None: # Should not happen if previous try-except for TA was structured well
                raise SimulationError("Internal error: TopologyAnalyzer instance is not available for connectivity checks.")

            if not topology_analyzer.are_ports_connected_to_active_ground():
                if len(circuit.external_ports) == 1:
                    the_only_port_name = list(circuit.external_ports.keys())[0]
                    raise SimulationError(
                        f"External port '{the_only_port_name}' is topologically floating (not connected to the active ground net '{circuit.ground_net_name}' via a conductive path)."
                    )
                else: # Multi-port case
                    raise SimulationError(
                        f"Circuit ground net '{circuit.ground_net_name}' is floating for AC analysis relative to external ports. "
                        f"Ensure a conductive path (not via a structurally open component like R=inf, C=0, L=inf) "
                        f"exists between at least one active external port and the ground net."
                    )
    
    logger.info(f"AC topological analysis and validation complete. Active nets for AC: {len(active_nets_for_ac) if active_nets_for_ac else 0}")

    # --- 3. MNA Assembler Initialization for AC Sweep ---
    # (Rest of the function remains the same as provided in the problem description)
    logger.info("Initializing MNA Assembler for AC sweep (with filtered netlist view if applicable)...")
    try:
        assembler = MnaAssembler(circuit, active_nets_override=active_nets_for_ac)
        ac_port_names_ordered = assembler.port_names 
        num_ac_ports = len(ac_port_names_ordered)
    except MnaInputError as e:
        logger.error(f"MNA Assembler initialization failed: {e}", exc_info=True)
        raise SimulationError(f"MNA Assembler setup failed: {e}") from e
    except Exception as e: 
        logger.error(f"Unexpected error during MNA Assembler initialization: {e}", exc_info=True)
        raise SimulationError(f"Unexpected error during MNA Assembler setup: {e}") from e

    # --- 4. AC Frequency Sweep Loop ---
    y_matrices_ac_sweep = np.empty((len(freq_array_hz), num_ac_ports, num_ac_ports), dtype=np.complex128)
    y_matrices_ac_sweep.fill(np.nan) 

    if num_ac_ports == 0 and circuit.external_ports:
        logger.warning(f"Circuit '{circuit.name}' has external ports defined, but none are active for AC analysis after topological filtering. AC results will be NaN for an empty (0x0) Y-matrix structure.")
    elif num_ac_ports == 0 and not circuit.external_ports:
        logger.info(f"Circuit '{circuit.name}' has no external ports defined (or none active). AC Y-matrix will be empty (0x0).")


    for idx, freq_val_hz in enumerate(freq_array_hz):
        freq_str = f"{freq_val_hz:.4e} Hz"
        is_dc_point = np.isclose(freq_val_hz, 0.0)

        if is_dc_point:
            logger.info(f"  Processing point {idx+1}/{len(freq_array_hz)} at {freq_str} (DC): Using pre-calculated DC results.")
            if dc_analysis_results and dc_analyzer_instance and \
               dc_analysis_results.get('Y_ports_dc') is not None and \
               dc_analysis_results.get('dc_port_mapping') is not None: 

                if num_ac_ports > 0:
                    y_dc_qty = dc_analysis_results['Y_ports_dc']
                    dc_port_mapping_orig_to_idx = dc_analysis_results['dc_port_mapping']
                    
                    if isinstance(y_dc_qty, Quantity) and y_dc_qty.check(ureg.siemens):
                        y_dc_mag_full = y_dc_qty.to_base_units().magnitude 
                        temp_dc_y_for_ac_structure = np.full((num_ac_ports, num_ac_ports), 0j, dtype=np.complex128)

                        ac_port_idx_to_dc_y_matrix_idx_map: Dict[int, int] = {}
                        for ac_idx, ac_port_name in enumerate(ac_port_names_ordered):
                            dc_y_matrix_idx_for_this_ac_port = dc_port_mapping_orig_to_idx.get(ac_port_name)
                            if dc_y_matrix_idx_for_this_ac_port is not None:
                                ac_port_idx_to_dc_y_matrix_idx_map[ac_idx] = dc_y_matrix_idx_for_this_ac_port

                        for ac_idx_row, dc_y_idx_row in ac_port_idx_to_dc_y_matrix_idx_map.items():
                            for ac_idx_col, dc_y_idx_col in ac_port_idx_to_dc_y_matrix_idx_map.items():
                                if 0 <= dc_y_idx_row < y_dc_mag_full.shape[0] and \
                                   0 <= dc_y_idx_col < y_dc_mag_full.shape[1]:
                                    temp_dc_y_for_ac_structure[ac_idx_row, ac_idx_col] = y_dc_mag_full[dc_y_idx_row, dc_y_idx_col]
                                else:
                                    logger.error(f"DC port index {dc_y_idx_row} or {dc_y_idx_col} out of bounds for Y_ports_dc shape {y_dc_mag_full.shape} when mapping for AC port '{ac_port_names_ordered[ac_idx_row if ac_idx_row < len(ac_port_names_ordered) else -1]}' or '{ac_port_names_ordered[ac_idx_col if ac_idx_col < len(ac_port_names_ordered) else -1]}'. Setting to NaN.")
                                    temp_dc_y_for_ac_structure[ac_idx_row, ac_idx_col] = np.nan + 1j*np.nan
                        
                        for ac_idx, ac_port_name in enumerate(ac_port_names_ordered):
                            if dc_port_mapping_orig_to_idx.get(ac_port_name) is None: 
                                try:
                                    original_port_net_supernode_rep_name = dc_analyzer_instance.get_supernode_representative_name(ac_port_name)
                                    if original_port_net_supernode_rep_name is not None and \
                                       dc_analyzer_instance.is_ground_supernode_by_representative(original_port_net_supernode_rep_name):
                                        logger.debug(f"AC port '{ac_port_name}' (AC idx {ac_idx}) was shorted to ground at DC. Setting Y[{ac_idx},{ac_idx}] to LARGE_ADMITTANCE.")
                                        temp_dc_y_for_ac_structure[ac_idx, ac_idx] = LARGE_ADMITTANCE_SIEMENS
                                    else:
                                        logger.debug(f"AC port '{ac_port_name}' (AC idx {ac_idx}) vanished into an internal DC node or was subsumed by another DC port representative. Y-params entries remain 0j.")
                                except Exception as e_helper:
                                    logger.warning(f"Could not determine DC fate of AC port '{ac_port_name}' which vanished at DC, due to: {e_helper}. Its Y-params entries remain 0j.")
                        
                        y_matrices_ac_sweep[idx] = temp_dc_y_for_ac_structure
                    else:
                        logger.warning(f"DC analysis result 'Y_ports_dc' is not a valid Admittance Quantity or is None. Using NaN for F=0.")
                else: 
                     logger.debug("DC point processing skipped as there are no AC ports (num_ac_ports=0). Result remains NaN (shape 0x0).")
            else: 
                logger.warning(f"DC analysis did not produce required results or failed. Using NaN for F=0.")
            continue 

        # --- Standard AC MNA Path for F > 0 ---
        logger.debug(f"  Processing point {idx+1}/{len(freq_array_hz)} at {freq_str} (AC): Using standard MNA assembly.")

        if assembler.node_count <=1 : 
            logger.warning(f"  Skipping MNA assembly for F={freq_str}, circuit reduced to <=1 node by active_nets filter. Results will be NaN.")
            continue

        try:
            Yn_full = assembler.assemble(freq_val_hz) 
            Y_intrinsic_current = np.full((num_ac_ports, num_ac_ports), np.nan + 0j, dtype=np.complex128)

            if assembler.node_count <= 1: 
                 if num_ac_ports > 0: raise SimulationError("AC: Circuit reduced to single node but ports defined.")
                 Y_intrinsic_current = np.empty((0,0), dtype=np.complex128)
            elif num_ac_ports == 0:
                 Y_intrinsic_current = np.empty((0,0), dtype=np.complex128)
            else:
                 Yn_reduced = Yn_full[1:, 1:].tocsc() if (assembler.node_map.get(assembler._effective_ground_net_name) == 0) else Yn_full.tocsc()
                 
                 if assembler.node_map.get(assembler._effective_ground_net_name) != 0:
                     logger.debug(f"AC Solve: Effective ground '{assembler._effective_ground_net_name}' is not index 0 or not active for MNA reduction. Reducing full system.")
                 
                 ext_indices_reduced = assembler.external_node_indices_reduced
                 int_indices_reduced = assembler.internal_node_indices_reduced
                 num_internal_nodes_reduced = len(int_indices_reduced)

                 Y_EE = Yn_reduced[np.ix_(ext_indices_reduced, ext_indices_reduced)]
                 Y_EI = Yn_reduced[np.ix_(ext_indices_reduced, int_indices_reduced)] if num_internal_nodes_reduced > 0 else sp.csc_matrix((num_ac_ports, 0), dtype=complex)
                 Y_IE = Yn_reduced[np.ix_(int_indices_reduced, ext_indices_reduced)] if num_internal_nodes_reduced > 0 else sp.csc_matrix((0, num_ac_ports), dtype=complex)
                 Y_II = Yn_reduced[np.ix_(int_indices_reduced, int_indices_reduced)] if num_internal_nodes_reduced > 0 else sp.csc_matrix((0, 0), dtype=complex)

                 if num_internal_nodes_reduced == 0:
                     Y_intrinsic_current = Y_EE.toarray()
                 else:
                     lu_II = factorize_mna_matrix(Y_II)
                     if Y_IE.nnz > 0:
                         X = lu_II.solve(Y_IE.toarray())
                         if np.any(np.isnan(X)) or np.any(np.isinf(X)): raise SingularMatrixError("AC Solve: Solution X for internal nodes contains NaN/Inf.")
                     else:
                         X = np.zeros((num_internal_nodes_reduced, num_ac_ports), dtype=complex)
                     Y_intrinsic_current = Y_EE.toarray() - (Y_EI @ X)
            
            if Y_intrinsic_current.shape != (num_ac_ports, num_ac_ports):
                 raise SimulationError(f"Internal Error: Calculated AC Y_intrinsic shape {Y_intrinsic_current.shape} mismatch with port count {num_ac_ports}.")
            y_matrices_ac_sweep[idx] = np.asarray(Y_intrinsic_current, dtype=np.complex128)
            logger.debug(f"  AC analysis successful for {freq_str}.")

        except (MnaInputError, SingularMatrixError, ComponentError, SimulationError) as e:
            logger.error(f"AC simulation failed at {freq_str}: {e}", exc_info=False) 
            continue
        except Exception as e:
            logger.error(f"Unexpected error during AC simulation at {freq_str}: {e}", exc_info=True)
            continue
        # --- End AC MNA Path ---

    logger.info(f"--- Frequency sweep finished for '{circuit.name}' ---")
    if np.any(np.isnan(y_matrices_ac_sweep)):
            logger.warning(f"Sweep for '{circuit.name}' completed, but one or more frequency points failed (results contain NaN).")

    return freq_array_hz, y_matrices_ac_sweep, dc_analysis_results