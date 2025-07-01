# src/rfsim_core/simulation/execution.py
import logging
import numpy as np
from typing import Tuple, Dict, Any, Optional, List, Set
import networkx as nx

from .mna import MnaAssembler, StampInfo
from .solver import factorize_mna_matrix
from .exceptions import (
    MnaInputError,
    ComponentError,
    DCAnalysisError,
    SingularMatrixError,
    SingleLevelSimulationFailure,
)
from ..data_structures import Circuit
from ..units import ureg
from ..components.subcircuit import SubcircuitInstance

# --- BEGIN Phase 9 Task 5 Change ---
# NEW IMPORT: Import the IMnaContributor protocol.
# This import is the cornerstone of the decoupling. The execution engine now depends on
# an abstract interface (the Protocol), not a concrete component base class method. This
# fulfills the "Decoupling through Queryable Interfaces" mandate.
from ..components.capabilities import IMnaContributor
# --- END Phase 9 Task 5 Change ---

from ..constants import LARGE_ADMITTANCE_SIEMENS
from ..analysis_tools import DCAnalyzer, TopologyAnalyzer
from ..parameters import ParameterManager, ParameterError
from ..validation import SemanticValidator
from ..validation.exceptions import SemanticValidationError
from ..errors import SimulationRunError, Diagnosable, format_diagnostic_report

logger = logging.getLogger(__name__)


def run_simulation(circuit: Circuit, freq_hz: float) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    freq_array = np.array([freq_hz], dtype=float)
    _, y_matrices, dc_results = run_sweep(circuit, freq_array)
    return y_matrices[0], dc_results


def run_sweep(
    circuit: Circuit, freq_array_hz: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, Any]]]:
    try:
        if not isinstance(circuit, Circuit) or not circuit.parameter_manager:
            raise ValueError("Input circuit object is not simulation-ready.")
        
        logger.info(f"--- Starting hierarchical frequency sweep for '{circuit.name}' ---")

        validator = SemanticValidator(circuit)
        issues = validator.validate()
        error_issues = [issue for issue in issues if issue.level == "ERROR"]
        if error_issues:
            raise SemanticValidationError(issues)
        
        simulation_cache: Dict[Tuple, Dict[str, Any]] = {}
        _populate_subcircuit_caches_recursive(
            circuit, freq_array_hz, circuit.parameter_manager, simulation_cache
        )
        
        logger.info(f"--- All subcircuit caches populated. Simulating top-level circuit '{circuit.name}'... ---")
        top_level_results = _run_single_level_simulation(circuit, freq_array_hz)
        
        logger.info(f"--- Hierarchical frequency sweep finished for '{circuit.name}' ---")
        return (
            top_level_results['frequencies_hz'],
            top_level_results['y_matrices'],
            top_level_results['dc_results'],
        )

    except Diagnosable as e:
        diagnostic_report = e.get_diagnostic_report()
        raise SimulationRunError(diagnostic_report) from e
    
    except Exception as e:
        report = format_diagnostic_report(
            error_type=f"An Unexpected Simulation Error Occurred ({type(e).__name__})",
            details=f"The simulator encountered an unexpected internal error: {str(e)}",
            suggestion="This may be a bug. Review the traceback and consider filing a bug report.",
            context={},
        )
        raise SimulationRunError(report) from e


def _populate_subcircuit_caches_recursive(
    circuit_node: Circuit,
    freq_array_hz: np.ndarray,
    global_pm: ParameterManager,
    simulation_cache: Dict[Tuple, Dict[str, Any]],
):
    for sim_comp in circuit_node.sim_components.values():
        if isinstance(sim_comp, SubcircuitInstance):
            _populate_subcircuit_caches_recursive(
                sim_comp.sub_circuit_object, freq_array_hz, global_pm, simulation_cache
            )
            
            cache_key = _compute_subcircuit_cache_key(sim_comp, freq_array_hz, global_pm)
            
            if cache_key in simulation_cache:
                logger.info(f"Cache HIT for subcircuit instance '{sim_comp.fqn}'. Re-using results.")
                results = simulation_cache[cache_key]
            else:
                logger.info(f"Cache MISS for '{sim_comp.fqn}'. Simulating definition '{sim_comp.sub_circuit_object.name}'.")
                results = _run_single_level_simulation(sim_comp.sub_circuit_object, freq_array_hz)
                simulation_cache[cache_key] = results
            
            sim_comp.cached_y_parameters_ac = results.get('y_matrices')
            sim_comp.cached_dc_analysis_results = results.get('dc_results')


def _run_single_level_simulation(
    circuit: Circuit, freq_array_hz: np.ndarray
) -> Dict[str, Any]:
    logger.info(f"--- Running single-level simulation for '{circuit.hierarchical_id}' ({len(freq_array_hz)} points) ---")
    
    try:
        dc_analyzer = DCAnalyzer(circuit)
        dc_analysis_results = dc_analyzer.analyze()

        topology_analyzer = TopologyAnalyzer(circuit)
        active_nets_for_ac = topology_analyzer.get_active_nets()
        
        if circuit.external_ports:
            if circuit.ground_net_name not in active_nets_for_ac:
                raise MnaInputError(hierarchical_context=circuit.hierarchical_id, details="Ground net is not part of the active AC circuit, but external ports exist.")
            if not topology_analyzer.are_ports_connected_to_active_ground():
                raise MnaInputError(hierarchical_context=circuit.hierarchical_id, details="No conductive path exists between any active external port and the active ground net.")

        assembler = MnaAssembler(circuit, active_nets_override=active_nets_for_ac)
        ac_port_names_ordered = assembler.port_names
        num_ac_ports = len(ac_port_names_ordered)
        y_matrices = np.full((len(freq_array_hz), num_ac_ports, num_ac_ports), np.nan + 0j, dtype=np.complex128)

        # --- BEGIN Phase 9 Task 5 Change: Refactored Vectorized Pre-computation Step ---
        
        # This step remains the same: all parameters are evaluated once for the entire sweep.
        logger.debug(f"[{circuit.hierarchical_id}] Evaluating all parameters for the full frequency sweep...")
        all_evaluated_params = circuit.parameter_manager.evaluate_all(freq_array_hz)
        
        logger.debug(f"[{circuit.hierarchical_id}] Pre-computing all vectorized MNA stamps via capabilities...")
        all_stamps_vectorized: Dict[str, List[StampInfo]] = {}
        
        # This loop is the heart of the "CRITICAL CORRECTION". It iterates through the
        # components ONCE, before the per-frequency loop, to gather all vectorized stamps.
        for comp_fqn, sim_comp in assembler.effective_sim_components.items():
            # STEP 1: Query for the capability.
            # This is the decoupled query. We ask "Can you contribute to MNA?" instead
            # of assuming `sim_comp.get_mna_stamps` exists.
            mna_contributor = sim_comp.get_capability(IMnaContributor)
            
            if mna_contributor:
                # STEP 2: If the capability exists, call its method.
                # The call is wrapped in a try/except block to provide robust, contextual
                # diagnostics if a component's implementation fails.
                try:
                    # The contract here is critical:
                    #   - The `sim_comp` instance is passed as context.
                    #   - The full `freq_array_hz` is passed.
                    #   - The capability is expected to perform a single, vectorized calculation.
                    # This fulfills the "Explicit and Vectorized Context Passing" mandate.
                    stamps = mna_contributor.get_mna_stamps(
                        sim_comp, freq_array_hz, all_evaluated_params
                    )
                    all_stamps_vectorized[comp_fqn] = stamps
                except Exception as e:
                    # Wrap any failure in a diagnosable, contextual error. This provides
                    # clear feedback to the user, pinpointing the failing component.
                    raise ComponentError(component_fqn=comp_fqn, details=f"Failed during vectorized MNA stamp computation: {e}") from e
            else:
                # STEP 3: If the capability does not exist, log it and move on.
                # This makes the system robust to components that are not MNA-aware (e.g.,
                # future non-electrical components). The assembler will correctly skip it as
                # its FQN will not be in the `all_stamps_vectorized` dictionary.
                logger.debug(f"Component '{comp_fqn}' does not provide IMnaContributor capability. Skipping MNA contribution.")
                
        logger.debug(f"[{circuit.hierarchical_id}] All vectorized stamps pre-computed.")

        # --- END Phase 9 Task 5 Change ---

        # --- Main Frequency Loop (Lean Assembly) ---
        # Crucially, this loop and its call to `assembler.assemble` REMAIN UNCHANGED
        # from Phase 8. The assembler's role is now simply to slice the pre-computed
        # vectorized data for the current frequency index. This preserves the high-performance
        # architecture.
        for idx, freq_val_hz in enumerate(freq_array_hz):
            try:
                if np.isclose(freq_val_hz, 0.0):
                    if dc_analysis_results and dc_analysis_results.get('Y_ports_dc') is not None and num_ac_ports > 0:
                        y_matrices[idx] = _map_dc_y_to_ac_ports(dc_analysis_results, ac_port_names_ordered, dc_analyzer)
                    continue

                if assembler.node_count <= 1: continue
                
                Yn_full = assembler.assemble(idx, all_stamps_vectorized)
                
                if num_ac_ports == 0:
                    y_matrices[idx] = np.empty((0, 0), dtype=np.complex128)
                    continue

                Yn_reduced = Yn_full[1:, 1:].tocsc()
                ext_indices_reduced = assembler.external_node_indices_reduced
                int_indices_reduced = assembler.internal_node_indices_reduced
                
                Y_EE = Yn_reduced[np.ix_(ext_indices_reduced, ext_indices_reduced)]
                if not int_indices_reduced:
                    y_matrices[idx] = Y_EE.toarray()
                else:
                    Y_EI = Yn_reduced[np.ix_(ext_indices_reduced, int_indices_reduced)]
                    Y_IE = Yn_reduced[np.ix_(int_indices_reduced, ext_indices_reduced)]
                    Y_II = Yn_reduced[np.ix_(int_indices_reduced, int_indices_reduced)]
                    lu_II = factorize_mna_matrix(Y_II, freq_val_hz)
                    X = lu_II.solve(Y_IE.toarray())
                    y_matrices[idx] = Y_EE.toarray() - (Y_EI @ X)

            except (MnaInputError, SingularMatrixError, ComponentError, ParameterError) as e:
                logger.error(f"[{circuit.hierarchical_id}] AC simulation failed at {freq_val_hz:.4e} Hz: {e}")
                continue

        return {
            'frequencies_hz': freq_array_hz,
            'y_matrices': y_matrices,
            'dc_results': dc_analysis_results,
        }

    except (DCAnalysisError, MnaInputError, ComponentError, SingularMatrixError, ParameterError) as e:
        raise SingleLevelSimulationFailure(
            circuit_fqn=circuit.hierarchical_id,
            circuit_source_path=circuit.source_file_path,
            original_error=e,
        ) from e


def _map_dc_y_to_ac_ports(
    dc_results: Dict[str, Any], ac_port_names: List[str], dc_analyzer: DCAnalyzer
) -> np.ndarray:
    num_ac_ports = len(ac_port_names)
    y_dc_qty = dc_results['Y_ports_dc']
    dc_port_map = dc_results['dc_port_mapping']
    y_dc_mag = y_dc_qty.to(ureg.siemens).magnitude
    
    y_mapped = np.zeros((num_ac_ports, num_ac_ports), dtype=np.complex128)

    ac_to_dc_idx_map = {ac_idx: dc_port_map.get(ac_name) for ac_idx, ac_name in enumerate(ac_port_names)}
    for r_ac, r_dc in ac_to_dc_idx_map.items():
        if r_dc is None: continue
        for c_ac, c_dc in ac_to_dc_idx_map.items():
            if c_dc is None: continue
            y_mapped[r_ac, c_ac] = y_dc_mag[r_dc, c_dc]

    for ac_idx, ac_name in enumerate(ac_port_names):
        if ac_to_dc_idx_map.get(ac_idx) is None:
            srep = dc_analyzer.get_supernode_representative_name(ac_name)
            if srep and dc_analyzer.is_ground_supernode_by_representative(srep):
                y_mapped[ac_idx, ac_idx] = LARGE_ADMITTANCE_SIEMENS

    return y_mapped


def _compute_subcircuit_cache_key(
    sub_inst: SubcircuitInstance, freq_array_hz: np.ndarray, global_pm: ParameterManager
) -> Tuple:
    """Implements the definitive cache key algorithm for subcircuit simulation."""
    # 1. Path to the definition file
    def_path_str = str(sub_inst.sub_circuit_object.source_file_path)
    
    # 2. Canonical tuple of parameter overrides from the instance's YAML
    overrides = sub_inst.raw_ir_data.raw_parameter_overrides
    canonical_overrides_tuple = tuple(sorted(
        (k, str(sorted(v.items()) if isinstance(v, dict) else v))
        for k, v in overrides.items()
    ))
    
    # 3. Get all external dependencies by querying the official ParameterManager API.
    # This is the single source of truth for dependency information.
    fqns_in_sub = {p.fqn for p in sub_inst.sub_circuit_object.parameter_manager.get_all_fqn_definitions()}
    const_ext_deps, freq_ext_deps = global_pm.get_external_dependencies_of_scope(fqns_in_sub)

    # 4. Create a canonical tuple of the *values* of constant external dependencies.
    ext_const_vals_tuple = tuple(sorted(
        (fqn, f"{global_pm.get_constant_value(fqn):~P}") for fqn in const_ext_deps
    ))
    
    # 5. Create a canonical tuple of the *definitions* of frequency-dependent external dependencies.
    ext_freq_defs_tuple = tuple(sorted(
        (p.fqn, p.raw_value_or_expression_str, p.declared_dimension_str)
        for fqn in freq_ext_deps if (p := global_pm.get_parameter_definition(fqn))
    ))
    
    external_context_tuple = (ext_const_vals_tuple, ext_freq_defs_tuple)
    
    # 6. Canonical tuple of the simulation frequencies
    frequency_array_tuple = tuple(np.sort(np.unique(freq_array_hz)))
    
    # The final, correct, and robust cache key.
    return (def_path_str, canonical_overrides_tuple, external_context_tuple, frequency_array_tuple) 