# src/rfsim_core/simulation/engine.py

"""
Defines the `SimulationEngine`, the stateless service that orchestrates the simulation.

This module is the heart of the service-oriented architecture. The SimulationEngine
contains all the imperative logic for running a hierarchical simulation (the "how"),
but it holds no state of its own. It operates on a `SimulationContext` object (the "what")
that is passed to it upon creation.
"""
import logging
import numpy as np
import pint
from typing import List, Optional, Tuple, Dict

from ..data_structures import Circuit
from ..components.subcircuit import SubcircuitInstance
from ..components.capabilities import IMnaContributor
from ..parameters import ParameterError
from ..constants import LARGE_ADMITTANCE_SIEMENS

from .mna import MnaAssembler
from .solver import factorize_mna_matrix
from ..units import ureg, Quantity, ADMITTANCE_DIMENSIONALITY

from .context import SimulationContext
from .results import SubcircuitSimResults
# --- REFINED IMPORTS: Explicitly import all required analysis contracts and services ---
from ..analysis import (
    DCAnalysisResults, DCAnalyzer, TopologyAnalyzer, TopologyAnalysisResults
)
from ..cache import create_subcircuit_sim_key, SimulationCache
from .exceptions import (
    MnaInputError,
    SingularMatrixError,
    SingleLevelSimulationFailure,
)
from ..analysis.exceptions import DCAnalysisError
from ..components.exceptions import ComponentError
from ..errors import FrameworkLogicError


logger = logging.getLogger(__name__)


class SimulationEngine:
    """
    A stateless service that orchestrates the entire simulation process.
    This object contains the imperative logic (the "how") and operates on a
    given SimulationContext (the "what"). It owns no state itself.
    """
    def __init__(self, context: SimulationContext):
        """
        Initializes the engine with the full context for a single simulation run.

        Args:
            context: The immutable SimulationContext object containing all inputs.
        """
        self.context: SimulationContext = context
        self.circuit: Circuit = context.top_level_circuit
        self.freq_array_hz: np.ndarray = context.freq_array_hz
        self.cache: SimulationCache = context.cache
        self.global_pm = self.circuit.parameter_manager
        self.ureg = ureg
        logger.debug(f"SimulationEngine initialized for '{self.circuit.name}'.")

    def execute_sweep(self) -> Tuple[np.ndarray, Optional[DCAnalysisResults]]:
        """
        The main entry point for the engine. Orchestrates the hierarchical sweep.
        """
        # 1. Recursively simulate all subcircuits, populating the cache.
        self._populate_subcircuit_caches_recursive(self.circuit)
        
        # 2. Simulate the top-level circuit, which will now find all subcircuit
        #    results in the cache.
        y_mats, dc_res = self._run_single_level_simulation(self.circuit)
        return y_mats, dc_res

    def _populate_subcircuit_caches_recursive(self, circuit_node: Circuit):
        """
        Recursively simulates subcircuits and populates the 'run' scope of the
        cache with formal `SubcircuitSimResults` objects, which now include
        topology, DC, and AC results.
        """
        for sim_comp in circuit_node.sim_components.values():
            if isinstance(sim_comp, SubcircuitInstance):
                # Recurse depth-first to ensure dependencies are simulated before their parents.
                self._populate_subcircuit_caches_recursive(sim_comp.sub_circuit_object)
                
                # Generate the definitive cache key using the centralized factory.
                cache_key = create_subcircuit_sim_key(sim_comp, self.freq_array_hz, self.global_pm)
                
                # Query the cache service for a pre-existing result.
                cached_results: Optional[SubcircuitSimResults] = self.cache.get(key=cache_key, scope='run')

                if cached_results:
                    logger.info(f"Cache HIT for subcircuit instance '{sim_comp.fqn}'. Re-using results.")
                    results_to_use = cached_results
                else:
                    logger.info(f"Cache MISS for '{sim_comp.fqn}'. Simulating definition '{sim_comp.sub_circuit_object.name}'.")
                    
                    sub_circuit_def = sim_comp.sub_circuit_object
                    
                    # Perform topology analysis for the subcircuit.
                    # This is efficient because TopologyAnalyzer uses the persistent process-level cache.
                    sub_topo_analyzer = TopologyAnalyzer(sub_circuit_def, self.cache)
                    sub_topo_results = sub_topo_analyzer.analyze()

                    # On miss, run the simulation for the subcircuit's definition.
                    y_mats, dc_res = self._run_single_level_simulation(sub_circuit_def)
                    
                    # Package ALL results into the formal dataclass.
                    results_to_use = SubcircuitSimResults(
                        y_parameters=y_mats,
                        dc_results=dc_res,
                        topology_results=sub_topo_results
                    )
                    
                    # Store the formal result object in the cache.
                    self.cache.put(key=cache_key, value=results_to_use, scope='run')

                # Populate all cached attributes on the instance for use by its capabilities.
                sim_comp.cached_y_parameters_ac = results_to_use.y_parameters
                sim_comp.cached_dc_analysis_results = results_to_use.dc_results
                sim_comp.cached_topology_results = results_to_use.topology_results

    def _run_single_level_simulation(self, circuit: Circuit) -> Tuple[np.ndarray, Optional[DCAnalysisResults]]:
        """
        Runs a full simulation for a single circuit level.
        This method uses cache-aware analysis services via dependency injection and enforces
        all component API contracts via the Validation Gateway.
        """
        logger.info(f"--- Running single-level simulation for '{circuit.hierarchical_id}' ({len(self.freq_array_hz)} points) ---")
        
        # --- FRAMEWORK INTEGRITY CHECK ---
        # This is a non-negotiable pre-condition. The engine verifies its own orchestration
        # logic by ensuring all sub-dependency caches have been populated.
        for sim_comp in circuit.sim_components.values():
            if isinstance(sim_comp, SubcircuitInstance):
                # Check all required cached attributes. A failure here is a bug in the engine.
                if sim_comp.cached_dc_analysis_results is None or sim_comp.cached_topology_results is None:
                    raise FrameworkLogicError(
                        f"Pre-condition failed for simulating '{circuit.hierarchical_id}': "
                        f"The cache for subcircuit instance '{sim_comp.fqn}' was not populated. "
                        "This indicates a critical bug in the SimulationEngine's recursive traversal logic."
                    )

        try:
            # 1. Instantiate analysis services, injecting the cache.
            dc_analyzer = DCAnalyzer(circuit, self.cache)
            dc_analysis_results = dc_analyzer.analyze()

            topology_analyzer = TopologyAnalyzer(circuit, self.cache)
            topo_results = topology_analyzer.analyze()

            # 2. Use the formal result objects for subsequent logic.
            active_nets_for_ac = topo_results.active_nets
            if circuit.external_ports:
                if circuit.ground_net_name not in active_nets_for_ac:
                    raise MnaInputError(hierarchical_context=circuit.hierarchical_id, details="Ground net is not part of the active AC circuit, but external ports exist.")
                if not topology_analyzer.are_ports_connected_to_active_ground():
                    raise MnaInputError(hierarchical_context=circuit.hierarchical_id, details="No conductive path exists between any active external port and the active ground net.")

            # 3. MNA Assembly and Pre-computation.
            assembler = MnaAssembler(circuit, active_nets_override=active_nets_for_ac)
            ac_port_names_ordered = assembler.port_names
            num_ac_ports = len(ac_port_names_ordered)
            y_matrices = np.full((len(self.freq_array_hz), num_ac_ports, num_ac_ports), np.nan + 0j, dtype=np.complex128)

            all_evaluated_params = circuit.parameter_manager.evaluate_all(self.freq_array_hz)
            
            # The dictionary now stores a single tuple per component, not a list.
            all_stamps_vectorized: Dict[str, Tuple[Quantity, List[str]]] = {}
            for comp_fqn, sim_comp in assembler.effective_sim_components.items():
                mna_contributor = sim_comp.get_capability(IMnaContributor)
                if mna_contributor:
                    try:
                        # Step 1: Call the untrusted plugin capability to get the stamp tuple.
                        stamp_info = mna_contributor.get_mna_stamps(
                            sim_comp, self.freq_array_hz, all_evaluated_params
                        )
                        
                        # --- NON-NEGOTIABLE VALIDATION GATEWAY ---
                        # The framework MUST enforce its contracts at the boundary.
                        if not (isinstance(stamp_info, tuple) and len(stamp_info) == 2):
                            raise ComponentError(
                                component_fqn=comp_fqn,
                                details=(f"MNA capability expected to return a 2-element tuple (Quantity, List[str]), "
                                         f"but returned type '{type(stamp_info).__name__}'.")
                            )
                        
                        y_qty, port_list = stamp_info
                        
                        
                        if not isinstance(y_qty, Quantity):
                            raise ComponentError(
                                component_fqn=comp_fqn,
                                details=f"Stamp value is not a pint.Quantity object, but type '{type(y_qty).__name__}'."
                            )
                        
                        if y_qty.dimensionality != ADMITTANCE_DIMENSIONALITY:
                            raise ComponentError(
                                component_fqn=comp_fqn,
                                details=(f"Stamp has incorrect physical dimension. Expected [admittance], "
                                         f"but got {y_qty.dimensionality}.")
                            ) from pint.DimensionalityError(y_qty.units, self.ureg.siemens)
                        
                        y_mag = y_qty.magnitude
                        if not isinstance(y_mag, np.ndarray) or y_mag.ndim != 3:
                            raise ComponentError(
                                component_fqn=comp_fqn,
                                details=f"Stamp magnitude must be a 3D NumPy array (freqs, N, N), but got ndim={y_mag.ndim} and shape={y_mag.shape}."
                            )

                        if not (isinstance(y_qty.magnitude, np.ndarray) and y_qty.magnitude.ndim == 3):
                            raise ComponentError(
                               component_fqn=comp_fqn,
                               details=(f"Stamp magnitude must be a 3D NumPy array (freqs, N, N), but got "
                                        f"ndim={y_qty.magnitude.ndim} and shape={y_qty.magnitude.shape}.")
                            )
                        # --- END VALIDATION GATEWAY ---
                        
                        all_stamps_vectorized[comp_fqn] = stamp_info

                    except Exception as e:
                        if isinstance(e, ComponentError): raise
                        raise ComponentError(
                            component_fqn=comp_fqn,
                            details=f"Failed during vectorized MNA stamp computation: {e}"
                        ) from e
                else:
                    logger.debug(f"Component '{comp_fqn}' does not provide IMnaContributor capability. Skipping.")

            # 4. Main Frequency Loop for Assembly and Solving.
            for idx, freq_val_hz in enumerate(self.freq_array_hz):
                try:
                    if np.isclose(freq_val_hz, 0.0):
                        if num_ac_ports > 0:
                            y_matrices[idx] = self._map_dc_y_to_ac_ports(dc_analysis_results, ac_port_names_ordered)
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
                except SingularMatrixError as e:
                    logger.error(f"[{circuit.hierarchical_id}] AC simulation failed at {freq_val_hz:.4e} Hz due to singular matrix: {e}")
                    continue
                except (MnaInputError, ComponentError, ParameterError) as e:
                    logger.error(f"[{circuit.hierarchical_id}] AC simulation failed at {freq_val_hz:.4e} Hz due to a fatal configuration error: {e}")
                    raise e

            return y_matrices, dc_analysis_results
        
        except (DCAnalysisError, MnaInputError, ComponentError, SingularMatrixError, ParameterError) as e:
            raise SingleLevelSimulationFailure(
                circuit_fqn=circuit.hierarchical_id,
                circuit_source_path=circuit.source_file_path,
                original_error=e,
            ) from e

    def _map_dc_y_to_ac_ports(self, dc_results: DCAnalysisResults, ac_port_names: List[str]) -> np.ndarray:
        """
        Maps the DC Y-matrix to the AC port ordering for F=0 simulation points.
        This method relies solely on the self-contained `DCAnalysisResults` data contract.
        """
        num_ac_ports = len(ac_port_names)
        y_mapped = np.zeros((num_ac_ports, num_ac_ports), dtype=np.complex128)

        y_dc_qty = dc_results.y_ports_dc
        
        if y_dc_qty is None:
            return y_mapped

        dc_port_map = dc_results.dc_port_mapping
        y_dc_mag = y_dc_qty.to(self.ureg.siemens).magnitude
        
        ac_to_dc_idx_map = {ac_idx: dc_port_map.get(ac_name) for ac_idx, ac_name in enumerate(ac_port_names)}

        for r_ac, r_dc in ac_to_dc_idx_map.items():
            if r_dc is None: continue
            for c_ac, c_dc in ac_to_dc_idx_map.items():
                if c_dc is None: continue
                y_mapped[r_ac, c_ac] = y_dc_mag[r_dc, c_dc]

        for ac_idx, ac_name in enumerate(ac_port_names):
            if ac_to_dc_idx_map.get(ac_idx) is None:
                srep = dc_results.dc_supernode_mapping.get(ac_name)
                if srep and srep == dc_results.ground_supernode_name:
                    y_mapped[ac_idx, ac_idx] = LARGE_ADMITTANCE_SIEMENS
        return y_mapped