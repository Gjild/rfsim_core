# --- Modify: src/rfsim_core/analysis_tools.py ---
import logging
from typing import Dict, Any, Optional, List, Tuple, Set
import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from scipy.linalg import lu_factor, lu_solve

from .data_structures import Circuit
from .data_structures import Component as ComponentDataStructure
from .components.base import ComponentBase, DCBehaviorType, ComponentError
from .units import ureg, Quantity, pint
from .parameters import ParameterManager, ParameterError, ParameterScopeError

logger = logging.getLogger(__name__)

class DCAnalysisError(ValueError):
    """Custom exception for errors during rigorous DC analysis."""
    pass

class DCAnalyzer:
    def __init__(self, circuit: Circuit):
        if not isinstance(circuit, Circuit):
            raise TypeError("DCAnalyzer requires a valid Circuit object.")
        if not hasattr(circuit, 'sim_components'):
            raise ValueError("Input Circuit object to DCAnalyzer is not simulation-ready (missing 'sim_components').")
        if not isinstance(circuit.parameter_manager, ParameterManager):
            raise ValueError("Circuit object provided to DCAnalyzer has an uninitialized or invalid ParameterManager.")

        self.circuit: Circuit = circuit
        self.parameter_manager: ParameterManager = circuit.parameter_manager
        self.ureg = ureg
        self.ground_net_name: str = circuit.ground_net_name

        self._supernode_map: Dict[str, str] = {}
        self._ground_supernode_representative_name: Optional[str] = None
        self._supernode_graph: Optional[nx.Graph] = None

        self._supernode_to_mna_index_map: Dict[str, int] = {}
        self._dc_port_supernode_representatives: Set[str] = set()
        self._dc_port_names_ordered: List[str] = []

        logger.info(f"DCAnalyzer initialized for circuit '{circuit.name}'.")

    def get_supernode_representative_name(self, original_net_name: str) -> Optional[str]:
        if not self._supernode_map:
             logger.warning(f"Attempted to get supernode representative for '{original_net_name}' before analysis or map is empty.")
             if original_net_name in self.circuit.nets:
                 return original_net_name 
             else:
                 return None 
        return self._supernode_map.get(original_net_name, original_net_name)

    def is_ground_supernode_by_representative(self, representative_supernode_name: str) -> bool:
        if self._ground_supernode_representative_name is None:
            logger.warning("Attempted to check for ground supernode before ground representative was identified.")
            ground_rep = self.get_supernode_representative_name(self.ground_net_name)
            if ground_rep is None: return False
            return representative_supernode_name == ground_rep
        return representative_supernode_name == self._ground_supernode_representative_name

    def analyze(self) -> Dict[str, Any]:
        logger.info(f"Starting rigorous DC analysis for circuit '{self.circuit.name}'...")
        dc_behaviors = self._resolve_and_collect_dc_behaviors()
        self._build_dc_supernode_graph(dc_behaviors)
        self._identify_supernodes()
        self._assign_dc_mna_indices()
        y_dc_super_full = self._build_dc_mna_matrix(dc_behaviors)
        self._identify_dc_ports()
        y_ports_dc_qty = self._calculate_dc_y_parameters(y_dc_super_full)
        dc_port_mapping = self._build_dc_port_mapping()
        logger.info("DC analysis complete.")
        results = {
            'Y_ports_dc': y_ports_dc_qty,
            'dc_port_names_ordered': self._dc_port_names_ordered,
            'dc_port_mapping': dc_port_mapping,
            'dc_supernode_mapping': self._supernode_map.copy()
        }
        return results

    def _resolve_and_collect_dc_behaviors(self) -> Dict[str, Tuple[DCBehaviorType, Optional[Quantity]]]:
        logger.debug("Resolving component parameters and getting DC behaviors...")
        dc_behaviors: Dict[str, Tuple[DCBehaviorType, Optional[Quantity]]] = {}
        freq_dc = np.array([0.0])
        evaluation_context: Dict[Tuple[str, str], Quantity] = {}

        for comp_id, sim_comp in self.circuit.sim_components.items():
            resolved_params_dict: Dict[str, Quantity] = {}
            component_param_specs = type(sim_comp).declare_parameters()

            try:
                for internal_name in sim_comp.parameter_internal_names:
                    name_parts = internal_name.split('.', 1)
                    if len(name_parts) != 2: continue 
                    base_name = name_parts[1]
                    expected_dimension_str = component_param_specs[base_name]

                    resolved_qty = self.parameter_manager.resolve_parameter(
                        internal_name, freq_dc, expected_dimension_str, evaluation_context
                    )
                    if isinstance(resolved_qty.magnitude, np.ndarray) and resolved_qty.magnitude.size == 1:
                         val = resolved_qty.magnitude.item()
                         if np.isnan(val) or np.isinf(val):
                             logger.warning(f"Parameter '{internal_name}' resolved to {val} at DC (F=0). Component '{comp_id}' get_dc_behavior must handle this.")
                    
                    resolved_params_dict[base_name] = resolved_qty

                behavior_type, admittance_qty = sim_comp.get_dc_behavior(resolved_params_dict)
                dc_behaviors[comp_id] = (behavior_type, admittance_qty)
                admittance_str_for_log = f"{admittance_qty:~P}" if admittance_qty is not None else "N/A"
                logger.debug(f"  Component '{comp_id}': DC Behavior={behavior_type.name}, Admittance={admittance_str_for_log}")

            except ComponentError as e:
                logger.error(f"Component error getting DC behavior for '{comp_id}': {e}")
                raise 
            except ParameterError as e:
                err_msg = f"Parameter resolution failed for DC analysis of component '{comp_id}': {e}"
                logger.error(err_msg)
                raise DCAnalysisError(err_msg) from e
            except Exception as e:
                err_msg = f"Unexpected error processing component '{comp_id}' for DC behavior: {e}"
                logger.error(err_msg, exc_info=True)
                raise DCAnalysisError(err_msg) from e
        return dc_behaviors
    # ... (rest of DCAnalyzer methods _build_dc_supernode_graph, _identify_supernodes, etc. remain unchanged) ...
    def _build_dc_supernode_graph(self, dc_behaviors: Dict[str, Tuple[DCBehaviorType, Optional[Quantity]]]):
        logger.debug("Building DC supernode connectivity graph...")
        self._supernode_graph = nx.Graph()
        self._supernode_graph.add_nodes_from(self.circuit.nets.keys())

        for comp_id, (behavior_type, _) in dc_behaviors.items():
            if behavior_type == DCBehaviorType.SHORT_CIRCUIT:
                raw_comp_data = self.circuit.components.get(comp_id)
                if not raw_comp_data:
                    logger.warning(f"Component '{comp_id}' declared DC short but raw data not found. Skipping.")
                    continue

                connected_nets = []
                for port_obj in raw_comp_data.ports.values():
                    if port_obj.net:
                        connected_nets.append(port_obj.net.name)
                    else:
                         logger.warning(f"Port '{port_obj.port_id}' of DC short component '{comp_id}' is not connected to a net. Skipping this port for supernode graph.")
                
                if len(connected_nets) >= 2:
                    from itertools import combinations
                    for net1, net2 in combinations(connected_nets, 2):
                        if net1 != net2: 
                            self._supernode_graph.add_edge(net1, net2, type='dc_short', component_id=comp_id)
                            logger.debug(f"  Added DC short edge: {net1} <-> {net2} (via '{comp_id}')")
                elif len(connected_nets) < 2:
                      logger.warning(f"DC short component '{comp_id}' connects to fewer than 2 nets ({connected_nets}). It will not contribute to node merging.")
        logger.debug(f"Supernode graph built: {self._supernode_graph.number_of_nodes()} nodes, {self._supernode_graph.number_of_edges()} edges.")

    def _identify_supernodes(self):
        if self._supernode_graph is None:
            raise DCAnalysisError("Supernode graph was not built before calling _identify_supernodes.")

        logger.debug("Identifying DC supernodes (connected components)...")
        self._supernode_map = {}
        self._ground_supernode_representative_name = None
        processed_nets = set()

        connected_components_list = list(nx.connected_components(self._supernode_graph)) 
        ground_rep_found = False
        for component_nets in connected_components_list:
            component_nets_list = sorted(list(component_nets)) 
            representative_name = component_nets_list[0] 

            if self.ground_net_name in component_nets:
                if self.ground_net_name in component_nets_list: 
                    representative_name = self.ground_net_name 
                self._ground_supernode_representative_name = representative_name
                ground_rep_found = True
                logger.info(f"Ground supernode identified. Representative: '{representative_name}'. Members: {component_nets_list}")

                for net_name in component_nets:
                    self._supernode_map[net_name] = representative_name
                processed_nets.update(component_nets)
                break 
        
        if not ground_rep_found:
             if self.ground_net_name in self._supernode_graph:
                 self._supernode_map[self.ground_net_name] = self.ground_net_name
                 self._ground_supernode_representative_name = self.ground_net_name
                 processed_nets.add(self.ground_net_name)
                 logger.info(f"Ground net '{self.ground_net_name}' is isolated. Treating as its own supernode.")
             else:
                  # This case means ground_net_name was defined in circuit, but not even added as a node
                  # to self._supernode_graph. This implies self.circuit.nets might not contain it, which is an inconsistency.
                  # _supernode_graph.add_nodes_from(self.circuit.nets.keys()) should add it if it's a key in self.circuit.nets.
                  # If self.circuit.nets doesn't contain the defined ground_net_name, that's a problem earlier.
                  # For robustness, check if it was supposed to be a node:
                  if self.ground_net_name in self.circuit.nets:
                       raise DCAnalysisError(f"Ground net '{self.ground_net_name}' is defined in circuit.nets but not found in the DC connectivity graph nodes after graph construction. This is unexpected.")
                  else: # Ground net not in circuit.nets - highly problematic, should be caught by semantic validation earlier.
                       raise DCAnalysisError(f"Circuit's designated ground net '{self.ground_net_name}' is not present in the circuit's net list (self.circuit.nets). Cannot proceed with DC analysis.")


        for component_nets in connected_components_list:
             if not component_nets.isdisjoint(processed_nets): 
                 continue
            
             component_nets_list = sorted(list(component_nets))
             representative_name = component_nets_list[0] 
             logger.debug(f"Identified non-ground supernode. Representative: '{representative_name}'. Members: {component_nets_list}")
             for net_name in component_nets:
                 self._supernode_map[net_name] = representative_name
             processed_nets.update(component_nets)

        remaining_nets = set(self.circuit.nets.keys()) - processed_nets
        for net_name in sorted(list(remaining_nets)):
            if net_name not in self._supernode_map: 
                 self._supernode_map[net_name] = net_name 
                 logger.debug(f"Net '{net_name}' is isolated at DC. Treating as its own supernode.")

        all_reps = set(self._supernode_map.values())
        all_mapped_nets = set(self._supernode_map.keys()) | all_reps # Include representatives themselves
        
        # Check if all nets from the circuit are covered by the map keys or are representatives
        missing_nets = set(self.circuit.nets.keys()) - all_mapped_nets
        if missing_nets:
            logger.error(f"Internal error: The following circuit nets were not accounted for in the supernode map: {missing_nets}. "
                         f"Map keys: {set(self._supernode_map.keys())}, Map values (reps): {set(self._supernode_map.values())}")
            # Attempt to map missing nets to themselves as a fallback, but log an error
            for net_name in missing_nets:
                self._supernode_map[net_name] = net_name
                logger.warning(f"Net '{net_name}' was missing from supernode map; mapped to itself as isolated node.")
            # No raise here, allow analysis to proceed with this recovery, but it's a sign of an issue.

        logger.info(f"Supernode identification complete. Total unique supernode representatives: {len(set(self._supernode_map.values()))}")
        logger.debug(f"Supernode Map (OriginalNet -> Representative): {self._supernode_map}")

    def _assign_dc_mna_indices(self):
        if not self._supernode_map:
             raise DCAnalysisError("Supernode map is not populated. Cannot assign DC MNA indices.")
        if self._ground_supernode_representative_name is None:
             raise DCAnalysisError("Ground supernode representative was not identified. Cannot assign DC MNA indices.")

        logger.debug("Assigning DC MNA indices to supernode representatives...")
        representatives = sorted(list(set(self._supernode_map.values())))

        self._supernode_to_mna_index_map = {}
        idx_counter = 0

        self._supernode_to_mna_index_map[self._ground_supernode_representative_name] = idx_counter
        logger.debug(f"  Index {idx_counter}: '{self._ground_supernode_representative_name}' (Ground Supernode)")
        idx_counter += 1

        for rep in representatives:
            if rep != self._ground_supernode_representative_name:
                 if rep in self._supernode_to_mna_index_map: continue 
                 self._supernode_to_mna_index_map[rep] = idx_counter
                 logger.debug(f"  Index {idx_counter}: '{rep}'")
                 idx_counter += 1

        if idx_counter != len(representatives):
             # This can happen if _ground_supernode_representative_name was not in the representatives list
             # (e.g. ground net was missing from circuit.nets but defined as ground_net_name).
             # _identify_supernodes should prevent this.
             unique_reps_in_map = set(self._supernode_map.values())
             if self._ground_supernode_representative_name not in unique_reps_in_map:
                 raise DCAnalysisError(f"Internal error: Ground representative '{self._ground_supernode_representative_name}' is not among the unique representatives from supernode map {unique_reps_in_map}. "
                                       f"Cannot assign indices consistently.")
             raise DCAnalysisError(f"Internal error: Number of assigned indices ({idx_counter}) does not match number of unique representatives ({len(representatives)}). "
                                   f"Indices: {self._supernode_to_mna_index_map}, Reps: {representatives}")

        logger.info(f"DC MNA indices assigned. Total DC nodes (supernodes): {idx_counter}.")
    
    def _build_dc_mna_matrix(self, dc_behaviors: Dict[str, Tuple[DCBehaviorType, Optional[Quantity]]]) -> np.ndarray:
        num_dc_nodes = len(self._supernode_to_mna_index_map)
        if num_dc_nodes == 0:
            logger.warning("No DC supernodes found (empty circuit?). Returning empty DC MNA matrix.")
            return np.array([], dtype=complex).reshape(0, 0)

        logger.debug(f"Building DC MNA matrix ({num_dc_nodes} x {num_dc_nodes})...")
        y_dc_super_full = np.zeros((num_dc_nodes, num_dc_nodes), dtype=complex)

        for comp_id, (behavior_type, admittance_qty) in dc_behaviors.items():
            if behavior_type == DCBehaviorType.ADMITTANCE:
                if admittance_qty is None:
                     logger.error(f"Internal inconsistency: Component '{comp_id}' reported ADMITTANCE but quantity is None. Skipping.")
                     continue
                if not isinstance(admittance_qty, Quantity) or not admittance_qty.check(self.ureg.siemens):
                    # Use a more specific error message if check fails
                    dim_check_reason = "not a Quantity" if not isinstance(admittance_qty, Quantity) else f"dimensionality check against Siemens failed (is {admittance_qty.dimensionality})"
                    logger.error(f"Component '{comp_id}' provided invalid DC admittance object ({dim_check_reason}): {admittance_qty!r}. Skipping.")
                    continue
                try:
                     y_mag_complex = complex(admittance_qty.to(self.ureg.siemens).magnitude)
                except Exception as e:
                    logger.error(f"Failed to convert DC admittance {admittance_qty:~P} from component '{comp_id}' to complex Siemens value: {e}. Skipping stamp.")
                    continue

                raw_comp_data = self.circuit.components.get(comp_id)
                if not raw_comp_data:
                    logger.warning(f"Cannot find raw data for component '{comp_id}' providing DC admittance. Skipping.")
                    continue

                connected_nets = []
                for port_obj in raw_comp_data.ports.values():
                    if port_obj.net:
                        connected_nets.append(port_obj.net.name)
                    else:
                        logger.warning(f"Port '{port_obj.port_id}' of component '{comp_id}' providing DC admittance is not connected. Skipping port.")

                if len(connected_nets) == 2:
                    net1_name, net2_name = connected_nets[0], connected_nets[1]
                    try:
                        srep1 = self.get_supernode_representative_name(net1_name)
                        srep2 = self.get_supernode_representative_name(net2_name)
                        if srep1 is None or srep2 is None: raise KeyError(f"Supernode representative not found for nets '{net1_name}' or '{net2_name}'")
                        idx1 = self._supernode_to_mna_index_map[srep1]
                        idx2 = self._supernode_to_mna_index_map[srep2]
                    except KeyError as e:
                        logger.error(f"Failed to get supernode index for net key {e} connected to component '{comp_id}'. Supernode map: {self._supernode_map}, Index map: {self._supernode_to_mna_index_map}. Skipping stamp.")
                        continue
                    
                    logger.debug(f"  Stamping 2-terminal DC admittance {y_mag_complex:.3e} S from '{comp_id}' "
                                 f"between supernodes '{srep1}'(idx {idx1}) and '{srep2}'(idx {idx2})")
                    y_dc_super_full[idx1, idx1] += y_mag_complex
                    y_dc_super_full[idx2, idx2] += y_mag_complex
                    y_dc_super_full[idx1, idx2] -= y_mag_complex
                    y_dc_super_full[idx2, idx1] -= y_mag_complex

                elif len(connected_nets) == 1:
                     net1_name = connected_nets[0]
                     try:
                         srep1 = self.get_supernode_representative_name(net1_name)
                         if srep1 is None: raise KeyError(f"Supernode representative not found for net '{net1_name}'")
                         idx1 = self._supernode_to_mna_index_map[srep1]
                     except KeyError as e:
                         logger.error(f"Failed to get supernode index for net key {e} connected to 1-port component '{comp_id}'. Supernode map: {self._supernode_map}, Index map: {self._supernode_to_mna_index_map}. Skipping stamp.")
                         continue
                     logger.debug(f"  Stamping 1-terminal DC admittance {y_mag_complex:.3e} S from '{comp_id}' "
                                  f"at supernode '{srep1}'(idx {idx1})")
                     y_dc_super_full[idx1, idx1] += y_mag_complex
                else:
                     logger.warning(f"Component '{comp_id}' providing DC admittance has {len(connected_nets)} terminals. "
                                    f"Current DC analysis stamping logic only supports 1 or 2 terminals. Skipping stamp.")
        logger.info("DC MNA matrix build complete.")
        return y_dc_super_full

    def _identify_dc_ports(self):
        logger.debug("Identifying DC ports...")
        self._dc_port_supernode_representatives = set()
        self._dc_port_names_ordered = []
        supernode_rep_to_ac_ports: Dict[str, List[str]] = {}

        for ac_port_name in self.circuit.external_ports.keys():
             srep = self.get_supernode_representative_name(ac_port_name)
             if srep is None:
                 logger.warning(f"Could not find supernode for AC port '{ac_port_name}'. Skipping for DC port identification.")
                 continue
            
             if not self.is_ground_supernode_by_representative(srep):
                 if srep not in supernode_rep_to_ac_ports:
                     supernode_rep_to_ac_ports[srep] = []
                 supernode_rep_to_ac_ports[srep].append(ac_port_name)

        self._dc_port_supernode_representatives = set(supernode_rep_to_ac_ports.keys())
        dc_port_reps_sorted = sorted(list(self._dc_port_supernode_representatives))
        
        for srep in dc_port_reps_sorted:
            ac_ports_in_supernode = sorted(supernode_rep_to_ac_ports[srep])
            chosen_ac_port_name = ac_ports_in_supernode[0] 
            self._dc_port_names_ordered.append(chosen_ac_port_name)

        logger.info(f"Identified {len(self._dc_port_names_ordered)} DC port(s).")
        logger.debug(f"DC Port Supernode Representatives: {dc_port_reps_sorted}")
        logger.debug(f"Chosen AC Port Names for DC Ports (Ordered): {self._dc_port_names_ordered}")

    def _calculate_dc_y_parameters(self, y_dc_super_full: np.ndarray) -> Optional[Quantity]:
        num_dc_ports = len(self._dc_port_names_ordered)
        num_dc_nodes = y_dc_super_full.shape[0]

        if num_dc_ports == 0:
            logger.info("No DC ports identified. Returning None for Y_ports_dc.")
            return None
        if num_dc_nodes <= 1 and num_dc_ports > 0 : 
             logger.error(f"DC MNA system has <= 1 node ({num_dc_nodes}) but {num_dc_ports} DC ports identified. This is inconsistent if ports are expected.")
             # If num_dc_nodes is 1 (ground only), Y_ports_dc should be empty or None.
             # If num_dc_ports > 0, this indicates a problem.
             raise DCAnalysisError(f"DC MNA system size ({num_dc_nodes} nodes) is insufficient for identified {num_dc_ports} DC ports.")
        elif num_dc_nodes == 0 and num_dc_ports > 0: # Should not happen if num_dc_nodes comes from _supernode_to_mna_index_map
             raise DCAnalysisError(f"DC MNA system has 0 nodes but {num_dc_ports} DC ports. Inconsistency.")


        logger.debug(f"Calculating {num_dc_ports}-port DC Y-parameters via Schur complement...")
        try:
            dc_port_full_indices = []
            for name in self._dc_port_names_ordered: # These are original AC port names chosen to represent DC ports
                 srep = self.get_supernode_representative_name(name)
                 if srep is None or srep not in self._supernode_to_mna_index_map:
                      raise DCAnalysisError(f"Failed to find MNA index for DC port representative name derived from AC port '{name}' (supernode '{srep}').")
                 dc_port_full_indices.append(self._supernode_to_mna_index_map[srep])
            
            all_indices_set = set(range(num_dc_nodes))
            port_indices_set = set(dc_port_full_indices)
            
            # Ground index is always 0 in the DC MNA system built by this DCAnalyzer
            ground_mna_index = self._supernode_to_mna_index_map.get(self._ground_supernode_representative_name)
            if ground_mna_index is None: # Should not happen if _assign_dc_mna_indices worked
                raise DCAnalysisError("Internal error: Ground MNA index not found for DC Schur complement.")
            if ground_mna_index != 0: # Defensive check, should always be 0
                logger.warning(f"DC Schur: Ground MNA index is {ground_mna_index}, not 0. This is unexpected. Proceeding with {ground_mna_index} as ground ref.")

            # Internal nodes are those not ground and not ports
            internal_full_indices = sorted(list(all_indices_set - {ground_mna_index} - port_indices_set))
            
            # Ensure port indices don't include the ground index, as Schur complement is usually for non-ground ports
            if ground_mna_index in port_indices_set:
                raise DCAnalysisError(f"A DC port's MNA index ({ground_mna_index}) is the same as the ground reference index. "
                                      f"This means a port is effectively the ground supernode, which should be handled by dc_port_mapping, not by including it in Y_ports_dc. "
                                      f"Ports mapping to ground should not be in _dc_port_names_ordered.")

            num_internal_nodes_schur = len(internal_full_indices)

            logger.debug(f"  DC Port Full MNA Indices (for Schur): {dc_port_full_indices}")
            logger.debug(f"  Internal DC Full MNA Indices (for Schur): {internal_full_indices}")
            logger.debug(f"  Ground MNA Index (for Schur reference): {ground_mna_index}")

            # Build Y_PP, Y_PI, Y_IP, Y_II relative to the ground reference (index ground_mna_index)
            # We need to select rows/cols from y_dc_super_full excluding the ground_mna_index row/column first,
            # then partition that. OR, directly partition y_dc_super_full and solve a system that includes
            # port-ground and internal-ground admittances.
            # The common Schur complement formulation assumes Y_II is for internal nodes *not including ground*.
            # The matrix y_dc_super_full is already the full nodal admittance matrix.
            # Standard reduction: A = [[Y_PP, Y_PI], [Y_IP, Y_II]], solve Y_ports = Y_PP - Y_PI * inv(Y_II) * Y_IP.
            # Here, P = dc_port_full_indices, I = internal_full_indices. Ground is implicitly handled if Y_II doesn't include it.

            Y_PP = y_dc_super_full[np.ix_(dc_port_full_indices, dc_port_full_indices)]
            if num_internal_nodes_schur > 0:
                 Y_PI = y_dc_super_full[np.ix_(dc_port_full_indices, internal_full_indices)]
                 Y_IP = y_dc_super_full[np.ix_(internal_full_indices, dc_port_full_indices)]
                 Y_II = y_dc_super_full[np.ix_(internal_full_indices, internal_full_indices)]
            else: 
                 Y_PI = np.empty((num_dc_ports, 0), dtype=complex)
                 Y_IP = np.empty((0, num_dc_ports), dtype=complex)
                 Y_II = np.empty((0, 0), dtype=complex)
            
            if num_internal_nodes_schur == 0: # No internal nodes to eliminate
                y_ports_dc_mag = Y_PP
            else:
                try:
                    # Using scipy.linalg.lu_factor and lu_solve for dense matrices
                    lu_piv_II = lu_factor(Y_II, check_finite=False) # check_finite=False as Y_II can be complex
                    
                    # Solve Y_II * X = Y_IP for X
                    # Y_IP is (num_internal, num_dc_ports)
                    # X will be (num_internal, num_dc_ports)
                    X = lu_solve(lu_piv_II, Y_IP, check_finite=False)
                    
                    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                        raise np.linalg.LinAlgError("Solution of Y_II * X = Y_IP resulted in NaN/Inf during DC Schur.")
                    
                    term2 = Y_PI @ X # Y_PI is (num_dc_ports, num_internal)
                    y_ports_dc_mag = Y_PP - term2

                except (np.linalg.LinAlgError, ValueError) as e:
                    logger.error(f"Linear algebra error during DC Schur complement (LU Factor/Solve for Y_II): {e}. "
                                 f"Y_II shape={Y_II.shape} might be singular or ill-conditioned. Y_II = {Y_II if Y_II.size < 10 else 'large'}")
                    raise DCAnalysisError(f"Failed to compute DC Schur complement via LU: {e}") from e

            if y_ports_dc_mag.shape != (num_dc_ports, num_dc_ports):
                 raise DCAnalysisError(f"Internal error: Calculated Y_ports_dc has wrong shape {y_ports_dc_mag.shape}. Expected ({num_dc_ports}, {num_dc_ports}).")
            
            y_ports_dc_qty = Quantity(y_ports_dc_mag, self.ureg.siemens)
            logger.info(f"Successfully calculated DC Y-parameters ({num_dc_ports}x{num_dc_ports}).")
            return y_ports_dc_qty

        except Exception as e:
            logger.error(f"Error during DC Y-parameter calculation: {e}", exc_info=True)
            if isinstance(e, DCAnalysisError): raise
            raise DCAnalysisError(f"Unexpected error calculating DC Y-parameters: {e}") from e

    def _build_dc_port_mapping(self) -> Dict[str, Optional[int]]:
        logger.debug("Building mapping from original AC ports to DC analysis results...")
        dc_port_mapping: Dict[str, Optional[int]] = {}
        dc_port_name_to_index = {name: idx for idx, name in enumerate(self._dc_port_names_ordered)}
        srep_to_chosen_name: Dict[str, str] = {}
        for chosen_name in self._dc_port_names_ordered: # chosen_name is an original AC port name
            srep = self.get_supernode_representative_name(chosen_name)
            if srep: srep_to_chosen_name[srep] = chosen_name

        for ac_port_name in self.circuit.external_ports.keys():
            srep = self.get_supernode_representative_name(ac_port_name)
            if srep is None:
                 logger.warning(f"Could not find supernode representative for AC port '{ac_port_name}' during mapping. Mapping to None.")
                 dc_port_mapping[ac_port_name] = None
            elif self.is_ground_supernode_by_representative(srep):
                 logger.debug(f"  AC Port '{ac_port_name}' mapped to GROUND supernode '{srep}'. Mapping to None.")
                 dc_port_mapping[ac_port_name] = None
            # Check if this srep corresponds to one of the chosen DC port representatives
            elif srep in self._dc_port_supernode_representatives:
                 # Find which chosen AC port name represents this srep
                 chosen_name_for_srep = srep_to_chosen_name.get(srep)
                 if chosen_name_for_srep and chosen_name_for_srep in dc_port_name_to_index:
                     dc_index = dc_port_name_to_index[chosen_name_for_srep]
                     dc_port_mapping[ac_port_name] = dc_index
                     logger.debug(f"  AC Port '{ac_port_name}' mapped to DC Port (represented by AC port '{chosen_name_for_srep}') (Index {dc_index}) via supernode '{srep}'.")
                 else:
                      logger.error(f"Internal inconsistency: Supernode '{srep}' for AC port '{ac_port_name}' is marked as a DC port representative, "
                                   f"but couldn't find its chosen AC port name ('{chosen_name_for_srep}') or that name's index in dc_port_name_to_index. "
                                   f"srep_to_chosen_name: {srep_to_chosen_name}, dc_port_name_to_index: {dc_port_name_to_index}. Mapping to None.")
                      dc_port_mapping[ac_port_name] = None
            else: # srep is not ground and not one of the DC port representatives
                 logger.debug(f"  AC Port '{ac_port_name}' merged into INTERNAL supernode '{srep}' (not chosen as a DC port representative). Mapping to None.")
                 dc_port_mapping[ac_port_name] = None
        logger.info("DC port mapping complete.")
        return dc_port_mapping


class TopologyAnalysisError(ValueError):
    """Custom exception for errors during topological analysis (e.g., floating nodes)."""
    pass

class TopologyAnalyzer:
    """
    Performs pre-sweep topological analysis to identify structurally floating nodes
    for AC analysis by considering ideal open circuits.
    """
    def __init__(self, circuit: Circuit):
        if not isinstance(circuit, Circuit):
            raise TypeError("TopologyAnalyzer requires a valid Circuit object.")
        if not hasattr(circuit, 'sim_components') or not hasattr(circuit, 'parameter_manager'):
            raise ValueError("Input Circuit object to TopologyAnalyzer is not simulation-ready (missing attributes).")
        
        self.circuit: Circuit = circuit
        self.parameter_manager: ParameterManager = circuit.parameter_manager
        self.ureg = ureg
        self._structurally_open_comp_ids: Optional[Set[str]] = None
        self._ac_graph: Optional[nx.Graph] = None # Cache the graph
        logger.info(f"TopologyAnalyzer initialized for circuit '{circuit.name}'.")

    def _resolve_and_identify_structurally_open_components(self) -> Set[str]:
        if self._structurally_open_comp_ids is not None:
            return self._structurally_open_comp_ids

        logger.debug("Identifying structurally open components for AC topology...")
        open_comp_ids: Set[str] = set()
        
        for comp_id, sim_comp in self.circuit.sim_components.items():
            component_param_specs = type(sim_comp).declare_parameters()
            resolved_constant_params: Dict[str, Quantity] = {}
            
            try:
                for base_param_name in component_param_specs.keys():
                    internal_name = f"{comp_id}.{base_param_name}"
                    is_const = False
                    try:
                        is_const = self.parameter_manager.is_constant(internal_name)
                    except ParameterScopeError:
                        logger.debug(f"Parameter '{internal_name}' not found for component '{comp_id}' structural check.")
                        continue 

                    if is_const:
                        try:
                            const_val_qty = self.parameter_manager.get_constant_value(internal_name)
                            resolved_constant_params[base_param_name] = const_val_qty
                        except ParameterError as pe:
                            logger.warning(f"Could not get constant value for '{internal_name}' of component '{comp_id}' (reason: {pe}).")
                    else:
                        logger.debug(f"Parameter '{internal_name}' for component '{comp_id}' is not constant.")
                
                if sim_comp.is_structurally_open(resolved_constant_params):
                    logger.info(f"Component '{comp_id}' identified as structurally open based on constant parameters.")
                    open_comp_ids.add(comp_id)
                else:
                    logger.debug(f"Component '{comp_id}' is NOT structurally open based on its constant parameters.")

            except ComponentError as e:
                logger.warning(f"ComponentError during is_structurally_open check for '{comp_id}': {e}. Component will not be considered structurally open.")
            except ParameterError as e: 
                logger.warning(f"ParameterError during constant param resolution for '{comp_id}' structural check: {e}. Component will not be considered structurally open.")
            except Exception as e:
                logger.error(f"Unexpected error checking structural openness for component '{comp_id}': {e}", exc_info=True)
        
        self._structurally_open_comp_ids = open_comp_ids
        logger.info(f"Structurally open components identified: {self._structurally_open_comp_ids if self._structurally_open_comp_ids else 'None'}")
        return self._structurally_open_comp_ids

    def _build_ac_graph(self) -> nx.Graph:
        if self._ac_graph is not None:
            return self._ac_graph

        structurally_open_components = self._resolve_and_identify_structurally_open_components()
        
        ac_graph = nx.Graph()
        all_net_names_in_circuit = set(self.circuit.nets.keys())
        if not all_net_names_in_circuit:
            logger.warning("Circuit has no nets defined. AC graph will be empty.")
            self._ac_graph = ac_graph
            return ac_graph
            
        ac_graph.add_nodes_from(all_net_names_in_circuit)
        logger.debug(f"Added {len(all_net_names_in_circuit)} nets as nodes to AC connectivity graph.")

        for comp_id, raw_comp_data in self.circuit.components.items():
            if comp_id in structurally_open_components:
                logger.debug(f"Skipping structurally open component '{comp_id}' for AC graph edges.")
                continue

            connected_nets_for_this_comp: List[str] = []
            for port_obj in raw_comp_data.ports.values():
                if port_obj.net and port_obj.net.name in all_net_names_in_circuit:
                    connected_nets_for_this_comp.append(port_obj.net.name)
                elif port_obj.net :
                     logger.warning(f"Net '{port_obj.net.name}' for port '{port_obj.port_id}' of component '{comp_id}' "
                                   f"is not in the circuit's registered nets list. Skipping for graph edge.")
            
            unique_connected_nets = sorted(list(set(connected_nets_for_this_comp)))
            if len(unique_connected_nets) >= 2:
                from itertools import combinations
                for net1_name, net2_name in combinations(unique_connected_nets, 2):
                    ac_graph.add_edge(net1_name, net2_name, component_id=comp_id)
                    logger.debug(f"  Added AC graph edge: {net1_name} <-> {net2_name} (via non-open '{comp_id}')")
            elif len(unique_connected_nets) == 1:
                 logger.debug(f"Component '{comp_id}' connects to only one unique net '{unique_connected_nets[0]}'. No edges added to AC graph from it.")
        
        self._ac_graph = ac_graph
        return ac_graph

    def get_active_nets(self) -> Set[str]:
        logger.info(f"Starting AC topological analysis for active nets in '{self.circuit.name}'...")
        
        ac_graph = self._build_ac_graph() # Builds or returns cached graph
        if not ac_graph.nodes:
            logger.info("AC graph has no nodes. Returning empty set of active nets.")
            return set()

        sources: Set[str] = set()
        if self.circuit.ground_net_name and self.circuit.ground_net_name in ac_graph:
            sources.add(self.circuit.ground_net_name)
            logger.debug(f"Added ground net '{self.circuit.ground_net_name}' as traversal source.")
        
        for port_net_name in self.circuit.external_ports.keys():
            if port_net_name in ac_graph:
                sources.add(port_net_name)
                logger.debug(f"Added external port net '{port_net_name}' as traversal source.")
            else: # Port net defined but not in graph (e.g., if circuit was empty and parsed to no nets)
                logger.warning(f"External port net '{port_net_name}' defined in circuit.external_ports, but not found in AC graph nodes. It cannot be a traversal source.")


        if not sources:
            if self.circuit.external_ports or self.circuit.ground_net_name:
                    logger.warning(f"No valid traversal sources (ground or external ports) found in the AC connectivity graph nodes. All ports/ground may be floating or isolated by structural opens.")
            else:
                    logger.info("No ground or external port sources for AC graph traversal (none defined in circuit). Circuit may be entirely floating or empty.")
            return set()

        active_nets: Set[str] = set()
        for source_node in sources:
            if source_node in ac_graph: 
                active_nets.add(source_node)
                try:
                    component_nodes = nx.node_connected_component(ac_graph, source_node)
                    active_nets.update(component_nodes)
                except nx.NetworkXError as e: 
                    logger.warning(f"NetworkX error during connected component search from source '{source_node}': {e}. This source might be isolated.")
                except KeyError as e: 
                    logger.warning(f"Source node '{source_node}' not found in graph during component search. Error: {e}")
            else: # Should not happen if sources are derived carefully from ac_graph.nodes()
                logger.debug(f"Traversal source '{source_node}' was not found in the built AC graph. Skipping.")

        # The block of code previously here (lines 483-499 in the original file) that
        # conditionally discarded self.circuit.ground_net_name has been removed.

        if not active_nets and (self.circuit.ground_net_name or self.circuit.external_ports):
            logger.warning("AC topological analysis resulted in an empty set of active nets, despite defined ground or ports. Check circuit connectivity and structural opens.")
        elif active_nets:
            logger.info(f"AC topological analysis complete. Found {len(active_nets)} active nets.")
            logger.debug(f"Active nets for AC: {sorted(list(active_nets))}")
        else: 
                logger.info("AC topological analysis complete. No active nets found (e.g. empty circuit or all sources isolated).")

        return active_nets
    
    def are_ports_connected_to_active_ground(self) -> bool:
        """
        Checks if any active external port is connected to the active ground net
        within the AC connectivity graph (_ac_graph).
        "Active ground" means self.circuit.ground_net_name is in self._ac_graph.
        "Active external port" means port_net_name is in self._ac_graph.

        Returns:
            True if ground is not defined, or no external ports are defined/active in graph,
                    or if at least one active external port is connected to active ground.
            False if ground is active in graph, and external ports are active in graph,
                    but no active external port is connected to the active ground.
        Raises:
            TopologyAnalysisError: If _ac_graph is not built (should not happen if called correctly).
        """
        if not self.circuit.external_ports:
            logger.debug("Ports-ground connectivity: No external ports defined. Condition met (vacuously true).")
            return True
        # circuit.ground_net_name should always be a valid string for a built circuit
        if not self.circuit.ground_net_name: 
            logger.error("Ports-ground connectivity: Circuit ground_net_name is not set. Cannot check.")
            # This indicates a more fundamental issue if ground_net_name can be None/empty here.
            # For now, treat as if ground is not connectable.
            return False 
        
        if self._ac_graph is None:
            logger.error("Ports-ground connectivity: _ac_graph not built. Cannot check.")
            raise TopologyAnalysisError("_ac_graph not built before checking port-ground connectivity.")

        ground_name = self.circuit.ground_net_name
        if ground_name not in self._ac_graph:
            logger.info(f"Ports-ground connectivity: Ground net '{ground_name}' not in AC graph. Considered disconnected from ports.")
            return False 

        try:
            ground_component_nodes = nx.node_connected_component(self._ac_graph, ground_name)
        except (nx.NetworkXError, KeyError): 
            logger.warning(f"Ports-ground connectivity: Could not get connected component for ground '{ground_name}'. Considered disconnected.")
            return False

        active_ports_in_graph = {
            p_name for p_name in self.circuit.external_ports.keys() if p_name in self._ac_graph
        }
        
        if not active_ports_in_graph:
            logger.debug("Ports-ground connectivity: No external ports are active in the AC graph. Condition met (vacuously true from port side).")
            return True

        for port_net_name in active_ports_in_graph:
            if port_net_name in ground_component_nodes:
                logger.debug(f"Ports-ground connectivity: Port '{port_net_name}' is connected to ground '{ground_name}'. Condition met.")
                return True
        
        logger.info(f"Ports-ground connectivity: No active port is connected to ground '{ground_name}'. Ports considered: {active_ports_in_graph}.")
        return False