# src/rfsim_core/analysis/tools.py

"""
Provides high-level, cache-aware analysis services for synthesized Circuit objects.
"""

import logging
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple, KeysView, Any

import pint
import networkx as nx
import numpy as np
from scipy.linalg import lu_factor, lu_solve

from ..components.base import ComponentBase, DCBehaviorType
# --- REVISED IMPORTS for Architectural Perfection ---
from ..components.exceptions import ComponentError
from ..components.capabilities import (
    IDcContributor, ITopologyContributor, IConnectivityProvider
)
# NEW: Import SubcircuitInstance for the defensive type check in the purified DC MNA builder.
from ..components.subcircuit import SubcircuitInstance
from ..data_structures import Circuit
from ..errors import FrameworkLogicError # For internal contract violations
from ..parameters import ParameterError, ParameterManager
from ..units import Quantity, ureg

from ..cache.keys import create_dc_analysis_key, create_topology_key
from ..cache.service import SimulationCache
from .exceptions import DCAnalysisError, TopologyAnalysisError
from .results import DCAnalysisResults, TopologyAnalysisResults

logger = logging.getLogger(__name__)

DcAdmittancePayload = Any


class TopologyAnalyzer:
    """
    Performs and caches topological analysis for a given circuit configuration.
    This is a stateless service that operates on a circuit and a cache.
    """
    def __init__(self, circuit: Circuit, cache: SimulationCache):
        """
        Initializes the TopologyAnalyzer service.

        Args:
            circuit: The simulation-ready Circuit object to be analyzed.
            cache: The centralized SimulationCache instance for this run.
        """
        if not isinstance(circuit, Circuit) or not circuit.parameter_manager:
            raise TypeError("TopologyAnalyzer requires a valid, simulation-ready Circuit object.")
        if not isinstance(cache, SimulationCache):
            raise TypeError("TopologyAnalyzer requires a valid SimulationCache instance.")

        self.circuit: Circuit = circuit
        self.parameter_manager: ParameterManager = circuit.parameter_manager
        self._analysis_results: Optional[TopologyAnalysisResults] = None
        self.cache: SimulationCache = cache
        logger.debug(f"TopologyAnalyzer service initialized for circuit '{circuit.hierarchical_id}'.")

    def analyze(self) -> TopologyAnalysisResults:
        """
        Performs a full topological analysis, leveraging the injected cache service.

        Returns:
            The formal, immutable TopologyAnalysisResults object for this circuit.
        """
        # 1. Check instance-level short-circuit cache first.
        if self._analysis_results is not None:
            return self._analysis_results

        # 2. Generate the definitive cache key using the centralized key factory.
        cache_key = create_topology_key(self.circuit)

        # 3. Query the persistent 'process' scope of the centralized cache.
        cached_results = self.cache.get(key=cache_key, scope='process')
        if cached_results:
            if not isinstance(cached_results, TopologyAnalysisResults):
                logger.warning(
                    f"Cache integrity issue: expected TopologyAnalysisResults but got "
                    f"{type(cached_results)}. Re-computing."
                )
            else:
                self._analysis_results = cached_results
                return cached_results

        # --- On Cache Miss: Perform the Full Analysis ---
        logger.debug(f"Cache MISS for topology of '{self.circuit.name}'. Performing full analysis.")

        try:
            open_comps = self._resolve_and_identify_structurally_open_components()
            # The private helpers are now architecturally pure.
            ac_graph = self._build_ac_graph(open_comps)
            active_nets = self._compute_active_nets(ac_graph)
            port_connectivity = self._compute_external_port_connectivity(ac_graph)

            results = TopologyAnalysisResults(
                structurally_open_components=open_comps,
                ac_graph=ac_graph,
                active_nets=active_nets,
                external_port_connectivity=port_connectivity,
            )

            self.cache.put(key=cache_key, value=results, scope='process')
            self._analysis_results = results
            return results
        except Exception as e:
            raise TopologyAnalysisError(
                hierarchical_context=self.circuit.hierarchical_id,
                details=f"An unexpected error occurred during topology analysis: {e}"
            ) from e

    def get_active_nets(self) -> Set[str]:
        """Returns the set of active nets for the circuit by running the analysis."""
        return self.analyze().active_nets

    def are_ports_connected_to_active_ground(self) -> bool:
        """Checks if any active external port has a path to the active ground."""
        results = self.analyze()
        ac_graph = results.ac_graph
        if not self.circuit.external_ports: return True
        ground_name = self.circuit.ground_net_name
        if ground_name not in ac_graph: return False

        return any(nx.has_path(ac_graph, port, ground_name) for port in self.circuit.external_ports if port in ac_graph)

    def _resolve_and_identify_structurally_open_components(self) -> Set[str]:
        """Determines which leaf components are structural opens based on their constant parameters."""
        open_comp_ids: Set[str] = set()
        for comp_id, sim_comp in self.circuit.sim_components.items():
            # Query for the capability. If the component does not provide it, skip.
            topology_contributor = sim_comp.get_capability(ITopologyContributor)
            if not topology_contributor:
                continue

            resolved_constant_params: Dict[str, Quantity] = {}
            for base_param_name, fqn in zip(type(sim_comp).declare_parameters(), sim_comp.parameter_fqns):
                if self.parameter_manager.is_constant(fqn):
                    try:
                        resolved_constant_params[base_param_name] = self.parameter_manager.get_constant_value(fqn)
                    except ParameterError as e:
                        logger.warning(f"Could not resolve constant '{fqn}' for structural open check: {e}")

            if topology_contributor.is_structurally_open(sim_comp, resolved_constant_params):
                open_comp_ids.add(comp_id)
        return open_comp_ids

    def _build_ac_graph(self, structurally_open_components: Set[str]) -> nx.Graph:
        """
        Builds the AC connectivity graph.
        It is fully component-agnostic and relies only on formal capabilities and API contracts.
        """
        ac_graph = nx.Graph()
        ac_graph.add_nodes_from(self.circuit.nets.keys())
        parent_ground_net = self.circuit.ground_net_name # Get the parent's ground name

        for sim_comp in self.circuit.sim_components.values():
            if sim_comp.instance_id in structurally_open_components:
                continue

            # 1. Query for the universal connectivity capability.
            connectivity_provider = sim_comp.get_capability(IConnectivityProvider)
            if not connectivity_provider:
                logger.debug(f"Component '{sim_comp.fqn}' provides no connectivity. Skipping.")
                continue

            # 2. Get the port-to-net mapping via the new, formal API contract.
            try:
                port_map = sim_comp.get_port_net_mapping()
            except Exception as e:
                raise FrameworkLogicError(
                    f"Component '{sim_comp.fqn}' failed to provide its port-to-net mapping: {e}"
                ) from e
            
            # 3. Get the connectivity from the capability.
            component_connectivity = connectivity_provider.get_connectivity(sim_comp)

            # 4. Add edges to the graph.
            for port1, port2 in component_connectivity:
                # Resolve port names to the parent circuit's net names.
                # If a subcircuit returns its own ground name, map it to the parent's ground.
                net1_raw = port_map.get(port1)
                net2_raw = port_map.get(port2)

                # If the sub-component uses its own ground name as a token...
                # Check if the component is a subcircuit and if the port name matches its internal ground
                if isinstance(sim_comp, SubcircuitInstance):
                    sub_ground_name = sim_comp.sub_circuit_object.ground_net_name
                    if port1 == sub_ground_name: net1_raw = parent_ground_net
                    if port2 == sub_ground_name: net2_raw = parent_ground_net
                
                # Check if the resolved nets are actually in the parent graph before adding an edge.
                if net1_raw and net2_raw and net1_raw in ac_graph and net2_raw in ac_graph:
                    ac_graph.add_edge(net1_raw, net2_raw)

        return ac_graph

    def _compute_active_nets(self, ac_graph: nx.Graph) -> Set[str]:
        """Computes the set of all nets connected to ground or an external port."""
        if not ac_graph.nodes: return set()

        sources = {p for p in self.circuit.external_ports if p in ac_graph}
        if self.circuit.ground_net_name in ac_graph:
            sources.add(self.circuit.ground_net_name)
        if not sources: return set()

        active_nets: Set[str] = set()
        for source in sources:
            if source in ac_graph and not nx.is_isolate(ac_graph, source):
                active_nets.update(nx.node_connected_component(ac_graph, source))
        return active_nets

    def _compute_external_port_connectivity(self, ac_graph: nx.Graph) -> List[Tuple[str, str]]:
        """Computes which pairs of this circuit's external ports are conductively connected."""
        ext_ports = list(self.circuit.external_ports.keys())
        connected_pairs = []
        for port1, port2 in combinations(sorted(ext_ports), 2):
            if port1 in ac_graph and port2 in ac_graph and nx.has_path(ac_graph, port1, port2):
                connected_pairs.append((port1, port2))
        return connected_pairs


class DCAnalyzer:
    """
    Performs a rigorous DC (F=0) analysis on a synthesized circuit.
    This service is now implemented as a stateless functional pipeline,
    dramatically improving its verifiability and testability. It is
    completely component-agnostic.
    """
    def __init__(self, circuit: Circuit, cache: SimulationCache):
        """
        Initializes the DCAnalyzer service. The instance only holds injected
        dependencies, not transient state.
        """
        if not isinstance(circuit, Circuit) or not circuit.parameter_manager:
            raise TypeError("DCAnalyzer requires a valid, simulation-ready Circuit object.")
        if not isinstance(cache, SimulationCache):
            raise TypeError("DCAnalyzer requires a valid SimulationCache instance.")

        self.circuit: Circuit = circuit
        self.parameter_manager: ParameterManager = circuit.parameter_manager
        self.cache: SimulationCache = cache
        self.ureg = ureg
        logger.debug(f"DCAnalyzer service initialized for circuit '{circuit.hierarchical_id}'.")

    def analyze(self) -> DCAnalysisResults:
        """
        Executes the full DC analysis pipeline, leveraging the injected cache.
        This method orchestrates calls to pure, stateless helper functions and
        includes robust, context-aware exception handling.
        """
        cache_key = create_dc_analysis_key(self.circuit)
        cached_results = self.cache.get(key=cache_key, scope='process')
        if cached_results:
            if not isinstance(cached_results, DCAnalysisResults):
                logger.warning(
                    f"Cache integrity issue: expected DCAnalysisResults but got "
                    f"{type(cached_results)}. Re-computing."
                )
            else:
                return cached_results

        # --- On Cache Miss: Execute the Stateless Functional Pipeline ---
        logger.info(f"Cache MISS. Starting rigorous DC analysis for circuit '{self.circuit.hierarchical_id}'...")

        try:
            # Data flows cleanly from one pure function to the next.
            dc_behaviors = self._resolve_and_collect_dc_behaviors()

            supernode_graph = self._build_dc_supernode_graph(
                dc_behaviors, self.circuit.sim_components, self.circuit.nets.keys()
            )
            s_map, gnd_s_rep = self._identify_supernodes(
                supernode_graph, self.circuit.nets.keys(), self.circuit.ground_net_name
            )
            s_to_mna_idx_map = self._assign_dc_mna_indices(s_map, gnd_s_rep)

            y_dc_super_full = self._build_dc_mna_matrix(
                dc_behaviors, self.circuit.sim_components, s_map, s_to_mna_idx_map
            )

            dc_port_names = self._identify_dc_ports(
                self.circuit.external_ports.keys(), s_map, gnd_s_rep
            )

            y_ports_dc_qty = self._calculate_dc_y_parameters(
                y_dc_super_full, dc_port_names, self.circuit.external_ports.keys(), s_map, s_to_mna_idx_map
            )

            dc_port_mapping = self._build_dc_port_mapping(
                self.circuit.external_ports.keys(), dc_port_names, s_map, gnd_s_rep
            )
        # This block correctly distinguishes between known and unknown failure modes.
        except ComponentError as e:
            # Wrap known ComponentError with context.
            raise DCAnalysisError(
                hierarchical_context=self.circuit.hierarchical_id,
                details=f"DC analysis failed due to a component error. See original error for details."
            ) from e
        except DCAnalysisError:
            # If the error is already a DCAnalysisError (e.g., from a failed
            # Schur complement), it already has context. Just re-raise it.
            raise
        except Exception as e:
            # Any other exception type (ValueError, KeyError, etc.) is a truly
            # unexpected internal framework error. We wrap it to provide context
            # and signal that this might be a bug in the simulator itself.
            raise DCAnalysisError(
                hierarchical_context=self.circuit.hierarchical_id,
                details=f"An unexpected internal error occurred during DC analysis: {e}"
            ) from e
        # --- END OF EXCEPTION HANDLING ---

        logger.info(f"DC analysis for '{self.circuit.hierarchical_id}' complete.")

        results = DCAnalysisResults(
            y_ports_dc=y_ports_dc_qty,
            dc_port_names_ordered=dc_port_names,
            dc_port_mapping=dc_port_mapping,
            dc_supernode_mapping=s_map,
            ground_supernode_name=gnd_s_rep
        )

        self.cache.put(key=cache_key, value=results, scope='process')
        return results

    # --- Stateless Helper Methods ---

    def _resolve_and_collect_dc_behaviors(self) -> Dict[str, Tuple[DCBehaviorType, Optional[Quantity]]]:
        """Evaluates all parameters at F=0 and collects the DC behavior from each component."""
        logger.debug(f"[{self.circuit.hierarchical_id}] Evaluating all parameters for DC (F=0)...")
        try:
            all_dc_params = self.parameter_manager.evaluate_all(np.array([0.0]))
        except ParameterError as e:
            raise DCAnalysisError(
                hierarchical_context=self.circuit.hierarchical_id,
                details=f"Failed to evaluate parameters for DC analysis: {e}"
            ) from e

        dc_behaviors: Dict[str, Tuple[DCBehaviorType, Optional[Quantity]]] = {}
        for comp_id, sim_comp in self.circuit.sim_components.items():
            try:
                dc_contributor = sim_comp.get_capability(IDcContributor)
                if dc_contributor:
                    behavior_type, admittance_qty = dc_contributor.get_dc_behavior(sim_comp, all_dc_params)
                    dc_behaviors[comp_id] = (behavior_type, admittance_qty)
                else:
                    dc_behaviors[comp_id] = (DCBehaviorType.OPEN_CIRCUIT, None)
            except ComponentError as e:
                raise e
            except KeyError as e:
                details = f"Error getting DC behavior for component '{sim_comp.fqn}': {e}"
                logger.error(details, exc_info=True)
                raise DCAnalysisError(hierarchical_context=self.circuit.hierarchical_id, details=details) from e
        return dc_behaviors

    def _build_dc_supernode_graph(
        self,
        dc_behaviors: Dict[str, Tuple[DCBehaviorType, Optional[Quantity]]],
        sim_components: Dict[str, ComponentBase],
        all_net_names: KeysView[str]
    ) -> nx.Graph:
        """Constructs a graph where edges represent ideal DC shorts between nets. (STATELESS)"""
        logger.debug(f"[{self.circuit.hierarchical_id}] Building DC supernode connectivity graph...")
        supernode_graph = nx.Graph()
        supernode_graph.add_nodes_from(all_net_names)

        for comp_id, (behavior_type, _) in dc_behaviors.items():
            if behavior_type != DCBehaviorType.SHORT_CIRCUIT:
                continue

            sim_comp = sim_components[comp_id]
            try:
                port_map = sim_comp.get_port_net_mapping()
            except Exception:
                continue

            connected_nets = [net_name for net_name in port_map.values() if net_name in all_net_names]
            if len(connected_nets) >= 2:
                for net1, net2 in combinations(sorted(list(set(connected_nets))), 2):
                    supernode_graph.add_edge(net1, net2, type='dc_short', component_id=comp_id)
        return supernode_graph

    def _identify_supernodes(
        self, supernode_graph: nx.Graph, all_net_names: KeysView[str], ground_net_name: str
    ) -> Tuple[Dict[str, str], Optional[str]]:
        """Finds connected components in the shorting graph to define supernodes. (STATELESS)"""
        supernode_map = {net: net for net in all_net_names}
        for component_nets in nx.connected_components(supernode_graph):
            is_gnd_component = ground_net_name in component_nets
            representative_name = ground_net_name if is_gnd_component else sorted(list(component_nets))[0]
            for net_name in component_nets:
                supernode_map[net_name] = representative_name
        
        ground_supernode_rep = supernode_map.get(ground_net_name)
        return supernode_map, ground_supernode_rep

    def _assign_dc_mna_indices(
        self, supernode_map: Dict[str, str], ground_supernode_rep: Optional[str]
    ) -> Dict[str, int]:
        """Assigns a unique integer index to each supernode for MNA matrix assembly. (STATELESS)"""
        if not supernode_map:
            raise DCAnalysisError(hierarchical_context=self.circuit.hierarchical_id, details="Supernode map not populated.")
        if ground_supernode_rep is None:
            raise DCAnalysisError(hierarchical_context=self.circuit.hierarchical_id, details="Ground supernode not identified.")
        
        supernode_to_mna_index_map: Dict[str, int] = {}
        representatives = sorted(list(set(supernode_map.values())))
        supernode_to_mna_index_map[ground_supernode_rep] = 0
        idx_counter = 1
        for rep in representatives:
            if rep != ground_supernode_rep:
                supernode_to_mna_index_map[rep] = idx_counter
                idx_counter += 1
        return supernode_to_mna_index_map

    def _build_dc_mna_matrix(
        self,
        dc_behaviors: Dict[str, Tuple[DCBehaviorType, Optional[DcAdmittancePayload]]],
        sim_components: Dict[str, ComponentBase],
        supernode_map: Dict[str, str],
        s_to_mna_idx_map: Dict[str, int]
    ) -> np.ndarray:
        """
        Assembles the full supernode MNA admittance matrix from component contributions.

        This method is stateless and completely component-agnostic. It relies solely
        on the formal, explicit contract of the `IDcContributor` capability and is fully
        decoupled from any specific component implementation details (e.g., SubcircuitInstance).

        The contract for a DC ADMITTANCE behavior is as follows:
        - For a simple 2-port element, the payload should be a scalar `pint.Quantity`.
        - For an N-port element, the payload MUST be a tuple of (`pint.Quantity`, `List[str]`),
        where the Quantity contains a 2D matrix and the list defines its port ordering.

        This method validates the received payload against this contract, providing
        actionable diagnostics for any violation.
        """
        num_dc_nodes = len(s_to_mna_idx_map)
        if num_dc_nodes == 0:
            return np.array([], dtype=complex).reshape(0, 0)

        y_dc_super_full = np.zeros((num_dc_nodes, num_dc_nodes), dtype=complex)

        for comp_id, (behavior_type, payload) in dc_behaviors.items():
            if behavior_type != DCBehaviorType.ADMITTANCE or payload is None:
                continue

            sim_comp = sim_components[comp_id]
            port_map = sim_comp.get_port_net_mapping()

            # Validate and dispatch based on payload type (enforces IDcContributor contract)
            if isinstance(payload, tuple):  # --- N-port Admittance Matrix Contract Path ---

                # == Contract Check 1: Tuple Structure ==
                # The contract for an N-port admittance is an exact two-element tuple.
                # We must verify its length BEFORE attempting to unpack it to prevent a
                # raw, uninformative `ValueError`.
                if not (len(payload) == 2 and isinstance(payload[1], list)):
                    raise ComponentError(
                        component_fqn=sim_comp.fqn,
                        details=(
                            "Component provided a tuple payload for DC admittance, but it had an invalid structure. "
                            "The contract requires a (pint.Quantity, List[str]) tuple, but the structure was malformed."
                        )
                    )

                # Unpacking is now guaranteed to be safe.
                admittance_qty, port_names = payload

                # == Contract Check 2: Payload Content Type ==
                # After unpacking, we verify that the contents are of the expected types.
                # The first element MUST be a `pint.Quantity`.
                if not isinstance(admittance_qty, Quantity):
                    raise ComponentError(
                        component_fqn=sim_comp.fqn,
                        details=(
                            "The first element of the N-port DC admittance tuple must be a pint.Quantity, "
                            f"but got type '{type(admittance_qty).__name__}'."
                        )
                    )

                # == Contract Check 3: Dimensionality ==
                # The framework's core solver operates in admittance. We verify the physical
                # dimension before any numerical operations. This prevents dimensionally
                # inconsistent data from corrupting the MNA matrix. The `.to()` call will
                # raise a `pint.DimensionalityError` if the units are incompatible.
                try:
                    y_mag = admittance_qty.to(self.ureg.siemens).magnitude
                except pint.DimensionalityError as e:
                    raise ComponentError(
                        component_fqn=sim_comp.fqn,
                        details=(
                            f"The DC admittance matrix has incorrect physical dimensions. Expected [admittance], "
                            f"but got {admittance_qty.dimensionality}. Cannot convert '{admittance_qty.units}' to siemens."
                        )
                    ) from e

                # == Contract Check 4: Numerical Shape ==
                # For an N-port component, the admittance must be a 2D matrix (N x N).
                # We verify the number of dimensions of the underlying NumPy array.
                if not (isinstance(y_mag, np.ndarray) and y_mag.ndim == 2):
                    raise ComponentError(
                        component_fqn=sim_comp.fqn,
                        details=(
                            "Component provided a tuple payload for DC admittance, which implies an N-port matrix, "
                            f"but the Quantity's magnitude was not a 2D NumPy array. Got ndim={y_mag.ndim}."
                        )
                    )

                # --- Stamping Logic (Now Guaranteed Safe) ---
                parent_indices = [
                    s_to_mna_idx_map[supernode_map[port_map[p]]]
                    for p in port_names
                ]

                for i, r_idx in enumerate(parent_indices):
                    for j, c_idx in enumerate(parent_indices):
                        y_dc_super_full[r_idx, c_idx] += y_mag[i, j]

            elif isinstance(payload, Quantity):  # --- Scalar Admittance Contract Path ---

                # == Contract Check 1: Dimensionality ==
                # Same as the N-port case, we first ensure dimensional correctness.
                try:
                    y_mag = payload.to(self.ureg.siemens).magnitude
                except pint.DimensionalityError as e:
                    raise ComponentError(
                        component_fqn=sim_comp.fqn,
                        details=(
                            f"The scalar DC admittance value has incorrect physical dimensions. Expected [admittance], "
                            f"but got {payload.dimensionality}. Cannot convert '{payload.units}' to siemens."
                        )
                    ) from e

                # == Contract Check 2: Numerical Shape ==
                # The contract for a simple component is a SCALAR admittance. We enforce
                # this by checking that the underlying magnitude has zero dimensions.
                if np.ndim(y_mag) != 0:
                    raise ComponentError(
                        component_fqn=sim_comp.fqn,
                        details=(
                            "Component provided a scalar Quantity payload for DC admittance, "
                            f"but its magnitude was not a scalar (ndim=0). Got ndim={np.ndim(y_mag)}."
                        )
                    )

                # --- Stamping Logic (Now Guaranteed Safe) ---
                scalar_y_mag = complex(y_mag)
                nets = [net for net in port_map.values() if net in supernode_map]
                unique_srep_indices = {s_to_mna_idx_map[supernode_map[n]] for n in nets}

                if len(unique_srep_indices) == 2:
                    idx1, idx2 = tuple(unique_srep_indices)
                    y_dc_super_full[idx1, idx1] += scalar_y_mag
                    y_dc_super_full[idx2, idx2] += scalar_y_mag
                    y_dc_super_full[idx1, idx2] -= scalar_y_mag
                    y_dc_super_full[idx2, idx1] -= scalar_y_mag
                elif len(unique_srep_indices) == 1:
                    idx1 = unique_srep_indices.pop()
                    y_dc_super_full[idx1, idx1] += scalar_y_mag

            else:  # --- Final Catch-All for Invalid Payload Types ---
                # If the payload is not a tuple and not a Quantity, it's a gross
                # violation of the contract. This is the final safeguard.
                raise ComponentError(
                    component_fqn=sim_comp.fqn,
                    details=(
                        f"Returned a DC ADMITTANCE behavior but its payload was of an invalid type. "
                        f"Expected a scalar pint.Quantity or a (Quantity, List[str]) tuple, "
                        f"but got '{type(payload).__name__}'."
                    )
                )

        return y_dc_super_full

    def _identify_dc_ports(
        self, external_ports: KeysView[str], supernode_map: Dict[str, str], gnd_s_rep: Optional[str]
    ) -> List[str]:
        """Identifies which supernodes correspond to external AC ports. (STATELESS)"""
        supernode_rep_to_ac_ports: Dict[str, List[str]] = {}
        for port_name in external_ports:
             srep = supernode_map.get(port_name, port_name)
             if srep is not None and srep != gnd_s_rep:
                 supernode_rep_to_ac_ports.setdefault(srep, []).append(port_name)
        # Sort by supernode name, then take the alphabetically first AC port name as the canonical representative
        return [sorted(ports)[0] for _, ports in sorted(supernode_rep_to_ac_ports.items())]

    def _calculate_dc_y_parameters(
        self,
        y_dc_super_full: np.ndarray,
        dc_port_names: List[str],
        external_ports: KeysView[str],
        supernode_map: Dict[str, str],
        s_to_mna_idx_map: Dict[str, int]
    ) -> Optional[Quantity]:
        """Calculates the DC N-port Y-matrix using Schur complement reduction. (STATELESS)"""
        num_dc_ports = len(dc_port_names)
        if num_dc_ports == 0:
            return None
        
        port_indices = [s_to_mna_idx_map[supernode_map[n]] for n in dc_port_names]
        internal_indices = sorted(list(set(range(1, y_dc_super_full.shape[0])) - set(port_indices)))
        
        Y_PP = y_dc_super_full[np.ix_(port_indices, port_indices)]
        if not internal_indices:
            return Quantity(Y_PP, self.ureg.siemens)
        
        Y_PI = y_dc_super_full[np.ix_(port_indices, internal_indices)]
        Y_IP = y_dc_super_full[np.ix_(internal_indices, port_indices)]
        Y_II = y_dc_super_full[np.ix_(internal_indices, internal_indices)]
        try:
            lu, piv = lu_factor(Y_II)
            X = lu_solve((lu, piv), Y_IP)
            y_ports_dc_mag = Y_PP - (Y_PI @ X)
            return Quantity(y_ports_dc_mag, self.ureg.siemens)
        except np.linalg.LinAlgError as e:
            raise DCAnalysisError(
                hierarchical_context=self.circuit.hierarchical_id,
                details=f"Failed to compute DC Schur complement (matrix may be singular at DC): {e}"
            ) from e

    def _build_dc_port_mapping(
        self,
        external_ports: KeysView[str],
        dc_port_names: List[str],
        supernode_map: Dict[str, str],
        gnd_s_rep: Optional[str]
    ) -> Dict[str, Optional[int]]:
        """Creates a map from original AC port names to their corresponding DC port indices. (STATELESS)"""
        dc_port_mapping: Dict[str, Optional[int]] = {}
        dc_port_to_idx = {name: i for i, name in enumerate(dc_port_names)}
        
        # This map allows us to find which canonical DC port name a supernode corresponds to.
        srep_to_port_name = {supernode_map.get(name): name for name in dc_port_names}
        
        for ac_port in external_ports:
            srep = supernode_map.get(ac_port, ac_port)
            if srep is None or srep == gnd_s_rep:
                dc_port_mapping[ac_port] = None
            else:
                # Find the canonical DC port name for this AC port's supernode
                canonical_dc_port_name = srep_to_port_name.get(srep)
                # Then look up the index of that canonical name
                dc_port_mapping[ac_port] = dc_port_to_idx.get(canonical_dc_port_name)
        return dc_port_mapping