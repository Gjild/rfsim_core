# src/rfsim_core/analysis/tools.py

"""
Provides high-level, cache-aware analysis services for synthesized Circuit objects.

**Architectural Refactoring (Phase 10):**

This module has been fundamentally refactored into a collection of stateless,
cache-aware services. The key architectural changes are:

1.  **Eradication of Hidden State:** All `static` (class-level) caches have been
    **DELETED**. This was a critical flaw that made simulations non-deterministic.
    The system is now free of hidden global state.

2.  **Dependency Injection:** Both `DCAnalyzer` and `TopologyAnalyzer` now accept a
    `SimulationCache` object in their constructor (`__init__`). This makes their
    dependency on the caching service explicit and testable.

3.  **Unified Caching Strategy:** Both services now use the same "get-from-cache-or-compute"
    pattern, leveraging the centralized `SimulationCache` and the `cache.keys` module.
    This ensures consistent, correct, and efficient caching behavior across the system.

4.  **Explicit Contracts:** The `analyze` methods now return formal, immutable,
    type-safe result objects (`DCAnalysisResults`, `TopologyAnalysisResults`) instead
    of raw dictionaries. This fulfills the "Explicit Contracts" mandate.
"""

import logging
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from scipy.linalg import lu_factor, lu_solve

from ..components.base import ComponentBase, ComponentError, DCBehaviorType
from ..components.capabilities import IDcContributor
from ..components.subcircuit import SubcircuitInstance
from ..data_structures import Circuit
from ..parameters import ParameterError, ParameterManager
from ..parser.raw_data import ParsedLeafComponentData
from ..units import Quantity, ureg

# --- Phase 10, Task 2: NEW IMPORTS ---
# These imports are the foundation of the new cache-aware service architecture.
from ..cache.keys import create_dc_analysis_key, create_topology_key
from ..cache.service import SimulationCache
from .exceptions import DCAnalysisError, TopologyAnalysisError
from .results import DCAnalysisResults, TopologyAnalysisResults

logger = logging.getLogger(__name__)


class TopologyAnalyzer:
    """
    Performs and caches topological analysis for a given circuit configuration.
    This is a stateless service that operates on a circuit and a cache.
    """
    # --- DELETED: The static, hidden cache is completely removed. ---

    def __init__(self, circuit: Circuit, cache: SimulationCache):
        """
        Initializes the TopologyAnalyzer service.

        Args:
            circuit: The simulation-ready Circuit object to be analyzed.
            cache: The centralized SimulationCache instance for this run.
                   This is an example of Dependency Injection.
        """
        if not isinstance(circuit, Circuit) or not circuit.parameter_manager:
            raise TypeError("TopologyAnalyzer requires a valid, simulation-ready Circuit object.")
        if not isinstance(cache, SimulationCache):
            raise TypeError("TopologyAnalyzer requires a valid SimulationCache instance.")

        self.circuit: Circuit = circuit
        self.parameter_manager: ParameterManager = circuit.parameter_manager
        # The instance-level cache holds the result for *this specific instance* only,
        # preventing re-computation if analyze() is called multiple times on the same object.
        self._analysis_results: Optional[TopologyAnalysisResults] = None
        # Store the injected cache service.
        self.cache: SimulationCache = cache
        logger.debug(f"TopologyAnalyzer service initialized for circuit '{circuit.hierarchical_id}'.")

    # --- DELETED: The private _get_cache_key method is removed. ---

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
            # Type-check the cached result to ensure cache integrity.
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
            ac_graph = self._build_ac_graph(open_comps)
            active_nets = self._compute_active_nets(ac_graph)
            port_connectivity = self._compute_external_port_connectivity(ac_graph)

            # 4. Package the results into the formal, immutable dataclass.
            results = TopologyAnalysisResults(
                structurally_open_components=open_comps,
                ac_graph=ac_graph,
                active_nets=active_nets,
                external_port_connectivity=port_connectivity,
            )

            # 5. Store the new, formal result object in the persistent 'process' cache.
            self.cache.put(key=cache_key, value=results, scope='process')
            self._analysis_results = results
            return results
        except Exception as e:
            # Wrap any unexpected error in a diagnosable exception.
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
            if isinstance(sim_comp, SubcircuitInstance): continue

            resolved_constant_params: Dict[str, Quantity] = {}
            for base_param_name, fqn in zip(type(sim_comp).declare_parameters(), sim_comp.parameter_fqns):
                if self.parameter_manager.is_constant(fqn):
                    try:
                        resolved_constant_params[base_param_name] = self.parameter_manager.get_constant_value(fqn)
                    except ParameterError as e:
                        logger.warning(f"Could not resolve constant '{fqn}' for structural open check: {e}")

            if sim_comp.is_structurally_open(resolved_constant_params):
                open_comp_ids.add(comp_id)
        return open_comp_ids

    def _build_ac_graph(self, structurally_open_components: Set[str]) -> nx.Graph:
        """Builds the AC connectivity graph for this circuit level, recursively analyzing subcircuits."""
        ac_graph = nx.Graph()
        ac_graph.add_nodes_from(self.circuit.nets.keys())

        for sim_comp in self.circuit.sim_components.values():
            if sim_comp.instance_id in structurally_open_components:
                continue

            if isinstance(sim_comp, SubcircuitInstance):
                # CRITICAL RECURSIVE FIX: The cache is passed down to the recursive analyzer.
                sub_ta = TopologyAnalyzer(sim_comp.sub_circuit_object, self.cache)
                sub_results = sub_ta.analyze()
                port_connectivity = sub_results.external_port_connectivity

                port_map = sim_comp.raw_ir_data.raw_port_mapping
                for sub_port1, sub_port2 in port_connectivity:
                    net1, net2 = port_map.get(sub_port1), port_map.get(sub_port2)
                    if net1 and net2 and net1 in ac_graph and net2 in ac_graph:
                        ac_graph.add_edge(net1, net2)
            elif isinstance(sim_comp, ComponentBase):
                raw_comp_data = sim_comp.raw_ir_data
                if not isinstance(raw_comp_data, ParsedLeafComponentData): continue

                nets = set(raw_comp_data.raw_ports_dict.values())
                for net1, net2 in combinations(sorted(list(nets)), 2):
                    if net1 in ac_graph and net2 in ac_graph:
                        ac_graph.add_edge(net1, net2)

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
    This is a stateless service that operates on a circuit and a cache.
    """
    def __init__(self, circuit: Circuit, cache: SimulationCache):
        """
        Initializes the DCAnalyzer service.

        Args:
            circuit: The simulation-ready Circuit object to be analyzed.
            cache: The centralized SimulationCache instance for this run.
        """
        if not isinstance(circuit, Circuit) or not circuit.parameter_manager:
            raise TypeError("DCAnalyzer requires a valid, simulation-ready Circuit object.")
        if not isinstance(cache, SimulationCache):
            raise TypeError("DCAnalyzer requires a valid SimulationCache instance.")

        self.circuit: Circuit = circuit
        self.parameter_manager: ParameterManager = circuit.parameter_manager
        self.cache: SimulationCache = cache  # Dependency Injection
        self.ureg = ureg

        # Internal state is now transient, reset for each analysis run.
        self.ground_net_name: str = circuit.ground_net_name
        self._supernode_map: Dict[str, str] = {}
        self._ground_supernode_representative_name: Optional[str] = None
        self._supernode_graph: Optional[nx.Graph] = None
        self._supernode_to_mna_index_map: Dict[str, int] = {}
        self._dc_port_names_ordered: List[str] = []

        logger.debug(f"DCAnalyzer service initialized for circuit '{circuit.hierarchical_id}'.")

    def analyze(self) -> DCAnalysisResults:
        """
        Executes the full DC analysis pipeline, leveraging the injected cache.

        Returns:
            The formal, immutable DCAnalysisResults object for this circuit.
        """
        # 1. Generate the definitive cache key using the centralized key factory.
        cache_key = create_dc_analysis_key(self.circuit)

        # 2. Query the persistent 'process' scope of the centralized cache.
        cached_results = self.cache.get(key=cache_key, scope='process')
        if cached_results:
            if not isinstance(cached_results, DCAnalysisResults):
                logger.warning(
                    f"Cache integrity issue: expected DCAnalysisResults but got "
                    f"{type(cached_results)}. Re-computing."
                )
            else:
                return cached_results

        # --- On Cache Miss: Perform the Full Analysis ---
        logger.info(f"Cache MISS. Starting rigorous DC analysis for circuit '{self.circuit.hierarchical_id}'...")

        # The entire, existing DC analysis logic is executed here, verbatim, only on a cache miss.
        # This logic populates the transient internal state (e.g., self._supernode_map).
        try:
            dc_behaviors = self._resolve_and_collect_dc_behaviors()
            self._build_dc_supernode_graph(dc_behaviors)
            self._identify_supernodes()
            self._assign_dc_mna_indices()
            y_dc_super_full = self._build_dc_mna_matrix(dc_behaviors)
            self._identify_dc_ports()
            y_ports_dc_qty = self._calculate_dc_y_parameters(y_dc_super_full)
            dc_port_mapping = self._build_dc_port_mapping()
        except Exception as e:
            # Wrap any unexpected error in a diagnosable exception.
            if isinstance(e, DCAnalysisError):  # Re-raise if already diagnosed
                raise
            raise DCAnalysisError(
                hierarchical_context=self.circuit.hierarchical_id,
                details=f"An unexpected error occurred during DC analysis: {e}"
            ) from e

        logger.info(f"DC analysis for '{self.circuit.hierarchical_id}' complete.")

        # 3. Create the formal, self-contained, and immutable result object.
        results = DCAnalysisResults(
            y_ports_dc=y_ports_dc_qty,
            dc_port_names_ordered=self._dc_port_names_ordered,
            dc_port_mapping=dc_port_mapping,
            dc_supernode_mapping=self._supernode_map.copy(),
            ground_supernode_name=self._ground_supernode_representative_name
        )

        # 4. Store the new, formal result object in the persistent 'process' cache.
        self.cache.put(key=cache_key, value=results, scope='process')
        return results

    # --- DELETED: The public helper methods are now obsolete. ---
    # `get_supernode_representative_name` and `is_ground_supernode_by_representative`
    # have been removed as their functionality is now contained within the DCAnalysisResults object.

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
            except (ComponentError, KeyError) as e:
                details = f"Error getting DC behavior for component '{sim_comp.fqn}': {e}"
                logger.error(details, exc_info=True)
                raise DCAnalysisError(hierarchical_context=self.circuit.hierarchical_id, details=details) from e
        return dc_behaviors

    def _build_dc_supernode_graph(self, dc_behaviors: Dict[str, Tuple[DCBehaviorType, Optional[Quantity]]]):
        """Constructs a graph where edges represent ideal DC shorts between nets."""
        logger.debug(f"[{self.circuit.hierarchical_id}] Building DC supernode connectivity graph...")
        self._supernode_graph = nx.Graph()
        self._supernode_graph.add_nodes_from(self.circuit.nets.keys())

        for comp_id, (behavior_type, _) in dc_behaviors.items():
            if behavior_type != DCBehaviorType.SHORT_CIRCUIT:
                continue

            sim_comp = self.circuit.sim_components[comp_id]
            raw_comp_data = sim_comp.raw_ir_data
            if not isinstance(raw_comp_data, ParsedLeafComponentData):
                logger.error(
                    f"Internal contract violation: Component '{sim_comp.fqn}' reported as a DC "
                    "short but is not a leaf component. This is an impossible state and will be ignored."
                )
                continue
            
            connected_nets = [net_name for net_name in raw_comp_data.raw_ports_dict.values() if net_name in self.circuit.nets]
            if len(connected_nets) >= 2:
                for net1, net2 in combinations(sorted(list(set(connected_nets))), 2):
                    self._supernode_graph.add_edge(net1, net2, type='dc_short', component_id=comp_id)

    def _identify_supernodes(self):
        """Finds connected components in the shorting graph to define supernodes."""
        if self._supernode_graph is None: raise DCAnalysisError(hierarchical_context=self.circuit.hierarchical_id, details="Supernode graph was not built.")
        self._supernode_map = {net: net for net in self.circuit.nets}
        for component_nets in nx.connected_components(self._supernode_graph):
            is_gnd_component = self.ground_net_name in component_nets
            representative_name = self.ground_net_name if is_gnd_component else sorted(list(component_nets))[0]
            for net_name in component_nets:
                self._supernode_map[net_name] = representative_name
        self._ground_supernode_representative_name = self._supernode_map.get(self.ground_net_name)

    def _assign_dc_mna_indices(self):
        """Assigns a unique integer index to each supernode for MNA matrix assembly."""
        if not self._supernode_map: raise DCAnalysisError(hierarchical_context=self.circuit.hierarchical_id, details="Supernode map not populated.")
        if self._ground_supernode_representative_name is None: raise DCAnalysisError(hierarchical_context=self.circuit.hierarchical_id, details="Ground supernode not identified.")
        
        representatives = sorted(list(set(self._supernode_map.values())))
        self._supernode_to_mna_index_map = {self._ground_supernode_representative_name: 0}
        idx_counter = 1
        for rep in representatives:
            if rep != self._ground_supernode_representative_name:
                self._supernode_to_mna_index_map[rep] = idx_counter
                idx_counter += 1

    def _build_dc_mna_matrix(self, dc_behaviors: Dict[str, Tuple[DCBehaviorType, Optional[Quantity]]]) -> np.ndarray:
        """Assembles the full supernode MNA admittance matrix from component contributions."""
        num_dc_nodes = len(self._supernode_to_mna_index_map)
        if num_dc_nodes == 0: return np.array([], dtype=complex).reshape(0, 0)
        
        y_dc_super_full = np.zeros((num_dc_nodes, num_dc_nodes), dtype=complex)
        for comp_id, (behavior_type, admittance_qty) in dc_behaviors.items():
            if behavior_type != DCBehaviorType.ADMITTANCE or admittance_qty is None:
                continue

            sim_comp = self.circuit.sim_components[comp_id]
            if isinstance(sim_comp, SubcircuitInstance):
                y_sub_mag = admittance_qty.to(self.ureg.siemens).magnitude
                port_map = sim_comp.raw_ir_data.raw_port_mapping
                parent_indices = [
                    self._supernode_to_mna_index_map[self._supernode_map[port_map[p]]]
                    for p in sim_comp.sub_circuit_external_port_names_ordered
                ]
                for i, r_idx in enumerate(parent_indices):
                    for j, c_idx in enumerate(parent_indices):
                        y_dc_super_full[r_idx, c_idx] += y_sub_mag[i, j]
            else:
                y_mag = complex(admittance_qty.to(self.ureg.siemens).magnitude)
                raw_comp = sim_comp.raw_ir_data
                nets = [net for net in raw_comp.raw_ports_dict.values() if net in self.circuit.nets]
                unique_srep_indices = {self._supernode_to_mna_index_map[self._supernode_map[n]] for n in nets}
                if len(unique_srep_indices) == 2:
                    idx1, idx2 = tuple(unique_srep_indices)
                    y_dc_super_full[idx1, idx1] += y_mag; y_dc_super_full[idx2, idx2] += y_mag
                    y_dc_super_full[idx1, idx2] -= y_mag; y_dc_super_full[idx2, idx1] -= y_mag
                elif len(unique_srep_indices) == 1:
                    idx1 = unique_srep_indices.pop()
                    y_dc_super_full[idx1, idx1] += y_mag
        return y_dc_super_full

    def _identify_dc_ports(self):
        """Identifies which supernodes correspond to external AC ports."""
        supernode_rep_to_ac_ports: Dict[str, List[str]] = {}
        for port_name in self.circuit.external_ports:
             srep = self._supernode_map.get(port_name, port_name)
             if srep is not None and srep != self._ground_supernode_representative_name:
                 supernode_rep_to_ac_ports.setdefault(srep, []).append(port_name)
        self._dc_port_names_ordered = [sorted(ports)[0] for _, ports in sorted(supernode_rep_to_ac_ports.items())]

    def _calculate_dc_y_parameters(self, y_dc_super_full: np.ndarray) -> Optional[Quantity]:
        """Calculates the DC N-port Y-matrix using Schur complement reduction."""
        num_dc_ports = len(self._dc_port_names_ordered)
        if num_dc_ports == 0: return None
        
        port_indices = [self._supernode_to_mna_index_map[self._supernode_map[n]] for n in self._dc_port_names_ordered]
        internal_indices = sorted(list(set(range(1, y_dc_super_full.shape[0])) - set(port_indices)))
        
        Y_PP = y_dc_super_full[np.ix_(port_indices, port_indices)]
        if not internal_indices: return Quantity(Y_PP, self.ureg.siemens)
        
        Y_PI = y_dc_super_full[np.ix_(port_indices, internal_indices)]
        Y_IP = y_dc_super_full[np.ix_(internal_indices, port_indices)]
        Y_II = y_dc_super_full[np.ix_(internal_indices, internal_indices)]
        try:
            X = lu_solve(lu_factor(Y_II), Y_IP)
            y_ports_dc_mag = Y_PP - (Y_PI @ X)
            return Quantity(y_ports_dc_mag, self.ureg.siemens)
        except np.linalg.LinAlgError as e:
            raise DCAnalysisError(hierarchical_context=self.circuit.hierarchical_id, details=f"Failed to compute DC Schur complement: {e}") from e

    def _build_dc_port_mapping(self) -> Dict[str, Optional[int]]:
        """Creates a map from original AC port names to their corresponding DC port indices."""
        dc_port_mapping: Dict[str, Optional[int]] = {}
        dc_port_to_idx = {name: i for i, name in enumerate(self._dc_port_names_ordered)}
        srep_to_port_name = {self._supernode_map.get(name): name for name in self._dc_port_names_ordered}
        for ac_port in self.circuit.external_ports:
            srep = self._supernode_map.get(ac_port, ac_port)
            if srep is None or srep == self._ground_supernode_representative_name:
                dc_port_mapping[ac_port] = None
            else:
                port_name = srep_to_port_name.get(srep)
                dc_port_mapping[ac_port] = dc_port_to_idx.get(port_name)
        return dc_port_mapping