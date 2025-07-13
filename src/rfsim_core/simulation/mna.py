# src/rfsim_core/simulation/mna.py

import logging
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import scipy.sparse as sp

from ..components.base import ComponentBase, StampInfo
from ..components.exceptions import ComponentError
# This import is the cornerstone of the purified MNA system.
from ..components.capabilities import IConnectivityProvider
from ..data_structures import Circuit, Net
from ..parameters import ParameterManager
from ..units import ureg, Quantity
from .exceptions import MnaInputError


logger = logging.getLogger(__name__)


class MnaAssembler:
    """
    Constructs the Modified Nodal Analysis (MNA) system for a given circuit.

    **Architecturally Purified Implementation:**
    This class is completely agnostic to component types. It contains no special-cased
    logic (e.g., `isinstance(comp, SubcircuitInstance)`) and interacts with ALL
    components solely through their formal, advertised capabilities (`IConnectivityProvider`)
    and API contracts (`get_port_net_mapping`). This purification fulfills the
    "Separation of Concerns" and "Explicit Contracts" mandates, making the MNA system
    robust, extensible, and verifiable.

    It is responsible for:
    1.  Determining the set of active nodes and components to include in the AC analysis.
    2.  Assigning a unique integer index to each node.
    3.  Pre-computing the MNA matrix's sparsity pattern by querying each component's
        `IConnectivityProvider` capability.
    4.  Assembling the final sparse MNA matrix for a specific frequency by collecting
        and stamping contributions from all components.
    """
    def __init__(self, circuit: Circuit, active_nets_override: Optional[Set[str]] = None):
        """
        Initializes the MnaAssembler.

        Args:
            circuit: The simulation-ready Circuit object.
            active_nets_override: An optional set of net names. If provided, the MNA
                                  system will only be built using these nets and the
                                  components exclusively connected to them. If None,
                                  all nets in the circuit are used.
        """
        if not isinstance(circuit, Circuit) or not circuit.raw_ir_root:
            raise MnaInputError(hierarchical_context=circuit.hierarchical_id, details="Circuit object must be fully built and simulation-ready.")

        self.circuit_orig: Circuit = circuit
        self.ureg = ureg
        self.parameter_manager: ParameterManager = circuit.parameter_manager

        self.active_nets_override: Optional[Set[str]] = active_nets_override

        self._effective_nets: Dict[str, Net]
        self._effective_sim_components_map: Dict[str, ComponentBase]
        self._effective_external_ports: Dict[str, Net]
        self._effective_ground_net_name: Optional[str] = None

        self._filter_circuit_elements()

        self.node_map: Dict[str, int] = {}
        self.node_count: int = 0
        self.port_names: List[str] = []
        self.port_indices: List[int] = []

        self._external_node_indices_reduced: List[int] = []
        self._internal_node_indices_reduced: List[int] = []

        self._cached_rows: Optional[np.ndarray] = None
        self._cached_cols: Optional[np.ndarray] = None
        self._sparsity_nnz: int = 0
        self._shape_full: Tuple[int, int] = (0, 0)

        self._assign_node_indices()
        self._identify_reduced_indices()
        self._compute_sparsity_pattern()

        logger.info(
            f"MNA Assembler initialized for circuit '{self.circuit_orig.name}'. "
            f"Effective nodes: {self.node_count}."
        )

    @property
    def effective_sim_components(self) -> Dict[str, ComponentBase]:
        """Returns the dictionary of components included in this MNA system."""
        return self._effective_sim_components_map

    def _filter_circuit_elements(self):
        """
        Filters the circuit to only include elements connected to the active nets,
        if an override is provided.
        """
        if self.active_nets_override is None:
            self._effective_nets = self.circuit_orig.nets
            self._effective_sim_components_map = self.circuit_orig.sim_components
            self._effective_external_ports = self.circuit_orig.external_ports
            self._effective_ground_net_name = self.circuit_orig.ground_net_name
            return

        active_nets_set = self.active_nets_override
        self._effective_nets = {n: net for n, net in self.circuit_orig.nets.items() if n in active_nets_set}
        self._effective_ground_net_name = self.circuit_orig.ground_net_name if self.circuit_orig.ground_net_name in active_nets_set else None

        self._effective_sim_components_map = {}
        for comp_id, sim_comp in self.circuit_orig.sim_components.items():
            # This logic correctly uses the formal get_port_net_mapping contract
            port_map = sim_comp.get_port_net_mapping()
            connected_nets = set(port_map.values())

            if connected_nets.issubset(active_nets_set):
                self._effective_sim_components_map[sim_comp.fqn] = sim_comp

        self._effective_external_ports = {n: net for n, net in self.circuit_orig.external_ports.items() if n in active_nets_set}

    def _assign_node_indices(self):
        """Assigns a unique integer index to each effective node."""
        if not self._effective_nets:
            self.node_count = 0
            self.port_names, self.port_indices = [], []
            self._shape_full = (0, 0)
            return

        self.node_map = {}
        idx = 0
        if self._effective_ground_net_name:
            self.node_map[self._effective_ground_net_name] = idx
            idx += 1

        for net_name in sorted(self._effective_nets.keys()):
            if net_name not in self.node_map:
                self.node_map[net_name] = idx
                idx += 1

        self.node_count = idx
        self._shape_full = (idx, idx)
        self.port_names = sorted(self._effective_external_ports.keys())
        self.port_indices = [self.node_map[name] for name in self.port_names]

    def _identify_reduced_indices(self):
        """
        Calculates the MNA matrix indices for external and internal nodes in the
        reduced system (where ground is removed).
        """
        if self.node_count <= 1:
            self._external_node_indices_reduced, self._internal_node_indices_reduced = [], []
            return

        ground_is_zero = self._effective_ground_net_name is not None
        all_indices_set = set(range(self.node_count))
        port_indices_set = set(self.port_indices)

        if ground_is_zero:
            all_indices_set.remove(0)
            port_indices_set.discard(0)

        reduced_map = {full_idx: (full_idx - 1 if ground_is_zero else full_idx) for full_idx in all_indices_set}
        self._external_node_indices_reduced = sorted([reduced_map[i] for i in port_indices_set])
        internal_indices_set = all_indices_set - port_indices_set
        self._internal_node_indices_reduced = sorted([reduced_map[i] for i in internal_indices_set])

    @property
    def external_node_indices_reduced(self) -> List[int]:
        """Returns the list of reduced indices corresponding to external ports."""
        return self._external_node_indices_reduced

    @property
    def internal_node_indices_reduced(self) -> List[int]:
        """Returns the list of reduced indices corresponding to internal nodes."""
        return self._internal_node_indices_reduced

    def _add_to_coo_pattern(self, rows: list, cols: list, r: int, c: int):
        """Helper to append a coordinate to the sparsity pattern lists."""
        rows.append(r)
        cols.append(c)

    def _compute_sparsity_pattern(self):
        """
        Computes and caches the sparsity pattern (non-zero locations) of the full MNA matrix.
        Enforces that N-port components (>2 ports) must not use the default, non-connecting
        IConnectivityProvider from ComponentBase, raising ComponentError if violated.
        Iterates over all components, queries connectivity, and builds the COO pattern.
        """
        rows, cols = [], []
        if self.node_count == 0:
            # If there are no nodes in the effective circuit, the pattern is empty.
            self._cached_rows, self._cached_cols, self._sparsity_nnz = np.array([], dtype=int), np.array([], dtype=int), 0
            return

        # Iterate through every component that will be included in the MNA system.
        for sim_comp in self._effective_sim_components_map.values():
            # Step 1: Query for the universal capability.
            connectivity_provider = sim_comp.get_capability(IConnectivityProvider)
            if not connectivity_provider:
                # If the component does not provide this capability (e.g., a conceptual
                # 1-port source), it cannot contribute to the connectivity graph.
                # This is a valid state, not an error.
                logger.debug(f"Component '{sim_comp.fqn}' provides no connectivity. Skipping for sparsity.")
                continue

            # Step 2: Validate that N-port components (>2 ports) do not use the default connectivity provider.
            num_ports = len(type(sim_comp).declare_ports())

            # Check if the connectivity provider is the default from ComponentBase.
            if num_ports > 2 and isinstance(connectivity_provider, ComponentBase.ConnectivityProvider):
                # This is a FATAL configuration error. Allowing the simulation to proceed
                # would mean accepting an ambiguous topology, violating the "Correctness by
                # Construction" principle. We raise a specific, actionable error.
                raise ComponentError(
                    component_fqn=sim_comp.fqn,
                    details=(
                        f"Component has {num_ports} ports but uses the default connectivity provider, which "
                        "is only valid for 2-port devices and assumes no internal connections for N-port devices. "
                        "This creates an ambiguous and likely incorrect circuit topology. To ensure correctness by "
                        "construction, you MUST implement a custom `IConnectivityProvider` capability for this "
                        "component type that explicitly defines its internal port-to-port connectivity."
                    )
                )
            # --- END OF HARDENING ---

            # Step 3: Retrieve port mapping and connectivity info (safe after validation).
            port_map = sim_comp.get_port_net_mapping()
            component_connectivity = connectivity_provider.get_connectivity(sim_comp)

            # Step 4: Add diagonal (self-admittance) terms for all connected ports.
            all_connected_ports = {port for pair in component_connectivity for port in pair}
            for port_name in all_connected_ports:
                net_name = port_map.get(port_name)
                if net_name in self.node_map:
                    net_idx = self.node_map[net_name]
                    # Add (i, i) to the pattern for self-admittance.
                    self._add_to_coo_pattern(rows, cols, net_idx, net_idx)

            # Step 5: Add off-diagonal (mutual-admittance) terms.
            for p1, p2 in component_connectivity:
                net1_name, net2_name = port_map.get(p1), port_map.get(p2)
                if net1_name in self.node_map and net2_name in self.node_map:
                    n1_idx, n2_idx = self.node_map[net1_name], self.node_map[net2_name]
                    if n1_idx != n2_idx:
                        # The component connects two different nodes.
                        self._add_to_coo_pattern(rows, cols, n1_idx, n2_idx)
                        self._add_to_coo_pattern(rows, cols, n2_idx, n1_idx)

        # Step 6: Cache the unique, sorted sparsity pattern.
        temp_coo = sp.coo_matrix((np.ones_like(rows), (rows, cols)), shape=self._shape_full)
        final_coo = temp_coo.tocsc().tocoo()

        # Cache the results for use in the `assemble` method.
        self._cached_rows = final_coo.row
        self._cached_cols = final_coo.col
        self._sparsity_nnz = final_coo.nnz

    def assemble(self, current_sweep_idx: int, all_stamps_vectorized: Dict[str, StampInfo]) -> sp.csc_matrix:
        """
        Assembles the full MNA matrix for a single frequency point.

        This method is architecturally pure and retrieves a component's port-to-net
        mapping via the formal `get_port_net_mapping()` API contract. It does not
        violate encapsulation by accessing any internal component data.

        Args:
            current_sweep_idx: The index of the current frequency in the sweep array.
            all_stamps_vectorized: A dictionary mapping component FQNs to their single,
                                   pre-computed, vectorized MNA stamp contribution.

        Returns:
            A SciPy CSC (Compressed Sparse Column) matrix representing the full MNA system.
        """
        if self._cached_rows is None or self._cached_cols is None:
            raise RuntimeError("Sparsity pattern has not been computed. This is a framework logic error.")
        if self.node_count == 0:
            return sp.csc_matrix((0, 0), dtype=np.complex128)

        mna_data = np.zeros(self._sparsity_nnz, dtype=np.complex128)
        rc_to_data_idx = {(r, c): i for i, (r, c) in enumerate(zip(self._cached_rows, self._cached_cols))}

        def add_value(r, c, value, comp_fqn):
            if (key := rc_to_data_idx.get((r, c))) is not None:
                mna_data[key] += value
            elif abs(value) > 1e-15:
                # This check protects against components trying to stamp outside their declared connectivity.
                raise MnaInputError(hierarchical_context=comp_fqn, details=f"Component tried to stamp ({value:.3e}) at ({r},{c}), which is outside its declared sparsity pattern.")

        for comp_fqn, sim_comp in self._effective_sim_components_map.items():
            if comp_fqn not in all_stamps_vectorized:
                continue

            try:
                # Per the purified contract, each component provides a single stamp tuple.
                # The redundant inner loop has been removed.
                stamp_matrix_qty, port_ids_from_stamp = all_stamps_vectorized[comp_fqn]

                # Slice the correct per-frequency matrix from the pre-computed vectorized stamp
                stamp_matrix_scalar = stamp_matrix_qty.magnitude[current_sweep_idx]

                # This uses the formal, explicit, and type-safe API contract.
                port_map = sim_comp.get_port_net_mapping()

                global_indices = [self.node_map[port_map[pid]] for pid in port_ids_from_stamp]

                for i, r_idx in enumerate(global_indices):
                    for j, c_idx in enumerate(global_indices):
                        add_value(r_idx, c_idx, stamp_matrix_scalar[i, j], comp_fqn)

            except (IndexError, KeyError) as e:
                raise ComponentError(
                    component_fqn=comp_fqn,
                    details=(
                        f"Stamping failed at sweep index {current_sweep_idx}. This is likely caused by a "
                        f"mismatch between the component's returned port IDs and the netlist's port mapping. "
                        f"Original error: {type(e).__name__}: {e}"
                    ),
                    frequency=None
                ) from e
        
        Yn_full_coo = sp.coo_matrix((mna_data, (self._cached_rows, self._cached_cols)), shape=self._shape_full)
        return Yn_full_coo.tocsc()