# --- Modify: src/rfsim_core/simulation/mna.py ---
import logging
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple, Optional, Set

from ..data_structures import Circuit, Net
from ..data_structures import Port as PortDataStructure # For type hint in helper
from ..data_structures import Component as ComponentData # Renamed import
from ..components.base import ComponentBase, ComponentError
from ..units import ureg, Quantity, pint
# LARGE_ADMITTANCE_SIEMENS not directly used here, but in components.
from ..parameters import ParameterManager, ParameterScopeError, ParameterError

logger = logging.getLogger(__name__)

class MnaInputError(ValueError):
    """Error related to inputs for MNA assembly."""
    pass

# Helper function for robust port lookup
def _get_port_obj_from_comp_data(
    comp_ports_dict: Dict[str | int, PortDataStructure],
    declared_port_id_from_type: str | int
) -> Optional[PortDataStructure]:
    """
    Robustly retrieves a PortDataStructure object from a component's port dictionary.
    Tries the declared_port_id_from_type directly, then attempts type conversion
    (int to str, or str to int) if the initial lookup fails.
    """
    port_obj = comp_ports_dict.get(declared_port_id_from_type)

    if port_obj is None:
        if isinstance(declared_port_id_from_type, int):
            # If declared is int, try lookup with string version
            port_obj = comp_ports_dict.get(str(declared_port_id_from_type))
        elif isinstance(declared_port_id_from_type, str):
            # If declared is string, try lookup with int version (if string is int-like)
            try:
                port_obj = comp_ports_dict.get(int(declared_port_id_from_type))
            except ValueError:
                # declared_port_id_from_type is a string, but not a simple integer string
                pass # port_obj remains None
    return port_obj

class MnaAssembler:
    """
    Assembles the MNA matrix (Yn) for a circuit.
    Can operate on the full circuit or a subset of "active nets".
    Initialization is frequency-independent. Assembly is done per frequency.
    """
    def __init__(self, circuit: Circuit, active_nets_override: Optional[Set[str]] = None):
        """
        Initializes the assembler, calculates node maps, port info, sparsity pattern,
        and identifies external/internal node indices for the (potentially reduced) system.

        Args:
            circuit: The simulation-ready Circuit object.
            active_nets_override: If provided, a set of net names to consider "active".
                                  MNA system will be built only for these nets and components
                                  fully connected to them. If None, full circuit is used.
        Raises:
            MnaInputError: If circuit is invalid or has configuration issues.
        """
        if not isinstance(circuit, Circuit) or not hasattr(circuit, 'sim_components'):
            raise MnaInputError("Circuit object must be simulation-ready (output of CircuitBuilder).")
        
        self.circuit_orig: Circuit = circuit 
        self.ureg = ureg
        self.parameter_manager: ParameterManager = circuit.parameter_manager
        if not isinstance(self.parameter_manager, ParameterManager):
            raise MnaInputError("Circuit's ParameterManager is not initialized or invalid.")

        self.active_nets_override: Optional[Set[str]] = active_nets_override
        
        self._effective_nets: Dict[str, Net] 
        self._effective_sim_components: Dict[str, ComponentBase] 
        self._effective_raw_components: Dict[str, ComponentData] 
        self._effective_external_ports: Dict[str, Net] 
        self._effective_ground_net_name: Optional[str] = None

        self._filter_circuit_elements() 

        self.node_map: Dict[str, int] = {}
        self.node_count: int = 0
        self.port_names: List[str] = [] 
        self.port_indices: List[int] = [] 
        self.port_ref_impedances: Dict[str, Quantity] = {}

        self._external_node_indices_reduced: List[int] = []
        self._internal_node_indices_reduced: List[int] = []

        self._cached_rows: Optional[np.ndarray] = None
        self._cached_cols: Optional[np.ndarray] = None
        self._sparsity_nnz: int = 0
        self._shape_full: Tuple[int, int] = (0, 0)

        if not self._effective_nets and self.active_nets_override is not None:
             logger.warning(f"MNA Assembler for '{circuit.name}': No effective nets after filtering with active_nets_override. "
                           f"Assembler will be empty (node_count=0).")
             self.node_count = 0
             self._shape_full = (0,0)
             self._cached_rows = np.array([], dtype=int)
             self._cached_cols = np.array([], dtype=int)
             self._sparsity_nnz = 0
             self._identify_reduced_indices() # Sets reduced indices to empty lists
             return 
        elif not self.circuit_orig.nets : # Original circuit itself has no nets
            # This case implies _effective_nets will also be empty if active_nets_override is None.
            # If active_nets_override is not None but circuit_orig.nets is empty, _filter_circuit_elements handles logging.
            if self.active_nets_override is None: # Only raise if trying to build for an inherently empty circuit
                raise MnaInputError(f"Circuit '{circuit.name}' contains no nets.")
            # If active_nets_override was specified for an empty circuit, it's already handled above.


        try:
            self._assign_node_indices()      
            self._store_reference_impedances()
            self._identify_reduced_indices()  
            self._compute_sparsity_pattern()  
        except Exception as e:
            logger.error(f"Error during MNA Assembler init for '{self.circuit_orig.name}': {e}", exc_info=True)
            if isinstance(e, (pint.DimensionalityError, ValueError, MnaInputError, ComponentError)):
                raise MnaInputError(f"Init failed for '{self.circuit_orig.name}': {e}") from e
            else:
                raise

        logger.info(f"MNA Assembler initialized for circuit '{self.circuit_orig.name}' (View: {'Filtered by active_nets' if self.active_nets_override else 'Full Circuit'}).")
        logger.debug(f"  Effective Nets: {len(self._effective_nets)}, Effective Sim Components: {len(self._effective_sim_components)}")
        logger.debug(f"  Node count (MNA system): {self.node_count}. Ground node: '{self._effective_ground_net_name}' (index 0 if active).")
        logger.debug(f"  Active External ports ({len(self.port_names)}): {self.port_names} -> MNA Indices: {self.port_indices}")
        logger.debug(f"  Reduced system dimension: {max(0, self.node_count - 1 if self._effective_ground_net_name and self.node_map.get(self._effective_ground_net_name) == 0 else self.node_count)}")
        logger.debug(f"  Reduced External Indices: {self._external_node_indices_reduced}")
        logger.debug(f"  Reduced Internal Indices: {self._internal_node_indices_reduced}")
        logger.debug(f"  Cached sparsity pattern: {self._sparsity_nnz} non-zero elements for shape {self._shape_full}.")

    def _filter_circuit_elements(self):
        logger.debug(f"Filtering circuit elements for MNA based on active_nets_override: {self.active_nets_override is not None}")

        if self.active_nets_override is None:
            self._effective_nets = self.circuit_orig.nets
            self._effective_sim_components = self.circuit_orig.sim_components
            self._effective_raw_components = self.circuit_orig.components
            self._effective_external_ports = self.circuit_orig.external_ports
            self._effective_ground_net_name = self.circuit_orig.ground_net_name
            logger.debug("No active_nets_override: MNA Assembler will use full original circuit.")
            return

        active_nets_set = self.active_nets_override
        self._effective_nets = { name: net_obj for name, net_obj in self.circuit_orig.nets.items() if name in active_nets_set }
        if not self._effective_nets and active_nets_set :
            logger.warning(f"active_nets_override provided ({active_nets_set}), but none of these nets exist in the original circuit or original circuit has no nets. Effective netlist will be empty.")
        
        if self.circuit_orig.ground_net_name in active_nets_set:
            self._effective_ground_net_name = self.circuit_orig.ground_net_name
        else:
            self._effective_ground_net_name = None 
            if self._effective_nets: 
                 logger.debug(f"Original ground net '{self.circuit_orig.ground_net_name}' is NOT in active_nets_override. MNA system will not have original ground as reference unless explicitly chosen.")

        self._effective_sim_components = {}
        self._effective_raw_components = {}
        for comp_id, sim_comp in self.circuit_orig.sim_components.items():
            raw_comp_data = self.circuit_orig.components.get(comp_id)
            if not raw_comp_data: continue 

            all_comp_nets_active = True
            if not raw_comp_data.ports: 
                all_comp_nets_active = True 
            else:
                for port_obj in raw_comp_data.ports.values():
                    if port_obj.net:
                        if port_obj.net.name not in active_nets_set:
                            all_comp_nets_active = False
                            break
            
            if all_comp_nets_active:
                self._effective_sim_components[comp_id] = sim_comp
                self._effective_raw_components[comp_id] = raw_comp_data
        
        self._effective_external_ports = { name: net_obj for name, net_obj in self.circuit_orig.external_ports.items() if name in active_nets_set }
        logger.debug(f"Filtering complete: Effective nets: {len(self._effective_nets)}, Sim_comps: {len(self._effective_sim_components)}, Ext_ports: {len(self._effective_external_ports)}, Ground: {self._effective_ground_net_name}")

    def _assign_node_indices(self):
        self.node_map = {}
        idx_counter = 0

        if not self._effective_nets:
            self.node_count = 0
            self.port_names = []
            self.port_indices = []
            self._shape_full = (0,0)
            logger.debug("No effective nets. Node count is 0. No ports assigned for MNA system.")
            return

        if self._effective_ground_net_name:
            self.node_map[self._effective_ground_net_name] = idx_counter
            idx_counter += 1
        # If no _effective_ground_net_name, node 0 will be assigned to the first sorted net below.
        
        for net_name in sorted(self._effective_nets.keys()):
            if net_name == self._effective_ground_net_name: 
                continue
            if net_name in self.node_map: continue 

            self.node_map[net_name] = idx_counter
            idx_counter += 1
        
        self.node_count = idx_counter
        self._shape_full = (self.node_count, self.node_count)

        if self.node_count > 0 and self._effective_ground_net_name is None:
             if self._effective_external_ports or len(self._effective_sim_components) > 0 :
                 # If there's something to solve (ports or components) and no ground is active,
                 # it's an issue for standard MNA. run_sweep should have caught this if ports are involved.
                 # If only internal components, one active net will become reference 0.
                 first_active_net_as_ref = sorted(self._effective_nets.keys())[0] if self._effective_nets else "None"
                 logger.warning(f"MNA system has active elements but no active ground net. Net '{first_active_net_as_ref}' "
                                f"will be MNA reference (index 0). System might be singular if floating and node_count > 1.")

        self.port_names = sorted(self._effective_external_ports.keys())
        self.port_indices = []
        port_mna_indices_found = set()
        for name in self.port_names:
            if name not in self.node_map:
                 raise MnaInputError(f"Active external port '{name}' (net '{name}') not found in built node_map of active nets. Effective nets: {list(self._effective_nets.keys())}, Node map: {self.node_map}")
            if self._effective_ground_net_name and name == self._effective_ground_net_name:
                 raise MnaInputError(f"Active external port '{name}' cannot be the active ground net ('{self._effective_ground_net_name}').")
            
            mna_idx_for_port = self.node_map[name]
            if mna_idx_for_port in port_mna_indices_found:
                 raise MnaInputError(f"Internal Error: Multiple active external ports map to MNA index {mna_idx_for_port}. Port: '{name}'.")

            self.port_indices.append(mna_idx_for_port)
            port_mna_indices_found.add(mna_idx_for_port)
        logger.debug(f"Node indices assigned for {self.node_count} active nets. Active ports: {self.port_names} -> MNA indices: {self.port_indices}")

    def _store_reference_impedances(self):
        self.port_ref_impedances = {}
        for port_name in self.port_names: # self.port_names now only contains active ports
            z0_str = self.circuit_orig.external_port_impedances.get(port_name)
            if z0_str is None:
                raise MnaInputError(f"Missing Z0 string for active external port '{port_name}' in original circuit.")
            try:
                z0 = self.ureg.Quantity(z0_str)
                if not z0.is_compatible_with("ohm"):
                    raise pint.DimensionalityError(z0.units, self.ureg.ohm, msg=f"Port '{port_name}' Z0 '{z0_str}'")
                self.port_ref_impedances[port_name] = z0
            except pint.DimensionalityError as e:
                raise MnaInputError(f"Invalid Z0 unit for {e.msg}. Expected ohms.") from e
            except Exception as e:
                raise MnaInputError(f"Cannot parse Z0 for active port '{port_name}' (Z0='{z0_str}'): {e}") from e
        logger.debug(f"Reference impedances stored for {len(self.port_ref_impedances)} active ports.")

    def _identify_reduced_indices(self):
        self._external_node_indices_reduced = []
        self._internal_node_indices_reduced = []

        if self.node_count == 0: return
        
        # Determine if index 0 is the designated ground for reduction purposes
        # Standard MNA reduction removes the ground node, assumed to be at index 0.
        index_0_is_ground_for_reduction = False
        if self._effective_ground_net_name and self.node_map.get(self._effective_ground_net_name) == 0:
            index_0_is_ground_for_reduction = True
        elif self._effective_ground_net_name is None and self.node_count > 0 :
             # No specific ground active, but system has nodes. Index 0 is an arbitrary reference.
             # For Schur complement, we still "remove" this arbitrary reference.
             index_0_is_ground_for_reduction = True # Treat index 0 as the reference to remove.

        if self.node_count == 1:
            # Single node system. If it's a port, it maps to reduced index 0.
            # If index_0_is_ground_for_reduction is true, this single node is ground, reduced system is 0x0.
            # If index_0_is_ground_for_reduction is false (e.g. single active non-ground node), it's reduced index 0.
            if self.port_indices and self.port_indices[0] == 0 and not index_0_is_ground_for_reduction : # Port is the single node, not ground
                 self._external_node_indices_reduced = [0]
            # else, empty reduced indices (e.g. single node is ground)
            return

        external_full_indices_set = set(self.port_indices)
        
        for full_idx in range(self.node_count):
            if index_0_is_ground_for_reduction and full_idx == 0:
                continue 
            
            reduced_idx = full_idx - 1 if index_0_is_ground_for_reduction else full_idx
            
            if full_idx in external_full_indices_set:
                self._external_node_indices_reduced.append(reduced_idx)
            else:
                self._internal_node_indices_reduced.append(reduced_idx)
        
        self._external_node_indices_reduced.sort()
        self._internal_node_indices_reduced.sort()
        
        expected_reduced_dim = self.node_count - 1 if index_0_is_ground_for_reduction else self.node_count
        if expected_reduced_dim < 0: expected_reduced_dim = 0 # For node_count = 0

        if len(self._external_node_indices_reduced) != len(self.port_indices):
             # This can happen if a port IS the ground node that's removed.
             # _assign_node_indices should prevent ports from being the active ground.
             # This check is more about consistency after reduction.
             if not (index_0_is_ground_for_reduction and self.port_indices and 0 in external_full_indices_set):
                # Only error if mismatch isn't due to a port being the removed ground
                logger.error(f"Reduced external indices count ({len(self._external_node_indices_reduced)}) "
                               f"mismatch with active port count ({len(self.port_names)}). External MNA indices: {self.port_indices}.")


        if (len(self._external_node_indices_reduced) + len(self._internal_node_indices_reduced)) != expected_reduced_dim:
             raise MnaInputError(f"Internal Error: Sum of reduced external/internal indices "
                                f"({len(self._external_node_indices_reduced)} + {len(self._internal_node_indices_reduced)})="
                                f"{len(self._external_node_indices_reduced) + len(self._internal_node_indices_reduced)} "
                                f"does not match expected reduced dimension {expected_reduced_dim} (node_count={self.node_count}, ground_is_0={index_0_is_ground_for_reduction}).")
        logger.debug(f"Reduced indices identified: External={self._external_node_indices_reduced}, Internal={self._internal_node_indices_reduced}")


    @property
    def external_node_indices_reduced(self) -> List[int]:
        return self._external_node_indices_reduced

    @property
    def internal_node_indices_reduced(self) -> List[int]:
        return self._internal_node_indices_reduced

    def _compute_sparsity_pattern(self):
        logger.debug("Computing MNA sparsity pattern using *effective* component connectivity...")
        rows, cols, data = [], [], []
        
        if self.node_count == 0: 
            self._cached_rows = np.array([], dtype=int)
            self._cached_cols = np.array([], dtype=int)
            self._sparsity_nnz = 0
            logger.debug("No active nodes, MNA sparsity pattern is empty.")
            return

        for comp_id, sim_comp in self._effective_sim_components.items():
            if comp_id not in self._effective_raw_components:
                 raise MnaInputError(f"Internal Error: Effective sim_comp '{comp_id}' missing from effective_raw_components for sparsity.")
            comp_data = self._effective_raw_components[comp_id]
            ComponentClass = type(sim_comp)

            try:
                connectivity = ComponentClass.declare_connectivity()
                declared_ports_by_type = set(ComponentClass.declare_ports())

                for port_id1_declared, port_id2_declared in connectivity:
                    if port_id1_declared not in declared_ports_by_type or \
                       str(port_id1_declared) not in declared_ports_by_type: # Check str version too
                         # Check if the string version is in declared_ports_by_type if port_id1_declared is int and vice-versa
                         # This logic might be complex if declare_ports can return mixed types and connectivity can too.
                         # Assuming declared_ports_by_type contains canonical IDs.
                         is_p1_declared = port_id1_declared in declared_ports_by_type
                         if not is_p1_declared: # Try converting for check
                            if isinstance(port_id1_declared, int) and str(port_id1_declared) in declared_ports_by_type: is_p1_declared = True
                            elif isinstance(port_id1_declared, str) and int(port_id1_declared) in declared_ports_by_type: is_p1_declared = True
                         
                         is_p2_declared = port_id2_declared in declared_ports_by_type
                         if not is_p2_declared:
                            if isinstance(port_id2_declared, int) and str(port_id2_declared) in declared_ports_by_type: is_p2_declared = True
                            elif isinstance(port_id2_declared, str) and int(port_id2_declared) in declared_ports_by_type: is_p2_declared = True

                         if not is_p1_declared or not is_p2_declared:
                              raise ComponentError(f"Comp type '{ComponentClass.component_type_str}' (instance '{comp_id}') uses undeclared port(s) "
                                                  f"'{port_id1_declared if not is_p1_declared else ''}{',' if not is_p1_declared and not is_p2_declared else ''}{port_id2_declared if not is_p2_declared else ''}' "
                                                  f"in its declare_connectivity(). Declared ports by type: {declared_ports_by_type}.")


                    # Use the robust lookup helper
                    port1_obj = _get_port_obj_from_comp_data(comp_data.ports, port_id1_declared)
                    port2_obj = _get_port_obj_from_comp_data(comp_data.ports, port_id2_declared)

                    # Enhanced error checking after lookup
                    if not port1_obj:
                        raise MnaInputError(f"Sparsity: Port '{port_id1_declared}' (declared by type {ComponentClass.component_type_str}) "
                                            f"not found in YAML-defined instance data for component '{comp_id}'. "
                                            f"Instance ports (from YAML): {list(comp_data.ports.keys())}")
                    if not port1_obj.net:
                        raise MnaInputError(f"Sparsity: Port '{port_id1_declared}' of component '{comp_id}' is not connected to any net ('port.net' is None).")
                    if port1_obj.net.name not in self.node_map:
                        raise MnaInputError(f"Sparsity: Port '{port_id1_declared}' of component '{comp_id}' connects to net "
                                            f"'{port1_obj.net.name}', which is not in the active node_map. "
                                            f"This indicates an inconsistency in filtering logic or an unconnected active component. "
                                            f"Active node_map keys: {list(self.node_map.keys())}")

                    if not port2_obj:
                        raise MnaInputError(f"Sparsity: Port '{port_id2_declared}' (declared by type {ComponentClass.component_type_str}) "
                                            f"not found in YAML-defined instance data for component '{comp_id}'. "
                                            f"Instance ports (from YAML): {list(comp_data.ports.keys())}")
                    if not port2_obj.net:
                        raise MnaInputError(f"Sparsity: Port '{port_id2_declared}' of component '{comp_id}' is not connected to any net ('port.net' is None).")
                    if port2_obj.net.name not in self.node_map:
                        raise MnaInputError(f"Sparsity: Port '{port_id2_declared}' of component '{comp_id}' connects to net "
                                            f"'{port2_obj.net.name}', which is not in the active node_map. "
                                            f"This indicates an inconsistency in filtering logic or an unconnected active component. "
                                            f"Active node_map keys: {list(self.node_map.keys())}")
                                            
                    net1_name, net2_name = port1_obj.net.name, port2_obj.net.name
                    n1, n2 = self.node_map[net1_name], self.node_map[net2_name]
        
                    self._add_to_coo_pattern(rows, cols, data, n1, n1, 1.0)
                    if n1 != n2:
                        self._add_to_coo_pattern(rows, cols, data, n1, n2, 1.0)
                        self._add_to_coo_pattern(rows, cols, data, n2, n1, 1.0)
                    self._add_to_coo_pattern(rows, cols, data, n2, n2, 1.0)
            except KeyError as e: # Should be caught by specific checks above now
                raise MnaInputError(f"Sparsity/KeyError: Comp '{comp_id}', net '{e}' not in active node_map (should have been caught by specific checks).")
            except ComponentError as e:
                 logger.error(f"Component definition error for '{comp_id}' type '{ComponentClass.__name__}': {e}")
                 raise
            except Exception as e:
                logger.error(f"Sparsity/Unexpected error for comp '{comp_id}': {e}", exc_info=True)
                raise MnaInputError(f"Failed connectivity for comp '{comp_id}': {e}") from e
        
        try:
            if not rows: 
                temp_coo = sp.coo_matrix(self._shape_full, dtype=np.complex128)
            else:
                temp_coo = sp.coo_matrix((data, (rows, cols)), shape=self._shape_full, dtype=np.complex128)
            
            temp_csc = temp_coo.tocsc(); temp_csc.eliminate_zeros(); final_coo = temp_csc.tocoo()
            self._cached_rows, self._cached_cols, self._sparsity_nnz = final_coo.row, final_coo.col, final_coo.nnz
            logger.debug(f"Sparsity pattern computed for active elements: NNZ={self._sparsity_nnz} for shape {self._shape_full}")
        except Exception as e:
            logger.error(f"Failed to create sparse matrix for sparsity pattern: {e}", exc_info=True)
            raise RuntimeError(f"Failed to compute MNA sparsity pattern: {e}") from e

    def _add_to_coo_pattern(self, rows: list, cols: list, data: list, r: int, c: int, val: complex):
        if not (0 <= r < self.node_count and 0 <= c < self.node_count):
            raise MnaInputError(f"Internal Error (Pattern): Stamp indices ({r}, {c}) out of bounds for active matrix size {self.node_count}.")
        if abs(val) > 1e-18: 
            rows.append(r); cols.append(c); data.append(val)

    def assemble(self, freq_hz: float) -> sp.csc_matrix:
        if freq_hz < 0: raise MnaInputError(f"MNA Assembly requires freq >= 0 Hz. Got {freq_hz} Hz.")
        if self._cached_rows is None or self._cached_cols is None:
             raise RuntimeError("Sparsity pattern not computed or invalid.")

        if self.node_count == 0:
            logger.debug(f"Assemble: System has 0 nodes. Returning empty 0x0 matrix for F={freq_hz:.4e} Hz.")
            return sp.csc_matrix((0,0), dtype=np.complex128)

        freq_str_log = f"{freq_hz:.4e} Hz"
        logger.debug(f"Assembling MNA matrix for {freq_str_log} using cached pattern (NNZ={self._sparsity_nnz}, Shape={self._shape_full})...")
        mna_data = np.zeros(self._sparsity_nnz, dtype=np.complex128)
        rc_to_data_idx: Dict[Tuple[int, int], int] = {
            (r, c): i for i, (r, c) in enumerate(zip(self._cached_rows, self._cached_cols))
        }
        def add_value_to_data(r: int, c: int, value: complex, comp_id_for_err: str):
            key = (r, c)
            if key in rc_to_data_idx:
                mna_data[rc_to_data_idx[key]] += value
            else:
                if abs(value) > 1e-15:
                    raise MnaInputError(
                        f"Component '{comp_id_for_err}' tried to stamp non-zero ({value:.3e}) at ({r},{c}), "
                        f"outside predicted sparsity for MNA shape {self._shape_full}. "
                        f"Check declare_connectivity vs get_mna_stamps or active net filtering."
                    )

        current_freq_array = np.array([freq_hz])
        evaluation_context: Dict[Tuple[str, str], Quantity] = {}

        for comp_id, sim_comp in self._effective_sim_components.items():
            if comp_id not in self._effective_raw_components:
                raise MnaInputError(f"Internal Error: sim_comp '{comp_id}' missing from raw_comp map during assembly.")
            comp_data = self._effective_raw_components[comp_id]

            resolved_params_dict: Dict[str, Quantity] = {}
            component_param_specs = type(sim_comp).declare_parameters()
            try:
                for internal_name in sim_comp.parameter_internal_names:
                    name_parts = internal_name.split('.', 1)
                    if len(name_parts)!=2: raise ComponentError(f"Invalid internal param name '{internal_name}' for '{comp_id}'.")
                    base_name = name_parts[1]
                    if base_name not in component_param_specs:
                         raise ComponentError(f"Internal Error: Param base name '{base_name}' not declared for '{comp_id}'.")
                    expected_dimension_str = component_param_specs[base_name]
                    resolved_qty = self.parameter_manager.resolve_parameter(
                        internal_name, current_freq_array, expected_dimension_str, evaluation_context
                    )
                    if not resolved_qty.is_compatible_with(expected_dimension_str):
                        raise pint.DimensionalityError(resolved_qty.units, self.ureg.parse_units(expected_dimension_str),
                                                       extra_msg=f"for param '{internal_name}' of '{comp_id}'. Expected '{expected_dimension_str}'.")
                    resolved_params_dict[base_name] = resolved_qty

                stamp_infos = sim_comp.get_mna_stamps(current_freq_array, resolved_params_dict)
            except (ParameterError, ParameterScopeError, pint.DimensionalityError) as e:
                raise ComponentError(f"Param resolution/validation failed for '{comp_id}' at {freq_str_log}: {e}") from e
            except ComponentError: raise
            except Exception as e:
                raise ComponentError(f"Unexpected error processing params/stamps for '{comp_id}' at {freq_str_log}: {e}") from e

            try:
                for stamp_idx, stamp_info in enumerate(stamp_infos):
                    admittance_matrix_qty, port_ids_yaml = stamp_info
                    if not isinstance(admittance_matrix_qty, Quantity): raise TypeError(f"Comp '{comp_id}' stamp mat != Quantity")
                    if not admittance_matrix_qty.is_compatible_with(self.ureg.siemens):
                        raise pint.DimensionalityError(
                            admittance_matrix_qty.units,
                            self.ureg.siemens, # Target unit for user understanding
                            admittance_matrix_qty.dimensionality,
                            self.ureg.siemens.dimensionality, # Correct target dimensionality
                            extra_msg=(
                                f"Component '{comp_id}' MNA stamp contributions (units: {admittance_matrix_qty.units:~P}) "
                                f"are not dimensionally compatible with admittance/conductance (expected: Siemens)."
                            )
                        )

                    stamp_values = admittance_matrix_qty.to(self.ureg.siemens).magnitude
                    current_stamp_matrix: np.ndarray
                    if stamp_values.ndim == 3: current_stamp_matrix = stamp_values[0, :, :]
                    elif stamp_values.ndim == 2: current_stamp_matrix = stamp_values
                    elif stamp_values.ndim == 0 and stamp_values.size == 1 and len(port_ids_yaml) == 1: # Corrected from problem description's version
                         current_stamp_matrix = np.array([[stamp_values.item()]])
                    else:
                         raise ValueError(f"Comp '{comp_id}' stamp has unexpected shape {stamp_values.shape}.")

                    global_indices = []
                    for port_yaml_id in port_ids_yaml:
                        # --- Use the helper function for lookup ---
                        port_obj = _get_port_obj_from_comp_data(comp_data.ports, port_yaml_id)
                        # --- End of change ---
                        if not (port_obj and port_obj.net and port_obj.net.name in self.node_map):
                            raise MnaInputError(f"Comp '{comp_id}' (active) trying to stamp port '{port_yaml_id}', "
                                       f"but its net ('{port_obj.net.name if port_obj and port_obj.net else 'N/A'}') "
                                       f"is not in active node_map. Inconsistency in filtering.")
                        global_indices.append(self.node_map[port_obj.net.name])

                    num_ports_in_stamp = len(port_ids_yaml)
                    if current_stamp_matrix.shape != (num_ports_in_stamp, num_ports_in_stamp):
                         raise ValueError(f"Comp '{comp_id}' stamp matrix shape {current_stamp_matrix.shape} != num ports {num_ports_in_stamp}.")

                    for i in range(num_ports_in_stamp):
                        for j in range(num_ports_in_stamp):
                            add_value_to_data(global_indices[i], global_indices[j], current_stamp_matrix[i,j], comp_id)
            except MnaInputError: raise
            except ComponentError as e:
                raise ComponentError(f"Failed processing '{comp_id}' stamps at {freq_str_log}: {e}") from e
            except Exception as e:
                raise ComponentError(f"Unexpected failure processing stamps for '{comp_id}' at {freq_str_log}: {e}") from e

        try:
            Yn_full_coo = sp.coo_matrix((mna_data, (self._cached_rows, self._cached_cols)), shape=self._shape_full, dtype=np.complex128)
            Yn_full_csc = Yn_full_coo.tocsc(); Yn_full_csc.eliminate_zeros()
            logger.debug(f"Full MNA matrix assembly complete for {freq_str_log}. Shape: {Yn_full_csc.shape}, NNZ: {Yn_full_csc.nnz}")
            return Yn_full_csc
        except Exception as e:
            raise RuntimeError(f"Failed creating sparse MNA matrix at {freq_str_log}: {e}") from e