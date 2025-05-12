# --- src/rfsim_core/simulation/mna.py ---
import logging
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple, Optional, Set

from ..data_structures import Circuit, Net
from ..data_structures import Component as ComponentData
from ..components.base import ComponentBase, ComponentError
from ..units import ureg, Quantity, pint
from ..components import LARGE_ADMITTANCE_SIEMENS
from ..parameters import ParameterManager, ParameterScopeError, ParameterError

logger = logging.getLogger(__name__)

class MnaInputError(ValueError):
    """Error related to inputs for MNA assembly."""
    pass

class MnaAssembler:
    """
    Assembles the MNA matrix (Yn) for a circuit based *only* on component topology.
    Initialization is frequency-independent. Assembly is done per frequency.
    Handles AC analysis (F > 0) only in Phase 4. Caches sparsity pattern.
    Identifies external/internal nodes relative to the reduced system.
    Stores port reference impedances for potential later use (e.g., S-params).
    """
    def __init__(self, circuit: Circuit):
        """
        Initializes the assembler, calculates node maps, port info, sparsity pattern,
        and identifies external/internal node indices for the reduced system.

        Args:
            circuit: The simulation-ready Circuit object (must have 'sim_components').

        Raises:
            MnaInputError: If circuit is invalid or has configuration issues.
        """
        # Check if it's a simulation-ready circuit
        if not isinstance(circuit, Circuit) or not hasattr(circuit, 'sim_components'):
             raise MnaInputError("Circuit object must be simulation-ready (output of CircuitBuilder).")
        if not circuit.nets:
             raise MnaInputError("Circuit contains no nets.")
        if not circuit.sim_components and not circuit.external_ports:
            # Allow circuits with only external ports (e.g., just measuring ports connected together)
            # But warn if there are no components *and* no ports.
             logger.warning(f"Circuit '{circuit.name}' contains no components and no external ports.")
        elif not circuit.sim_components:
             logger.warning(f"Circuit '{circuit.name}' contains no components, only external ports.")

        self.circuit = circuit
        self.ureg = ureg
        # Use the sim_components from the passed-in simulation-ready circuit
        self.sim_components: Dict[str, ComponentBase] = circuit.sim_components
        # Still need raw component data for port->net mapping during stamping
        self.raw_components: Dict[str, ComponentData] = circuit.components # For port->net mapping

        # Access ParameterManager from the circuit
        if not isinstance(circuit.parameter_manager, ParameterManager):
            raise MnaInputError("Circuit's ParameterManager is not initialized or invalid.")
        self.parameter_manager: ParameterManager = circuit.parameter_manager

        # --- Frequency Independent Setup ---
        self.node_map: Dict[str, int] = {}
        self.node_count: int = 0
        self.port_indices: List[int] = [] # Full indices
        self.port_names: List[str] = []
        self.port_ref_impedances: Dict[str, Quantity] = {}
    
        # --- Indices relative to REDUCED system ---
        self._external_node_indices_reduced: List[int] = []
        self._internal_node_indices_reduced: List[int] = []

        # --- Sparsity Pattern Cache ---
        self._cached_rows: Optional[np.ndarray] = None
        self._cached_cols: Optional[np.ndarray] = None
        self._sparsity_nnz: Optional[int] = None
        self._shape_full: Tuple[int, int] = (0, 0)

        try:
            self._assign_node_indices()
            self._store_reference_impedances()
            self._identify_reduced_indices()
            self._compute_sparsity_pattern() # Requires node map and components
        except Exception as e:
            logger.error(f"Error during MNA Assembler initialization: {e}", exc_info=True)
            if isinstance(e, (pint.DimensionalityError, ValueError, MnaInputError, ComponentError)): # Added ComponentError
                 raise MnaInputError(f"Initialization failed: {e}") from e
            else:
                 raise

        logger.info(f"MNA Assembler initialized for '{circuit.name}'.")
        logger.debug(f"Node count (full): {self.node_count}. Ground node: '{circuit.ground_net_name}' (index 0).")
        logger.debug(f"External ports ({len(self.port_names)}): {self.port_names} -> Full Indices: {self.port_indices}")
        logger.debug(f"Reduced system dimension: {self.node_count - 1}")
        logger.debug(f"Reduced External Indices: {self._external_node_indices_reduced}")
        logger.debug(f"Reduced Internal Indices: {self._internal_node_indices_reduced}")
        logger.debug(f"Cached sparsity pattern: {self._sparsity_nnz} non-zero elements.")

    @property
    def external_node_indices_reduced(self) -> List[int]:
        """Returns the sorted list of reduced node indices corresponding to external ports."""
        return self._external_node_indices_reduced

    @property
    def internal_node_indices_reduced(self) -> List[int]:
        """Returns the sorted list of reduced node indices corresponding to internal nodes."""
        return self._internal_node_indices_reduced

    def _assign_node_indices(self):
        """Assigns unique integer indices (0..N-1) to each net, ground is always 0."""
        ground_name = self.circuit.ground_net_name
        if ground_name not in self.circuit.nets:
             raise MnaInputError(f"Ground net '{ground_name}' not found in the circuit nets.")

        self.node_map = {}
        idx_counter = 0
        self.node_map[ground_name] = idx_counter
        idx_counter += 1

        for net_name in sorted(self.circuit.nets.keys()):
            if net_name != ground_name:
                if net_name in self.node_map: continue
                self.node_map[net_name] = idx_counter
                idx_counter += 1
        self.node_count = idx_counter
        self._shape_full = (self.node_count, self.node_count)

        if self.node_count <= 0: # Should have at least ground if nets exist
             raise MnaInputError("Internal Error: Node count is zero or negative after indexing.")

        # Map external port names to their *full* indices, sorted by port name
        self.port_names = sorted(self.circuit.external_ports.keys())
        self.port_indices = []
        port_nets_found = set()
        for name in self.port_names:
            port_net_name = name # Port name IS the net name
            if port_net_name not in self.node_map:
                 raise MnaInputError(f"External port '{name}' refers to net ('{port_net_name}') not found in node map. Is it connected?")
            if port_net_name == ground_name:
                 # Standard MNA/S-params usually assume ports are not ground
                 raise MnaInputError(f"External port '{name}' cannot be the ground net ('{ground_name}').")
            if port_net_name in port_nets_found:
                 raise MnaInputError(f"Multiple external ports defined on the same net '{port_net_name}'.")

            self.port_indices.append(self.node_map[port_net_name])
            port_nets_found.add(port_net_name)


    def _store_reference_impedances(self):
        """Parses Z0 strings and stores them as Quantities."""
        self.port_ref_impedances = {}
        for port_name, z0_str in self.circuit.external_port_impedances.items():
            try:
                z0 = self.ureg.Quantity(z0_str)
                if not z0.is_compatible_with("ohm"):
                    raise pint.DimensionalityError(z0.units, self.ureg.ohm, msg=f"Port '{port_name}' impedance '{z0_str}'")

                # Store the impedance quantity (could be complex later)
                self.port_ref_impedances[port_name] = z0
                logger.debug(f"Stored Port '{port_name}': Z0={z0:~P}")

            except pint.DimensionalityError as e:
                 raise MnaInputError(f"Invalid reference impedance unit for {e.msg}. Expected ohms.") from e
            except Exception as e:
                 raise MnaInputError(f"Cannot pars  e reference impedance for port '{port_name}' (Z0='{z0_str}'): {e}") from e

    def _identify_reduced_indices(self):
        """
        Identifies external and internal node indices relative to the reduced system
        (after ground removal).
        """
        # Set of full indices corresponding to external ports (already validated > 0)
        external_full_indices = set(self.port_indices)
        # All possible reduced indices range from 0 to node_count-2
        all_reduced_indices = set(range(self.node_count - 1))

        ext_reduced = set()
        int_reduced = set()

        # Iterate through full map (excluding ground 0)
        for net_name, full_idx in self.node_map.items():
            if full_idx == 0: # Skip ground
                 continue

            reduced_idx = full_idx - 1 # Calculate corresponding reduced index

            if full_idx in external_full_indices:
                ext_reduced.add(reduced_idx)
            else:
                int_reduced.add(reduced_idx)

        # Store as sorted lists
        self._external_node_indices_reduced = sorted(list(ext_reduced))
        self._internal_node_indices_reduced = sorted(list(int_reduced))

        # Sanity check
        if len(self._external_node_indices_reduced) != len(self.port_indices):
             raise MnaInputError("Internal Error: Count mismatch between ports and reduced external indices.")
        if len(ext_reduced) + len(int_reduced) != (self.node_count - 1):
             raise MnaInputError("Internal Error: Sum of reduced external/internal indices doesn't match reduced dimension.")


    def _compute_sparsity_pattern(self):
        """
        Computes the MNA matrix sparsity pattern based on component declared connectivity.
        """
        logger.debug("Computing MNA sparsity pattern using component connectivity...")
        rows, cols, data = [], [], []
        self._shape_full = (self.node_count, self.node_count) # Ensure shape is set

        # Check if there are components to process
        if not self.sim_components:
             logger.debug("No simulation components found, sparsity pattern will be empty (diagonal only if needed).")
             # Handle case with only ports later if necessary, for now pattern is empty
             # Create empty pattern arrays
             self._cached_rows = np.array([], dtype=int)
             self._cached_cols = np.array([], dtype=int)
             self._sparsity_nnz = 0
             return # Exit early

        # Iterate through simulation-ready components to get their type and connectivity
        for comp_id, sim_comp in self.sim_components.items():
            # Need raw data for port->net map
            if comp_id not in self.raw_components:
                 # This would indicate an internal inconsistency
                 raise MnaInputError(f"Internal Error: Simulation component '{comp_id}' not found in raw component data during sparsity calculation.")
            comp_data = self.raw_components[comp_id]
            ComponentClass = type(sim_comp)

            try:
                connectivity = ComponentClass.declare_connectivity()
                declared_ports = set(ComponentClass.declare_ports())

                logger.debug(f"Component '{comp_id}' (type {ComponentClass.component_type_str}) declared connectivity: {connectivity}")

                for port_id1, port_id2 in connectivity:
                    # Basic validation: are these ports declared by the class?
                    if port_id1 not in declared_ports or port_id2 not in declared_ports:
                         raise ComponentError(f"Component type '{ComponentClass.component_type_str}' declares connectivity between '{port_id1}' and '{port_id2}', but one or both are not in its declared_ports list {declared_ports}. Fix component definition.")

                    # Check if these ports actually exist AND ARE CONNECTED in the instance
                    port1_obj = comp_data.ports.get(port_id1)
                    port2_obj = comp_data.ports.get(port_id2)

                    if not port1_obj or not port1_obj.net:
                        logger.warning(f"Sparsity calculation for '{comp_id}': Declared connectivity uses port '{port_id1}', but it's not connected in the instance. Skipping pair ({port_id1}, {port_id2}).")
                        continue
                    if not port2_obj or not port2_obj.net:
                        logger.warning(f"Sparsity calculation for '{comp_id}': Declared connectivity uses port '{port_id2}', but it's not connected in the instance. Skipping pair ({port_id1}, {port_id2}).")
                        continue

                    # Get the nets these ports are connected to
                    net1_name = port1_obj.net.name
                    net2_name = port2_obj.net.name

                    # Get the global node indices for these nets
                    n1 = self.node_map[net1_name]
                    n2 = self.node_map[net2_name]

                    # Add non-zero entries for this connection to the COO pattern
                    # Use a dummy value (1.0) for pattern calculation
                    # The helper handles bounds checks
                    self._add_to_coo_pattern(rows, cols, data, n1, n1, 1.0)
                    if n1 != n2: # Avoid adding off-diagonal twice if self-loop
                        self._add_to_coo_pattern(rows, cols, data, n1, n2, 1.0)
                        self._add_to_coo_pattern(rows, cols, data, n2, n1, 1.0)
                    self._add_to_coo_pattern(rows, cols, data, n2, n2, 1.0)


            except KeyError as e:
                # More specific error message for net name not found
                raise MnaInputError(f"Error getting node index for component '{comp_id}' during pattern build: Net name '{e}' (from port connection) not found in node map. Check netlist connections.")
            except AttributeError as e:
                 # Catch errors like trying to access .net on None if port lookup failed unexpectedly
                 raise MnaInputError(f"Error accessing port/net connection for component '{comp_id}' during pattern build: {e}. Check netlist structure.")
            except ComponentError as e: # Catch specific ComponentError raised above
                 logger.error(f"Component definition error for '{comp_id}' type '{ComponentClass.__name__}': {e}")
                 raise # Re-raise ComponentError
            except Exception as e:
                logger.error(f"Unexpected error getting connectivity for component '{comp_id}' type '{ComponentClass.__name__}': {e}", exc_info=True)
                raise MnaInputError(f"Failed to get connectivity for component '{comp_id}': {e}") from e

        # Create sparse matrix from COO lists (unchanged logic)
        try:
            if not rows: # Handle empty circuit or circuit with no connected components
                logger.debug("No component connectivity found to build sparsity pattern. Using empty pattern.")
                temp_coo = sp.coo_matrix(self._shape_full, dtype=np.complex128)
            else:
                temp_coo = sp.coo_matrix((data, (rows, cols)), shape=self._shape_full, dtype=np.complex128)

            # Consolidate duplicates (important!) and convert to CSC/COO
            temp_csc = temp_coo.tocsc()
            temp_csc.eliminate_zeros()
            final_coo = temp_csc.tocoo() # Get final pattern indices

            self._cached_rows = final_coo.row
            self._cached_cols = final_coo.col
            self._sparsity_nnz = final_coo.nnz
            logger.debug(f"Sparsity pattern computed: NNZ={self._sparsity_nnz}")

        except Exception as e:
            logger.error(f"Failed to create sparse matrix for sparsity pattern: {e}", exc_info=True)
            raise RuntimeError(f"Failed to compute MNA sparsity pattern: {e}") from e
        

    def _add_to_coo_pattern(self, rows: list, cols: list, data: list, r: int, c: int, val: complex):
        """ Helper for adding to COO lists during pattern generation. """
        if not (0 <= r < self.node_count and 0 <= c < self.node_count):
            # This indicates a bug in index calculation or component definition
            raise MnaInputError(f"Internal Error (Pattern): Stamp indices ({r}, {c}) out of bounds for matrix size {self.node_count}.")
        # Add non-zero value (1.0 used for pattern)
        if abs(val) > 1e-18: # Basic check against zero
            rows.append(r)
            cols.append(c)
            data.append(val)

    def assemble(self, freq_hz: float) -> sp.csc_matrix:
        """
        Assembles the N x N *full* MNA matrix (Yn_full) for a specific frequency > 0,
        using generalized component stamps and the cached sparsity pattern.
        """
        if freq_hz < 0:
             raise MnaInputError(f"MNA Assembly requires frequency >= 0 Hz. Got {freq_hz} Hz.")
        if self._cached_rows is None or self._cached_cols is None:
             raise RuntimeError("Sparsity pattern was not computed or is invalid.")
        
        freq_str_log = f"{freq_hz:.4e} Hz"
        if freq_hz == 0:
            freq_str_log = "0 Hz (DC - component value eval)"
        logger.debug(f"Assembling MNA matrix for {freq_str_log} using cached pattern (NNZ={self._sparsity_nnz})...")

        # Initialize data array for the current frequency based on cached pattern
        mna_data = np.zeros(self._sparsity_nnz, dtype=np.complex128)
        # Create mapping from (row, col) in pattern to index in mna_data
        rc_to_data_idx: Dict[Tuple[int, int], int] = {
            (r, c): i for i, (r, c) in enumerate(zip(self._cached_rows, self._cached_cols))
        }

        # Helper to add value to the correct index in mna_data
        def add_value_to_data(r: int, c: int, value: complex, comp_id_for_err: str):
            """Adds a value to the MNA data array using the precomputed map."""
            key = (r, c)
            if key in rc_to_data_idx:
                mna_data[rc_to_data_idx[key]] += value
            else:
                # A component is trying to stamp a location that
                # wasn't predicted by declare_connectivity. This is an error.
                if abs(value) > 1e-15: # Only raise if the value is significant
                    raise MnaInputError(
                        f"Component '{comp_id_for_err}' attempted to stamp non-zero value ({value:.3e}) "
                        f"at matrix location ({r}, {c}), which is outside the sparsity pattern predicted by its "
                        f"'declare_connectivity()' method. Ensure the method accurately reflects all "
                        f"connections modified by 'get_mna_stamps()'."
                    )
                # If value is effectively zero, we can ignore it silently.

        # --- Stamp Simulation Components using Generalized Method ---
        current_freq_array = np.array([freq_hz]) # For parameter resolution and component methods
        evaluation_context: Dict[Tuple[str, str], Quantity] = {} # Per-frequency memoization cache

        for comp_id, sim_comp in self.sim_components.items():
            # Get raw data for port->net lookup
            if comp_id not in self.raw_components:
                raise MnaInputError(f"Internal Error: Simulation component '{comp_id}' not found in raw component data during assembly.")
            comp_data = self.raw_components[comp_id]

            resolved_params_dict: Dict[str, Quantity] = {}
            component_param_specs = type(sim_comp).declare_parameters()

            try:
                for internal_name in sim_comp.parameter_internal_names:
                    # Extract base_name (e.g., "resistance" from "R1.resistance")
                    # ParameterManager._parse_internal_name is protected, use simple split
                    name_parts = internal_name.split('.', 1)
                    if len(name_parts) != 2:
                        raise ComponentError(f"Invalid internal parameter name format '{internal_name}' for component '{comp_id}'.")
                    base_name = name_parts[1]

                    if base_name not in component_param_specs:
                        # This would be an internal inconsistency from CircuitBuilder
                        raise ComponentError(f"Internal Error: Parameter base name '{base_name}' (from internal '{internal_name}') "
                                             f"not in declared parameters {list(component_param_specs.keys())} for component '{comp_id}' of type '{type(sim_comp).__name__}'.")
                    
                    expected_dimension_str = component_param_specs[base_name]

                    # Resolve parameter value
                    resolved_qty = self.parameter_manager.resolve_parameter(
                        internal_name,
                        current_freq_array, # Pass as np.array([current_scalar_freq])
                        expected_dimension_str, # Target dimension for the result
                        evaluation_context
                    )

                    # Final Validation (as per plan, though resolve_parameter should ensure this)
                    if not resolved_qty.is_compatible_with(expected_dimension_str):
                        # This error indicates a problem in resolve_parameter or type declaration mismatch
                        raise pint.DimensionalityError(
                            resolved_qty.units,
                            self.ureg.parse_units(expected_dimension_str), # For better error message
                            extra_msg=(f"for resolved parameter '{internal_name}' of component '{comp_id}'. "
                                       f"Expected dimension '{expected_dimension_str}', but "
                                       f"resolve_parameter returned '{resolved_qty.dimensionality}'. "
                                       f"Value: {resolved_qty:~P}.")
                        )
                    
                    resolved_params_dict[base_name] = resolved_qty
                
                # Call component's get_mna_stamps with resolved parameters
                stamp_infos = sim_comp.get_mna_stamps(current_freq_array, resolved_params_dict)

            except (ParameterError, ParameterScopeError, pint.DimensionalityError) as e:
                err_msg = f"Failed to resolve/validate parameters for component '{comp_id}' at {freq_str_log}: {e}"
                logger.error(err_msg, exc_info=True) # Show more detail for param errors
                raise ComponentError(err_msg) from e
            except ComponentError: # Catch ComponentErrors raised from above or during get_mna_stamps
                raise # Re-raise as it already has context
            except Exception as e: # Catch other unexpected errors during param resolution or get_mna_stamps
                err_msg = f"Unexpected error processing parameters or getting stamps for component '{comp_id}' at {freq_str_log}: {type(e).__name__} - {e}"
                logger.error(err_msg, exc_info=True)
                raise ComponentError(err_msg) from e

            try:
                for stamp_idx, stamp_info in enumerate(stamp_infos):
                    admittance_matrix_qty, port_ids = stamp_info
                    if not isinstance(admittance_matrix_qty, Quantity):
                         raise TypeError(f"Comp '{comp_id}' stamp {stamp_idx} matrix != Quantity")
                    if not admittance_matrix_qty.check('siemens'):
                         raise pint.DimensionalityError(admittance_matrix_qty.units, self.ureg.siemens, extra_msg=f"Comp '{comp_id}' stamp {stamp_idx} matrix")

                    stamp_values = admittance_matrix_qty.to(self.ureg.siemens).magnitude
                    current_stamp_matrix: np.ndarray
                    if stamp_values.ndim == 3:
                        if stamp_values.shape[0] == 1:
                            current_stamp_matrix = stamp_values[0, :, :]
                        else:
                             logger.warning(f"Component '{comp_id}' returned unexpected multi-frequency shape {stamp_values.shape} within single-frequency assembly loop. Using first slice.")
                             current_stamp_matrix = stamp_values[0, :, :]
                    elif stamp_values.ndim == 2:
                        current_stamp_matrix = stamp_values
                    elif stamp_values.ndim == 0 and stamp_values.size == 1 and len(port_ids) == 1:
                         current_stamp_matrix = np.array([[stamp_values.item()]])
                    else:
                         raise ValueError(f"Component '{comp_id}' returned stamp with unexpected dimension/shape {stamp_values.shape} (ndim={stamp_values.ndim}, size={stamp_values.size}). Expected 2D (N,N) or 3D (1,N,N).")

                    global_indices = []
                    valid_ports = True
                    for port_id in port_ids:
                        port_obj = comp_data.ports.get(port_id)
                        if not port_obj or not port_obj.net:
                            logger.error(f"Component '{comp_id}' stamp {stamp_idx} refers to port ID '{port_id}' which is not connected in the netlist instance. Skipping stamp.")
                            valid_ports = False
                            break
                        net_name = port_obj.net.name
                        if net_name not in self.node_map:
                             raise MnaInputError(f"Internal Error: Net '{net_name}' from component '{comp_id}' port '{port_id}' not found in node map during assembly.")
                        global_indices.append(self.node_map[net_name])
                    if not valid_ports: continue

                    num_ports_in_stamp = len(port_ids)
                    if current_stamp_matrix.shape != (num_ports_in_stamp, num_ports_in_stamp):
                         raise ValueError(f"Component '{comp_id}' stamp {stamp_idx} matrix shape {current_stamp_matrix.shape} mismatch with number of ports {num_ports_in_stamp}.")

                    for i in range(num_ports_in_stamp):
                        for j in range(num_ports_in_stamp):
                            global_row = global_indices[i]
                            global_col = global_indices[j]
                            stamp_val = current_stamp_matrix[i, j]
                            add_value_to_data(global_row, global_col, stamp_val, comp_id)
                    logger.debug(f"Stamped '{comp_id}' contribution {stamp_idx} for ports {port_ids} -> nodes {global_indices}")
            
            except MnaInputError: raise # Propagate sparsity errors immediately
            except ComponentError as e:
                logger.error(f"Component error during stamp processing for '{comp_id}' at {freq_str_log}: {e}", exc_info=False) # Already logged by component usually
                raise ComponentError(f"Failed processing component '{comp_id}' stamps at {freq_str_log}: {e}") from e # Re-raise with context
            except Exception as e:
                logger.error(f"Unexpected failure processing stamps for component '{comp_id}' at {freq_str_log}: {e}", exc_info=True)
                raise ComponentError(f"Unexpected failure processing component '{comp_id}' stamps at {freq_str_log}: {e}") from e

        # --- Create Final Sparse Matrix ---
        try:
            Yn_full_coo = sp.coo_matrix(
                (mna_data, (self._cached_rows, self._cached_cols)),
                shape=self._shape_full,
                dtype=np.complex128
            )
            Yn_full_csc = Yn_full_coo.tocsc()
            Yn_full_csc.eliminate_zeros()
            logger.debug(f"Full MNA matrix assembly complete for {freq_str_log}. Shape: {Yn_full_csc.shape}, NNZ: {Yn_full_csc.nnz}")
            return Yn_full_csc
        except Exception as e:
            logger.error(f"Failed to create final sparse matrix at {freq_str_log}: {e}", exc_info=True)
            raise RuntimeError(f"Failed creating sparse MNA matrix at {freq_str_log}: {e}") from e