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
        if not isinstance(circuit, Circuit) or not hasattr(circuit, 'sim_components'):
             raise MnaInputError("Circuit object must be processed by CircuitBuilder first.")
        if not circuit.nets:
             raise MnaInputError("Circuit contains no nets.")

        self.circuit = circuit
        self.ureg = ureg
        self.sim_components: Dict[str, ComponentBase] = circuit.sim_components
        # Need access to the raw component data for port->net mapping during stamping
        self.raw_components: Dict[str, ComponentData] = circuit.components

        # --- Frequency Independent Setup (Node maps, port info) ---
        self.node_map: Dict[str, int] = {}
        self.node_count: int = 0
        self.port_indices: List[int] = []
        self.port_names: List[str] = []
        self.port_ref_impedances: Dict[str, Quantity] = {}
    
        # --- Indices relative to REDUCED system (ground removed) ---
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
            self._compute_sparsity_pattern()
        except Exception as e:
            logger.error(f"Error during MNA Assembler initialization: {e}", exc_info=True)
            if isinstance(e, (pint.DimensionalityError, ValueError, MnaInputError)):
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

        # Iterate through simulation-ready components to get their type and connectivity
        for comp_id, sim_comp in self.sim_components.items():
            comp_data = self.raw_components[comp_id] # Get raw data for port->net map
            ComponentClass = type(sim_comp)

            try:
                # Get declared connectivity for this component *type*
                connectivity = ComponentClass.declare_connectivity()
                declared_ports = set(ComponentClass.declare_ports()) # For validation

                logger.debug(f"Component '{comp_id}' (type {ComponentClass.component_type_str}) declared connectivity: {connectivity}")

                # For each connected pair declared by the type...
                for port_id1, port_id2 in connectivity:
                    # Check if these ports actually exist in the instance connection
                    if port_id1 not in comp_data.ports or port_id2 not in comp_data.ports:
                        # This might happen if declare_connectivity is inconsistent with declare_ports
                        # or if user didn't connect all ports needed for internal paths.
                         logger.warning(f"Sparsity calculation for '{comp_id}': Connectivity declared between ports '{port_id1}' and '{port_id2}', but one or both are not connected in the instance. Skipping this pair for sparsity.")
                         continue
                    if port_id1 not in declared_ports or port_id2 not in declared_ports:
                         # Should be caught by builder, but check defensively
                         logger.error(f"Internal Error: Sparsity calculation for '{comp_id}': Connectivity uses ports '{port_id1}' or '{port_id2}' which are not in declared_ports {declared_ports}. Fix component definition.")
                         continue

                    # Get the nets these ports are connected to
                    net1_name = comp_data.ports[port_id1].net.name
                    net2_name = comp_data.ports[port_id2].net.name

                    # Get the global node indices for these nets
                    n1 = self.node_map[net1_name]
                    n2 = self.node_map[net2_name]

                    # Add the four non-zero entries for this connection to the COO pattern
                    # (n1, n1), (n1, n2), (n2, n1), (n2, n2)
                    # Use a dummy value (1.0) for pattern calculation
                    self._add_to_coo_pattern(rows, cols, data, n1, n1, 1.0)
                    self._add_to_coo_pattern(rows, cols, data, n1, n2, 1.0) # Sign doesn't matter for pattern
                    self._add_to_coo_pattern(rows, cols, data, n2, n1, 1.0) # Sign doesn't matter for pattern
                    self._add_to_coo_pattern(rows, cols, data, n2, n2, 1.0)

            except KeyError as e:
                raise MnaInputError(f"Error getting node index for component '{comp_id}' during pattern build: Net '{e}' not in map.")
            except AttributeError as e:
                raise MnaInputError(f"Error accessing connection for component '{comp_id}' during pattern build: {e}.")
            except Exception as e:
                logger.error(f"Unexpected error getting connectivity for component '{comp_id}' type '{ComponentClass.__name__}': {e}", exc_info=True)
                raise MnaInputError(f"Failed to get connectivity for component '{comp_id}': {e}") from e

        # Create sparse matrix from COO lists to consolidate duplicates & get final pattern
        try:
            if not rows: # Handle empty circuit or circuit with no connected components
                logger.warning("No component connectivity found to build sparsity pattern. Matrix will be empty.")
                temp_coo = sp.coo_matrix(self._shape_full, dtype=np.complex128)
            else:
                temp_coo = sp.coo_matrix((data, (rows, cols)), shape=self._shape_full, dtype=np.complex128)

            temp_csc = temp_coo.tocsc()
            temp_csc.eliminate_zeros() # Good practice
            final_coo = temp_csc.tocoo() # Get final pattern indices after consolidation

            self._cached_rows = final_coo.row
            self._cached_cols = final_coo.col
            self._sparsity_nnz = final_coo.nnz
            logger.debug(f"Sparsity pattern computed: NNZ={self._sparsity_nnz}")

        except Exception as e:
            logger.error(f"Failed to create sparse matrix for sparsity pattern: {e}", exc_info=True)
            raise RuntimeError(f"Failed to compute MNA sparsity pattern: {e}") from e

    def _add_to_coo_pattern(self, rows: list, cols: list, data: list, r: int, c: int, val: complex):
        """ Helper for adding to COO lists during pattern generation. """
        # (Logic remains the same - bounds check)
        if not (0 <= r < self.node_count and 0 <= c < self.node_count):
            logger.error(f"Internal Error (Pattern): Stamp indices ({r}, {c}) out of bounds ({self.node_count}).")
            return
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
        if not freq_hz > 0:
             raise MnaInputError(f"MNA Assembly requires frequency > 0 Hz. Got {freq_hz} Hz.")
        if self._cached_rows is None or self._cached_cols is None:
             raise RuntimeError("Sparsity pattern was not computed or is invalid.")

        logger.debug(f"Assembling MNA matrix for {freq_hz:.4e} Hz using cached pattern (NNZ={self._sparsity_nnz})...")

        # Initialize data array for the current frequency based on cached pattern
        mna_data = np.zeros(self._sparsity_nnz, dtype=np.complex128)
        # Create mapping from (row, col) in pattern to index in mna_data
        rc_to_data_idx: Dict[Tuple[int, int], int] = {
            (r, c): i for i, (r, c) in enumerate(zip(self._cached_rows, self._cached_cols))
        }

        # Helper to add value to the correct index in mna_data
        def add_value_to_data(r: int, c: int, value: complex):
            """Adds a value to the MNA data array using the precomputed map."""
            key = (r, c)
            if key in rc_to_data_idx:
                mna_data[rc_to_data_idx[key]] += value
            else:
                # This *shouldn't* happen if sparsity pattern is correct,
                # but log if it does. It means a component is trying to stamp
                # a location that wasn't predicted by declare_connectivity.
                if abs(value) > 1e-15: # Log only if value is significant
                    logger.warning(f"Attempted to stamp non-zero value ({value:.2e}) at ({r}, {c}) which is not in the cached sparsity pattern. Check component's declare_connectivity(). Ignoring stamp.")


        # --- Stamp Simulation Components using Generalized Method ---
        current_freq_array = np.array([freq_hz]) # Pass frequency as array
        for comp_id, sim_comp in self.sim_components.items():
            comp_data = self.raw_components[comp_id] # Raw data for port->net lookup
            try:
                # Get list of stamp contributions from the component
                stamp_infos = sim_comp.get_mna_stamps(current_freq_array)

                # Process each stamp contribution
                for stamp_idx, stamp_info in enumerate(stamp_infos):
                     admittance_matrix_qty, port_ids = stamp_info

                     # --- Enforce Contract ---
                     if not isinstance(admittance_matrix_qty, Quantity):
                         raise TypeError(f"Comp '{comp_id}' stamp {stamp_idx} matrix != Quantity")
                     if not admittance_matrix_qty.check('siemens'):
                         raise pint.DimensionalityError(admittance_matrix_qty.units, ureg.siemens, msg=f"Comp '{comp_id}' stamp {stamp_idx} matrix")

                     # Extract numerical matrix/array (in Siemens)
                     # Need to handle scalar freq vs array freq output from component
                     stamp_values = admittance_matrix_qty.to(ureg.siemens).magnitude

                     # If component handled freq array, select the data for the current freq
                     if stamp_values.ndim == 3:
                         # Handles (num_freqs, N, N), including the case (1, N, N)
                         if stamp_values.shape[0] == 1:
                             # Correctly extract the 2D slice for the single frequency case
                             current_stamp_matrix = stamp_values[0, :, :]
                             logger.debug(f"Extracted 2D stamp from 3D shape {stamp_values.shape} for single frequency.")
                         else:
                             # This case shouldn't be hit in the current single-frequency loop,
                             # but handle defensively if component somehow got multi-freq array.
                             logger.warning(f"Component '{comp_id}' returned unexpected multi-frequency shape {stamp_values.shape} within single-frequency assembly loop. Using first slice.")
                             current_stamp_matrix = stamp_values[0, :, :]

                     elif stamp_values.ndim == 2:
                         # Handles (N, N) case (e.g., scalar frequency input or freq-independent component)
                         current_stamp_matrix = stamp_values
                         logger.debug(f"Using 2D stamp directly shape={stamp_values.shape}.")

                     else:
                         # Handles truly unexpected shapes (1D, 4D+, etc.)
                         raise ValueError(f"Component '{comp_id}' returned stamp with unexpected dimension/shape {stamp_values.shape} (expected 2D or 3D).")


                     # Get global node indices corresponding to the port_ids for this instance
                     global_indices = []
                     valid_ports = True
                     for port_id in port_ids:
                         if port_id not in comp_data.ports:
                             logger.error(f"Component '{comp_id}' stamp {stamp_idx} refers to port ID '{port_id}' which is not connected in the netlist instance. Skipping stamp.")
                             valid_ports = False
                             break
                         net_name = comp_data.ports[port_id].net.name
                         global_indices.append(self.node_map[net_name])
                     if not valid_ports: continue # Skip this stamp

                     # Check dimensions match
                     num_ports = len(port_ids)
                     if current_stamp_matrix.shape != (num_ports, num_ports):
                         raise ValueError(f"Component '{comp_id}' stamp {stamp_idx} matrix shape {current_stamp_matrix.shape} mismatch with number of ports {num_ports}.")

                     # Add the elements of the component's stamp matrix to the global MNA data
                     for i in range(num_ports): # Local row index in stamp_matrix
                         for j in range(num_ports): # Local col index in stamp_matrix
                             global_row = global_indices[i]
                             global_col = global_indices[j]
                             stamp_val = current_stamp_matrix[i, j]

                             # Add using the helper (handles check against sparsity pattern)
                             add_value_to_data(global_row, global_col, stamp_val)

                     logger.debug(f"Stamped '{comp_id}' contribution {stamp_idx} for ports {port_ids} -> nodes {global_indices}")

            except Exception as e:
                logger.error(f"Failed to get/process stamps for component '{comp_id}' at {freq_hz} Hz: {e}", exc_info=True)
                # Decide whether to raise, or just log and continue (making the matrix potentially wrong)
                # Raising is safer to indicate simulation failure.
                raise ComponentError(f"Failed processing component '{comp_id}' stamps at {freq_hz} Hz: {e}") from e

        # --- Create Final Sparse Matrix ---
        try:
            # Use the cached pattern indices and the just-computed data
            Yn_full_coo = sp.coo_matrix(
                (mna_data, (self._cached_rows, self._cached_cols)),
                shape=self._shape_full,
                dtype=np.complex128
            )
            # Convert to CSC for efficient solvers later, eliminate explicit zeros
            Yn_full_csc = Yn_full_coo.tocsc()
            Yn_full_csc.eliminate_zeros()

            logger.debug(f"Full MNA matrix assembly complete for {freq_hz:.4e} Hz. Shape: {Yn_full_csc.shape}, NNZ: {Yn_full_csc.nnz}")
            return Yn_full_csc
        except Exception as e:
            logger.error(f"Failed to create final sparse matrix at {freq_hz} Hz: {e}", exc_info=True)
            raise RuntimeError(f"Failed creating sparse MNA matrix at {freq_hz} Hz: {e}") from e