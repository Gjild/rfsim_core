# src/rfsim_core/simulation/mna.py
import logging
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple

from ..data_structures import Circuit, Net
from ..components.base import ComponentBase, ComponentError
from ..units import ureg, Quantity, pint

logger = logging.getLogger(__name__)

# Define a large admittance value to represent ideal shorts numerically (for F>0)
# This should be configurable later.
LARGE_ADMITTANCE_SIEMENS = 1e12 # Siemens (1 / microOhm)

class MnaInputError(ValueError):
    """Error related to inputs for MNA assembly."""
    pass

class MnaAssembler:
    """
    Assembles the Modified Nodal Analysis (MNA) matrix (Yn) and
    current vector (I) for a circuit at a single frequency (F > 0).
    """
    def __init__(self, circuit: Circuit, freq_hz: float):
        if not isinstance(circuit, Circuit) or not hasattr(circuit, 'sim_components'):
             raise MnaInputError("Circuit object must be processed by CircuitBuilder first.")
        if not freq_hz > 0:
             # This assembler is for AC (F>0) analysis only. DC is separate.
             raise MnaInputError(f"MNA Assembler requires frequency > 0 Hz. Got {freq_hz} Hz. Use dedicated DC analysis for F=0.")

        self.circuit = circuit
        self.freq_hz = freq_hz
        self.ureg = ureg # Use shared registry

        # Node mapping and port info will be populated during assembly
        self.node_map: Dict[str, int] = {} # Net name -> index
        self.node_count: int = 0
        self.port_indices: List[int] = [] # Indices of external port nodes
        self.port_names: List[str] = [] # Names of external ports in order of indices
        self.port_ref_admittances: Dict[str, Quantity] = {} # Port name -> Y0 Quantity

        self._assign_node_indices()
        self._calculate_reference_admittances()

        # COO format lists for sparse matrix construction
        self._rows: List[int] = []
        self._cols: List[int] = []
        self._data: List[complex] = []

        logger.info(f"MNA Assembler initialized for '{circuit.name}' at {freq_hz} Hz.")
        logger.debug(f"Node count: {self.node_count}. Ground node: '{circuit.ground_net_name}' (index 0).")
        logger.debug(f"External ports: {self.port_names}")

    def _assign_node_indices(self):
        """Assigns unique integer indices to each net, ground is always 0."""
        ground_name = self.circuit.ground_net_name
        if ground_name not in self.circuit.nets:
             raise MnaInputError(f"Ground net '{ground_name}' not found in the circuit.")

        self.node_map = {}
        idx_counter = 0

        # Assign ground node index 0
        self.node_map[ground_name] = idx_counter
        idx_counter += 1

        # Assign indices to other nodes
        for net_name in sorted(self.circuit.nets.keys()): # Sort for consistent ordering
            if net_name != ground_name:
                self.node_map[net_name] = idx_counter
                idx_counter += 1

        self.node_count = idx_counter

        # Map external port names to their indices
        # Store in a predictable order (e.g., sorted by name)
        self.port_names = sorted(self.circuit.external_ports.keys())
        self.port_indices = [self.node_map[name] for name in self.port_names]


    def _calculate_reference_admittances(self):
        """Parses Z0 strings and calculates Y0 for external ports."""
        self.port_ref_admittances = {}
        for port_name, z0_str in self.circuit.external_port_impedances.items():
            try:
                z0 = self.ureg.Quantity(z0_str)
                if not z0.is_compatible_with("ohm"):
                    raise pint.DimensionalityError(z0.units, self.ureg.ohm, z0.dimensionality, self.ureg.ohm.dimensionality)
                if z0.magnitude == 0:
                    # Zero impedance port - infinite admittance
                    y0 = Quantity(np.inf + 0j, ureg.siemens)
                    logger.warning(f"External port '{port_name}' has zero reference impedance. Using infinite admittance.")
                else:
                    y0 = (1.0 / z0).to(ureg.siemens)

                self.port_ref_admittances[port_name] = y0
                logger.debug(f"Port '{port_name}': Z0={z0:~P}, Y0={y0:~P}")
            except pint.DimensionalityError as e:
                raise MnaInputError(f"Invalid reference impedance unit for port '{port_name}': '{z0_str}'. Expected ohms. Error: {e}") from e
            except Exception as e:
                raise MnaInputError(f"Cannot parse reference impedance for port '{port_name}': '{z0_str}'. Error: {e}") from e

    def _add_stamp(self, r: int, c: int, val: complex):
        """Appends non-zero values to COO lists."""
        # Important: Do not add explicit zeros to sparse matrix data
        if abs(val) > 1e-18: # Threshold to avoid numerical noise as zero
            self._rows.append(r)
            self._cols.append(c)
            self._data.append(val)

    def assemble(self) -> sp.csc_matrix:
        """Performs the MNA matrix assembly."""
        logger.info(f"Assembling MNA matrix ({self.node_count}x{self.node_count}) at {self.freq_hz} Hz...")

        # Reset COO lists
        self._rows, self._cols, self._data = [], [], []

        # Create a 1-element NumPy array for the current frequency
        current_freq_array = np.array([self.freq_hz])

        # 1. Stamp Simulation Components
        sim_components: Dict[str, ComponentBase] = self.circuit.sim_components
        for comp_id, comp in sim_components.items():
            try:
                # Get admittance (complex Quantity in Siemens)
                admittance_qty = comp.get_admittance(current_freq_array)[0] # Only one element in ndarray

                # --- Enforce Contract ---
                if not isinstance(admittance_qty, Quantity):
                     raise TypeError(f"Component '{comp_id}' get_admittance did not return a Pint Quantity.")
                if not admittance_qty.is_compatible_with("siemens"):
                     raise pint.DimensionalityError(admittance_qty.units, ureg.siemens, admittance_qty.dimensionality, ureg.siemens.dimensionality,
                                                   f"Component '{comp_id}' get_admittance returned quantity with wrong dimension.")

                # Extract complex value in Siemens
                y = admittance_qty.to(ureg.siemens).magnitude

                # Handle numerical infinity (replace with large number)
                if np.isinf(y):
                     y_stamp = np.copysign(LARGE_ADMITTANCE_SIEMENS, y.real) + 0j # Use large real value
                     logger.warning(f"Component '{comp_id}' returned infinite admittance. Stamping with {y_stamp:.2e} S.")
                elif np.isnan(y):
                     logger.error(f"Component '{comp_id}' returned NaN admittance. Stamping with 0.")
                     y_stamp = 0.0 + 0.0j
                else:
                     y_stamp = y # Should already be complex

                # Get node indices - Requires access to port->net mapping
                # Assuming a simple 2-terminal component model for RLC now
                comp_data = self.circuit.components[comp_id] # Get original data for ports
                if len(comp_data.ports) != 2:
                     logger.warning(f"Component '{comp_id}' is not 2-terminal. Skipping basic admittance stamping.")
                     continue # Simple model only handles 2 terminals for now

                port_ids = list(comp_data.ports.keys())
                port1_net_name = comp_data.ports[port_ids[0]].net.name
                port2_net_name = comp_data.ports[port_ids[1]].net.name
                n1 = self.node_map[port1_net_name]
                n2 = self.node_map[port2_net_name]

                logger.debug(f"Stamping '{comp_id}': Y={y_stamp} S between nodes {n1} ('{port1_net_name}') and {n2} ('{port2_net_name}')")

                # Stamp Yn matrix (Nodal Admittance part)
                self._add_stamp(n1, n1, y_stamp)
                self._add_stamp(n1, n2, -y_stamp)
                self._add_stamp(n2, n1, -y_stamp)
                self._add_stamp(n2, n2, y_stamp)

            except Exception as e:
                logger.error(f"Failed to process or stamp component '{comp_id}': {e}", exc_info=True)
                # Decide whether to raise, or just log and continue
                raise ComponentError(f"Failed to process or stamp component '{comp_id}': {e}") from e


        # 2. Stamp External Port Reference Admittances (to ground)
        for port_name, y0_qty in self.port_ref_admittances.items():
            if port_name in self.node_map:
                net_obj = self.circuit.nets[port_name]
                # Skip ports that have no real connections
                if len(net_obj.connected_components) == 0:
                    logger.warning(
                        "External port '%s' is floating (no connected components); "
                        "skipping reference-impedance stamp.", port_name)
                    continue
                node_index_port = self.node_map[port_name] # Node index of the port
                y0 = y0_qty.to(ureg.siemens).magnitude # Complex value

                if np.isinf(y0):
                    y0_stamp = np.copysign(LARGE_ADMITTANCE_SIEMENS, y0.real) + 0j
                    logger.warning(f"Port '{port_name}' has infinite reference admittance. Stamping {y0_stamp:.2e} S to ground.")
                elif np.isnan(y0):
                    logger.error(f"Port '{port_name}' has NaN reference admittance. Stamping with 0.")
                    y0_stamp = 0.0 + 0.0j
                else:
                    y0_stamp = y0

                logger.debug(f"Stamping port '{port_name}' ref admittance: Y0={y0_stamp:.3e} S at node {node_index_port}")
                # Stamp admittance to ground (node 0)
                self._add_stamp(node_index_port, node_index_port, y0_stamp)
            else:
                # Should not happen if parsing was correct
                 logger.error(f"External port '{port_name}' refers to an unknown net.")


        # 3. Create Sparse Matrix
        try:
            Yn_full = sp.csc_matrix(
                (self._data, (self._rows, self._cols)),
                shape=(self.node_count, self.node_count),
                dtype=np.complex128
            )
            Yn_full.eliminate_zeros() # Clean up explicit zeros if any were added
            logger.info(f"MNA matrix assembly complete. Shape: {Yn_full.shape}, NNZ: {Yn_full.nnz}")
            return Yn_full
        except Exception as e:
            logger.error(f"Failed to create sparse matrix: {e}", exc_info=True)
            raise RuntimeError(f"Failed to create sparse MNA matrix: {e}") from e