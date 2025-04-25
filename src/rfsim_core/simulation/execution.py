# src/rfsim_core/simulation/execution.py
import logging
import numpy as np
import scipy.sparse as sp

from .mna import MnaAssembler, MnaInputError
from .solver import solve_mna, SingularMatrixError
from ..data_structures import Circuit
from .. import ureg, Quantity, ComponentError # Access units if needed

logger = logging.getLogger(__name__)

class SimulationError(Exception):
    """General error during simulation execution."""
    pass

def run_simulation(circuit: Circuit, freq_hz: float) -> np.ndarray:
    """
    Runs a single-frequency simulation and calculates the N-port Y-matrix.

    Args:
        circuit: The Circuit object, processed by CircuitBuilder.
        freq_hz: The simulation frequency in Hz (must be > 0).

    Returns:
        y_matrix: The N-port Y-matrix as a complex NumPy array (NumPorts x NumPorts).

    Raises:
        SimulationError: If simulation fails.
        MnaInputError: If frequency is invalid or circuit not ready.
        SingularMatrixError: If the MNA matrix is singular.
    """
    logger.info(f"--- Starting simulation for '{circuit.name}' at {freq_hz} Hz ---")

    try:
        # 1. Assemble MNA matrix (once for the frequency)
        assembler = MnaAssembler(circuit, freq_hz)
        Yn_full = assembler.assemble()

        num_ports = len(assembler.port_indices)
        if num_ports == 0:
            logger.warning("Circuit has no external ports. Returning empty Y-matrix.")
            return np.array([], dtype=np.complex128).reshape(0, 0)

        logger.info(f"Calculating {num_ports}-port Z-matrix...")
        z_matrix = np.zeros((num_ports, num_ports), dtype=np.complex128)
        node_count = assembler.node_count
        port_node_indices = assembler.port_indices

        # 2. Calculate Z-matrix column by column
        for k in range(num_ports): # k is the index of the port being excited
            port_k_node_idx = port_node_indices[k]
            port_k_name = assembler.port_names[k]
            logger.debug(f"  Exciting port {k+1}/{num_ports} ('{port_k_name}', node {port_k_node_idx}) with 1A.")

            # Create excitation vector I: 1A at port k, 0 otherwise
            I_full = np.zeros(node_count, dtype=np.complex128)
            I_full[port_k_node_idx] = 1.0 + 0.0j

            # Solve MNA system: Yn * V = I
            V_full = solve_mna(Yn_full, I_full)

            # Extract voltages at all port nodes for this excitation
            V_ports_k = V_full[port_node_indices]

            # This V_ports_k vector is the k-th column of the Z-matrix
            z_matrix[:, k] = V_ports_k
            logger.debug(f"  Resulting port voltages (Z-matrix col {k}): {V_ports_k}")

        # 3. Calculate Y-matrix from Z-matrix
        logger.info("Inverting Z-matrix to obtain Y-matrix...")
        try:
            # Use pseudo-inverse for potentially better stability if Z is near singular,
            # though a truly singular Z implies circuit issues. np.linalg.inv is standard.
            y_matrix = np.linalg.inv(z_matrix)
            logger.info("Y-matrix calculation successful.")
        except np.linalg.LinAlgError as e:
             logger.error(f"Failed to invert Z-matrix: {e}. The Z-matrix might be singular, indicating a potential circuit problem (e.g., unconnected ports).")
             raise SimulationError(f"Failed to invert Z-matrix: {e}") from e

        logger.info(f"--- Simulation finished for {freq_hz} Hz ---")
        return y_matrix

    except (MnaInputError, SingularMatrixError, ComponentError) as e:
         logger.error(f"Simulation failed at {freq_hz} Hz: {e}", exc_info=True)
         raise # Re-raise specific simulation errors
    except Exception as e:
         logger.error(f"Unexpected error during simulation at {freq_hz} Hz: {e}", exc_info=True)
         raise SimulationError(f"Unexpected simulation error at {freq_hz} Hz: {e}") from e