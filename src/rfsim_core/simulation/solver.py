# src/rfsim_core/simulation/solver.py
import logging
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

logger = logging.getLogger(__name__)

class SingularMatrixError(np.linalg.LinAlgError):
    """Custom exception for singular matrix during solve."""
    pass

def solve_mna(Yn_full: sp.csc_matrix, I_full: np.ndarray) -> np.ndarray:
    """
    Solves the MNA system Yn * V = I, handling the ground node constraint.

    Args:
        Yn_full: The complete N x N MNA matrix (including ground node 0).
        I_full: The complete N x 1 excitation vector (including ground node 0).

    Returns:
        V_full: The N x 1 node voltage solution vector (including V[0]=0).

    Raises:
        SingularMatrixError: If the matrix is singular after reduction.
        TypeError: If input types are incorrect.
        ValueError: If shapes are inconsistent.
    """
    if not sp.isspmatrix_csc(Yn_full) and not sp.isspmatrix_csr(Yn_full):
        raise TypeError("MNA matrix Yn_full must be a SciPy CSC or CSR sparse matrix.")
    if not isinstance(I_full, np.ndarray):
        raise TypeError("Excitation vector I_full must be a NumPy array.")
    if Yn_full.shape[0] != Yn_full.shape[1]:
        raise ValueError("MNA matrix Yn_full must be square.")
    if Yn_full.shape[0] != I_full.shape[0]:
        raise ValueError(f"Shape mismatch: Yn_full {Yn_full.shape} and I_full {I_full.shape}.")
    if Yn_full.shape[0] < 2:
        raise ValueError("MNA system must have at least 2 nodes (including ground).")

    N = Yn_full.shape[0]
    logger.debug(f"Solving MNA system of size {N}x{N}.")

    # --- Reduced System Approach ---
    # Create the reduced matrix by removing row 0 and column 0 (ground node)
    # Note: Slicing sparse matrices creates views or copies depending on format/version.
    # It's generally efficient.
    Yn_reduced = Yn_full[1:, 1:]
    I_reduced = I_full[1:]

    # Adjust RHS for terms connected to ground from non-ground nodes
    # I_reduced_adj = I_reduced - Yn_full[1:, 0] * V0 (where V0 = 0)
    # Since V0 is 0, no adjustment needed for RHS in this standard formulation.
    logger.debug(f"Reduced system size: {Yn_reduced.shape}")

    # Solve the reduced system: Yn_reduced * V_reduced = I_reduced
    try:
        # Use sparse LU factorization and solve (generally robust)
        lu = splinalg.splu(Yn_reduced.tocsc()) # splu prefers CSC
        V_reduced = lu.solve(I_reduced)

        # Check for NaN or Inf in solution, which might indicate near singularity not caught by splu
        if np.any(np.isnan(V_reduced)) or np.any(np.isinf(V_reduced)):
            logger.error("NaN or Inf detected in MNA solution vector.")
            raise SingularMatrixError("MNA system appears singular (NaN/Inf in solution).")

    except (RuntimeError, np.linalg.LinAlgError) as e:
        # Catch potential errors from splu (e.g., singular matrix)
        logger.error(f"Sparse solve failed: {e}", exc_info=True)
        # Perform singularity check here (Phase 9) or raise specific error
        raise SingularMatrixError(f"MNA system appears singular: {e}") from e
    except Exception as e:
        # Catch unexpected errors
        logger.error(f"Unexpected error during sparse solve: {e}", exc_info=True)
        raise e # Re-raise

    # Reconstruct the full solution vector V_full
    V_full = np.zeros(N, dtype=np.complex128)
    V_full[1:] = V_reduced
    # V_full[0] is already 0.0 + 0.0j

    logger.debug("MNA solve successful.")
    return V_full