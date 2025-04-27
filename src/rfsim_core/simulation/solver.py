# --- src/rfsim_core/simulation/solver.py ---
import logging
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from typing import Optional

logger = logging.getLogger(__name__)

class SingularMatrixError(np.linalg.LinAlgError):
    """Custom exception for singular matrix during factorization or solve."""
    pass

def factorize_mna_matrix(Yn_reduced: sp.csc_matrix) -> splinalg.SuperLU:
    """
    Factorizes the reduced MNA matrix using sparse LU decomposition.

    Args:
        Yn_reduced: The (N-1) x (N-1) reduced MNA matrix (CSC format recommended).

    Returns:
        The LU factorization object (splinalg.SuperLU).

    Raises:
        SingularMatrixError: If the matrix is singular during factorization.
        TypeError: If input is not a sparse matrix.
    """
    if not sp.isspmatrix(Yn_reduced):
        raise TypeError("Input Yn_reduced must be a SciPy sparse matrix.")
    if Yn_reduced.shape[0] != Yn_reduced.shape[1]:
        raise ValueError("Reduced MNA matrix must be square.")

    logger.debug(f"Factorizing reduced MNA matrix ({Yn_reduced.shape})...")
    try:
        # Use splu (provides SuperLU object with solve method)
        # Ensure matrix is CSC format for splu efficiency
        lu = splinalg.splu(Yn_reduced.tocsc())

        # Basic singularity check provided by splu internals (e.g., zero pivot)
        # More advanced checks (condition number) belong in Phase 9
        # splu raises RuntimeError for singular matrix typically

        logger.debug("LU factorization successful.")
        return lu

    except RuntimeError as e:
        # splu often raises RuntimeError for singularity
        logger.error(f"LU factorization failed, matrix appears singular: {e}")
        raise SingularMatrixError(f"Reduced MNA matrix appears singular during LU factorization: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error during LU factorization: {e}", exc_info=True)
        raise e # Re-raise other unexpected errors


def solve_mna_system(lu_factorization: splinalg.SuperLU, I_reduced: np.ndarray) -> np.ndarray:
    """
    Solves the reduced MNA system Yn_reduced * V_reduced = I_reduced using a pre-computed LU factorization.

    Args:
        lu_factorization: The pre-computed sparse LU factorization object (from splu).
        I_reduced: The (N-1) x 1 reduced excitation vector.

    Returns:
        V_reduced: The (N-1) x 1 reduced node voltage solution vector.

    Raises:
        SingularMatrixError: If the solve fails (e.g., NaN/Inf results).
        TypeError: If input types are incorrect.
        ValueError: If shapes are inconsistent.
    """
    if not isinstance(lu_factorization, splinalg.SuperLU):
        raise TypeError("lu_factorization must be a SuperLU object from splinalg.splu.")
    if not isinstance(I_reduced, np.ndarray):
        raise TypeError("Excitation vector I_reduced must be a NumPy array.")

    expected_size = lu_factorization.shape[0] # N-1
    if I_reduced.shape != (expected_size,):
        raise ValueError(f"Shape mismatch: Reduced excitation vector I_reduced shape {I_reduced.shape} does not match LU factorization shape ({expected_size},).")

    logger.debug("Solving reduced MNA system using LU factorization...")
    try:
        V_reduced = lu_factorization.solve(I_reduced)

        # Check for NaN or Inf in solution, indicating potential numerical issues
        if np.any(np.isnan(V_reduced)) or np.any(np.isinf(V_reduced)):
            logger.error("NaN or Inf detected in MNA solution vector.")
            # This might indicate issues not caught by splu, treat as singular
            raise SingularMatrixError("MNA system solve resulted in NaN/Inf values.")

        logger.debug("Reduced MNA solve successful.")
        return V_reduced

    except (RuntimeError, ValueError) as e:
        # Catch potential errors during solve method
        logger.error(f"Sparse solve using LU factorization failed: {e}")
        # Treat solve failures as indicative of singularity/instability
        raise SingularMatrixError(f"MNA system solve failed using LU factorization: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error during sparse solve: {e}", exc_info=True)
        raise e