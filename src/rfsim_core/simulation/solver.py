# src/rfsim_core/simulation/solver.py
import logging
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from typing import Optional

from .exceptions import SingularMatrixError

logger = logging.getLogger(__name__)

def factorize_mna_matrix(Yn_reduced: sp.csc_matrix, frequency: float) -> splinalg.SuperLU:
    """
    Factorizes the reduced MNA matrix using sparse LU decomposition.

    Args:
        Yn_reduced: The (N-1) x (N-1) reduced MNA matrix (CSC format).
        frequency: The frequency at which the factorization is being performed, for context.

    Returns:
        The LU factorization object (splinalg.SuperLU).

    Raises:
        SingularMatrixError: A diagnosable error if the matrix is singular.
        TypeError: If input is not a sparse matrix.
    """
    if not sp.isspmatrix(Yn_reduced):
        raise TypeError("Input Yn_reduced must be a SciPy sparse matrix.")
    if Yn_reduced.shape[0] != Yn_reduced.shape[1]:
        raise ValueError("Reduced MNA matrix must be square.")

    logger.debug(f"Factorizing reduced MNA matrix ({Yn_reduced.shape}) at {frequency:.4e} Hz...")
    try:
        lu = splinalg.splu(Yn_reduced.tocsc())
        logger.debug("LU factorization successful.")
        return lu
    except RuntimeError as e:
        logger.error(f"LU factorization failed at {frequency:.4e} Hz, matrix appears singular: {e}")
        raise SingularMatrixError(details=str(e), frequency=frequency) from e
    except Exception as e:
        logger.error(f"Unexpected error during LU factorization at {frequency:.4e} Hz: {e}", exc_info=True)
        raise e

def solve_mna_system(lu_factorization: splinalg.SuperLU, I_reduced: np.ndarray) -> np.ndarray:
    """
    Solves the reduced MNA system using a pre-computed LU factorization.
    """
    # (Implementation of this function remains the same as proposed previously,
    # raising ValueError/TypeError for bad inputs.)
    if not isinstance(lu_factorization, splinalg.SuperLU):
        raise TypeError("lu_factorization must be a SuperLU object from splinalg.splu.")
    
    logger.debug("Solving reduced MNA system using LU factorization...")
    V_reduced = lu_factorization.solve(I_reduced)

    if np.any(np.isnan(V_reduced)) or np.any(np.isinf(V_reduced)):
        logger.error("NaN or Inf detected in MNA solution vector.")
        raise SingularMatrixError(details="MNA system solve resulted in NaN/Inf values.", frequency=None) # No freq context here
        
    return V_reduced