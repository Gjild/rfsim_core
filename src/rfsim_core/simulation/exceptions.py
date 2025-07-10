# src/rfsim_core/simulation/exceptions.py
"""
Defines custom, diagnosable exceptions specific to the simulation execution phase.

This module provides a hierarchy of exception classes that represent known, non-fatal
failure modes that can occur *after* a circuit has been successfully built and is
undergoing simulation (e.g., numerical issues, invalid component behavior at a specific
frequency).

**Architectural Revision:**
In accordance with the project's definitive architecture, all exceptions in this module
inherit from the `DiagnosableError` base class. This provides three key benefits:
1.  **Catchability:** They are concrete classes that can be caught explicitly
    in `try...except` blocks (e.g., `except MnaInputError:`).
2.  **Contract Enforcement:** They are guaranteed by their base class to implement
    the `get_diagnostic_report()` method, fulfilling the `Diagnosable` protocol.
3.  **Hierarchy:** They are all catchable under the common `DiagnosableError` type,
    allowing for clean, layered error handling.
"""
import numpy as np
from typing import Optional, Any
from dataclasses import dataclass
from pathlib import Path

# Import the new, concrete base class and the report formatter.
from ..errors import DiagnosableError, format_diagnostic_report


@dataclass()
class MnaInputError(DiagnosableError):
    """
    Raised for logical or structural errors encountered during the setup of the
    MNA system, before any matrix assembly.
    """
    hierarchical_context: str
    details: str

    def get_diagnostic_report(self) -> str:
        """Generates the diagnostic report for an MNA input error."""
        return format_diagnostic_report(
            error_type="MNA Input Error",
            details=self.details,
            suggestion="This is often caused by a circuit topology that violates MNA rules, such as a floating port with no path to ground. Review the connectivity of the specified circuit context.",
            context={'fqn': self.hierarchical_context}
        )


@dataclass()
class SingularMatrixError(DiagnosableError, np.linalg.LinAlgError):
    """
    Raised when a matrix is found to be singular during LU factorization.

    This class uses multiple inheritance to be catchable both as our custom
    `DiagnosableError` and as a standard `LinAlgError`, providing flexibility.
    """
    details: str
    frequency: Optional[float] = None

    def __str__(self):
        freq_str = f" at frequency {self.frequency:.4e} Hz" if self.frequency is not None else ""
        return f"Singular matrix detected{freq_str}: {self.details}"

    def get_diagnostic_report(self) -> str:
        """Generates the diagnostic report for a singular matrix error."""
        return format_diagnostic_report(
            error_type="Singular Matrix Encountered",
            details=self.details,
            suggestion="This is often caused by a floating sub-circuit, a loop of ideal voltage sources, or a loop of ideal inductors. Check your circuit topology and component values, especially for ideal (zero/infinite) cases at the specified frequency.",
            context={'frequency': f"{self.frequency:.4e} Hz" if self.frequency is not None else "N/A"}
        )


@dataclass()
class SingleLevelSimulationFailure(DiagnosableError):
    """
    An internal, context-enriching exception that wraps a low-level diagnosable
    error with the high-level hierarchical context in which it occurred.

    This is a key part of the "Actionable Diagnostics" mandate for hierarchical
    designs. It allows the simulation engine to bubble up a failure from a deeply
    nested subcircuit while adding the necessary context at each step.
    """
    circuit_fqn: str
    circuit_source_path: Path
    original_error: DiagnosableError

    def get_diagnostic_report(self) -> str:
        """
        Generates a composite report that prepends the hierarchical context
        to the original root cause report.
        """
        # Get the full, formatted report from the original error.
        root_cause_report = self.original_error.get_diagnostic_report()

        # Create a new, enriched report that clearly shows the hierarchy.
        enriched_report = format_diagnostic_report(
            error_type="Simulation Failure in Hierarchical Context",
            details=(
                f"An error occurred while simulating the sub-circuit '{self.circuit_fqn}'.\n"
                f"This sub-circuit is defined in: {self.circuit_source_path}\n\n"
                f"--- Details of the Root Cause ---\n{root_cause_report}"
            ),
            suggestion="Address the root cause error detailed above. The error originates within the specified sub-circuit definition.",
            # The primary context is the location of the failure.
            context={'fqn': self.circuit_fqn, 'source_file': self.circuit_source_path}
        )
        return enriched_report