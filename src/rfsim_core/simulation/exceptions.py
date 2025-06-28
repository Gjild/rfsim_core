# src/rfsim_core/simulation/exceptions.py
import numpy as np
from typing import Optional, Any
from dataclasses import dataclass
from pathlib import Path

from ..errors import Diagnosable, format_diagnostic_report

@dataclass(frozen=True)
class MnaInputError(ValueError, Diagnosable):
    """Error related to inputs for MNA assembly."""
    hierarchical_context: str
    details: str

    def get_diagnostic_report(self) -> str:
        return format_diagnostic_report(
            error_type="MNA Input Error",
            details=self.details,
            suggestion="This is often an internal error or a circuit topology that violates MNA rules (e.g., floating ports).",
            context={'fqn': self.hierarchical_context}
        )

@dataclass(frozen=True)
class ComponentError(ValueError, Diagnosable):
    """Custom exception for component-related errors during simulation."""
    component_fqn: str
    details: str
    frequency: Optional[float] = None

    def get_diagnostic_report(self) -> str:
        return format_diagnostic_report(
            error_type="Component Simulation Error",
            details=self.details,
            suggestion="Check the component's parameters and ensure they are valid for the given frequency.",
            context={'fqn': self.component_fqn, 'frequency': f"{self.frequency:.4e} Hz" if self.frequency is not None else "N/A"}
        )

@dataclass(frozen=True)
class DCAnalysisError(ValueError, Diagnosable):
    """Custom exception for errors during rigorous DC analysis."""
    hierarchical_context: str
    details: str
    
    def get_diagnostic_report(self) -> str:
        return format_diagnostic_report(
            error_type="DC Analysis Error",
            details=self.details,
            suggestion="Review the circuit for issues with DC shorts (R=0, L=0) or opens (C=inf) that might create an invalid topology at F=0.",
            context={'fqn': self.hierarchical_context}
        )

@dataclass(frozen=True)
class SingularMatrixError(np.linalg.LinAlgError, Diagnosable):
    """Custom exception for singular matrix during factorization or solve."""
    details: str
    frequency: Optional[float] = None

    def __str__(self):
        freq_str = f" at frequency {self.frequency:.4e} Hz" if self.frequency is not None else ""
        return f"Singular matrix detected{freq_str}: {self.details}"

    def get_diagnostic_report(self) -> str:
        return format_diagnostic_report(
            error_type="Singular Matrix Encountered",
            details=self.details,
            suggestion="This is often caused by a floating sub-circuit or a shorted loop of ideal components at a specific frequency. Check your circuit topology and component values.",
            context={'frequency': f"{self.frequency:.4e} Hz" if self.frequency else "N/A"}
        )

@dataclass(frozen=True)
class SingleLevelSimulationFailure(Exception, Diagnosable):
    """
    An internal, context-enriching exception that wraps a low-level diagnosable
    error with the high-level hierarchical context in which it occurred.
    """
    circuit_fqn: str
    circuit_source_path: Path
    original_error: Diagnosable

    def get_diagnostic_report(self) -> str:
        # Get the report from the original, low-level error
        original_report = self.original_error.get_diagnostic_report()
        
        # Prepend the hierarchical context to the report
        header = [
            f"Error occurred within circuit context: '{self.circuit_fqn}'",
            f"Defined in file: '{self.circuit_source_path}'",
        ]
        
        # We slice the original report to remove its header and combine it with our new, richer one.
        original_lines = original_report.strip().splitlines()
        report_body = "\n".join(original_lines[2:]) # Skip the "===", "Error Type:"
        
        # Re-assemble the full report with the new, richer header
        full_report_lines = [
            "\n",
            "================ RFSim Core: Actionable Diagnostic Report ================",
        ] + header + [report_body] + [
            "========================================================================"
        ]
        
        return "\n".join(full_report_lines)