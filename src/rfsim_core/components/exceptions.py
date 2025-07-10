# src/rfsim_core/components/exceptions.py
"""
Defines the custom, diagnosable exceptions for the components subsystem.
This is the single source of truth for component-related, diagnosable errors.
"""
from dataclasses import dataclass
from typing import Optional

# Import the foundational contracts from the top-level errors module.
from ..errors import DiagnosableError, format_diagnostic_report

@dataclass()
class ComponentError(DiagnosableError):
    """
    The canonical, diagnosable exception for all component-related errors.

    Raised when a specific component fails during its simulation calculations,
    such as when provided with an invalid parameter value for a given frequency.
    This class is the single, unified type for all such errors.
    """
    component_fqn: str
    details: str
    frequency: Optional[float] = None

    def get_diagnostic_report(self) -> str:
        """Generates the definitive diagnostic report for a component simulation error."""
        return format_diagnostic_report(
            error_type="Component Simulation Error",
            details=self.details,
            suggestion="Check the component's parameters and ensure they are valid for the given frequency (e.g., non-negative resistance, real capacitance).",
            context={
                'fqn': self.component_fqn,
                'frequency': f"{self.frequency:.4e} Hz" if self.frequency is not None else "N/A (DC or constant analysis)"
            }
        )