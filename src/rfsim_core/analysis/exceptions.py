# src/rfsim_core/analysis/exceptions.py
"""
Defines custom, diagnosable exceptions for the analysis services.
"""
from dataclasses import dataclass
from ..errors import Diagnosable, format_diagnostic_report


@dataclass()
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


@dataclass()
class TopologyAnalysisError(ValueError, Diagnosable):
    """Custom exception for errors during topological analysis."""
    hierarchical_context: str
    details: str

    def get_diagnostic_report(self) -> str:
        return format_diagnostic_report(
            error_type="Topological Analysis Error",
            details=self.details,
            suggestion="This may indicate an internal error or a fundamental problem with the circuit's connectivity graph.",
            context={'fqn': self.hierarchical_context}
        )