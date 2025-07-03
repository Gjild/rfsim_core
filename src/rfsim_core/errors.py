# src/rfsim_core/errors.py
import logging
from typing import Any, Dict, Protocol, abstractmethod
from typing import runtime_checkable  # NEW: Import for runtime protocol checking

logger = logging.getLogger(__name__)

# --- User-Facing Exception Hierarchy ---

class RFSimError(Exception):
    """Base class for all custom, user-facing errors in RFSim Core."""
    pass

class CircuitBuildError(RFSimError):
    """
    Raised when the circuit construction process fails for any reason, from parsing
    to parameter resolution. The message is a pre-formatted, user-friendly diagnostic report.
    """
    pass

class SimulationRunError(RFSimError):
    """
    Raised when the simulation execution fails for any reason after a successful build,
    such as a semantic validation error or a numerical issue.
    The message is a pre-formatted, user-friendly diagnostic report.
    """
    pass


# --- Diagnostic Protocol & Base Exception (The Explicit Contract) ---

@runtime_checkable  # NEW: Makes this protocol compatible with isinstance() at runtime.
class Diagnosable(Protocol):
    """
    A protocol for exceptions that can generate their own rich diagnostic report.
    This remains for static type analysis, ensuring that any code can work with a
    "diagnosable" object without needing to know its concrete type.
    """
    def get_diagnostic_report(self) -> str:
        """Generates a complete, user-friendly, multi-line report string."""
        ...

class DiagnosableError(Exception, Diagnosable):
    """
    A common, concrete base class for all internal exceptions that are diagnosable.

    This is the core of the architectural fix. It serves two purposes:
    1. It inherits from `Exception`, making it a valid, concrete class for use in
       `except` clauses.
    2. It implements the `Diagnosable` protocol and declares `get_diagnostic_report`
       as an abstract method. This uses the "Correctness by Construction" principle
       to FORCE all subclasses to provide an implementation for generating a
       user-friendly report, or they will fail at instantiation time.
    """
    @abstractmethod
    def get_diagnostic_report(self) -> str:
        """
        Abstract method to generate the diagnostic report.
        Subclasses MUST implement this.
        """
        raise NotImplementedError


# --- Stateless Formatting Utility ---

def format_diagnostic_report(
    error_type: str,
    details: str,
    suggestion: str,
    context: Dict[str, Any]
) -> str:
    """
    A stateless helper to format the final multi-line report string, ensuring a
    consistent look and feel for all user-facing diagnostics.

    Args:
        error_type: The high-level category of the error (e.g., "Unresolved Symbol").
        details: A detailed, potentially multi-line description of the problem.
        suggestion: Actionable advice for the user to resolve the issue.
        context: A dictionary of contextual information (FQN, file path, user input, etc.).

    Returns:
        A formatted, user-friendly diagnostic report string ready for display.
    """
    lines = [
        "\n",
        "================ RFSim Core: Actionable Diagnostic Report ================",
        f"Error Type:     {error_type}",
    ]
    if fqn := context.get('fqn'):
        lines.append(f"FQN:            {fqn}")
    if source_file := context.get('source_file'):
        lines.append(f"Source File:    {source_file}")
    if user_input := context.get('user_input'):
        lines.append(f"User Input:     '{user_input}'")
    if frequency := context.get('frequency'):
        lines.append(f"Frequency:      {frequency}")

    lines.append("\nDetails:")
    for line in details.splitlines():
        lines.append(f"  {line}")

    if suggestion:
        lines.append("\nSuggestion:")
        for line in suggestion.splitlines():
            lines.append(f"  {line}")

    lines.append("========================================================================")
    return "\n".join(lines)