# src/rfsim_core/errors.py
import logging
from typing import Any, Dict, Protocol

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


# --- Diagnostic Protocol (The Explicit Contract) ---

class Diagnosable(Protocol):
    """
    A protocol for exceptions that can generate their own rich diagnostic report.
    This decouples error handlers from the specific types of exceptions they process.
    """
    def get_diagnostic_report(self) -> str:
        """Generates a complete, user-friendly, multi-line report string."""
        ...


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