# src/rfsim_core/parameters/exceptions.py
"""
Defines the custom, diagnosable exceptions for the parameter subsystem.

This module establishes a clear and robust hierarchy for all errors that can occur
during parameter definition, resolution, parsing, and evaluation.

Architectural Significance:
- All exceptions defined herein inherit from the new `DiagnosableError` base class
  (either directly or through the local `ParameterError` base).
- This ensures that every parameter-related error is a concrete, catchable type
  that is guaranteed to provide a rich, user-friendly diagnostic report via the
  `get_diagnostic_report()` method.
- This design avoids generic error handling and fulfills the project's core mandate
  for "Actionable Diagnostics" by embedding all necessary context for debugging
  directly within the exception objects.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

# This is the cornerstone import for the revised architecture.
# We import the new concrete base class and the stateless formatting utility.
from ..errors import DiagnosableError, format_diagnostic_report


class ParameterError(DiagnosableError):
    """
    A concrete base class for all parameter-related errors.

    By inheriting from DiagnosableError, this class and all its children can be
    caught with a single, explicit `except ParameterError:` block, while still
    guaranteeing that the `get_diagnostic_report` method is available.
    """
    def get_diagnostic_report(self) -> str:
        """
        Provides a generic, fallback diagnostic report for the base class.
        Subclasses are expected to provide more specific implementations.
        """
        return format_diagnostic_report(
            error_type="Generic Parameter Error",
            details=str(self),
            suggestion="Review the parameters in your netlist files for correctness.",
            context={}
        )


@dataclass(frozen=True)
class ParameterDefinitionError(ParameterError):
    """Raised for errors during the definition or parsing of a single parameter."""
    fqn: str
    user_input: str
    source_yaml_path: Path
    details: str

    def __str__(self):
        return (f"Parameter '{self.fqn}': {self.details} "
                f"(input: '{self.user_input}', file: '{self.source_yaml_path}')")

    def get_diagnostic_report(self) -> str:
        """Generates a detailed report for an invalid parameter definition."""
        return format_diagnostic_report(
            error_type="Invalid Parameter Definition",
            details=self.details,
            suggestion="Ensure the parameter value has valid syntax and, if applicable, correct units.",
            context={
                'fqn': self.fqn,
                'source_file': self.source_yaml_path,
                'user_input': self.user_input,
            }
        )


@dataclass(frozen=True)
class ParameterScopeError(ParameterError):
    """Raised when a symbol in an expression cannot be resolved in its lexical scope."""
    owner_fqn: str
    unresolved_symbol: str
    user_input: str
    source_yaml_path: Path
    resolution_path_details: str

    def __str__(self):
        return (f"Unresolved symbol '{self.unresolved_symbol}' in expression for "
                f"'{self.owner_fqn}' ('{self.user_input}')")

    def get_diagnostic_report(self) -> str:
        """Generates a detailed report for a symbol that could not be found."""
        main_details = (
            f"The symbol '{self.unresolved_symbol}' could not be resolved within the scope of parameter '{self.owner_fqn}'.\n\n"
            f"{self.resolution_path_details}"
        )
        return format_diagnostic_report(
            error_type="Unresolved Symbol in Expression",
            details=main_details,
            suggestion="Check the spelling of the symbol. Ensure it is defined in an accessible scope (e.g., in the 'parameters' block of the current file, a parent file, or as a component parameter within the same circuit).",
            context={
                'fqn': self.owner_fqn,
                'source_file': self.source_yaml_path,
                'user_input': self.user_input,
            }
        )


@dataclass(frozen=True)
class ParameterSyntaxError(ParameterError):
    """Raised for invalid syntax in a parameter expression that prevents parsing."""
    owner_fqn: str
    user_input: str
    source_yaml_path: Path
    details: str

    def __str__(self):
        return f"Invalid syntax for parameter '{self.owner_fqn}' ('{self.user_input}'): {self.details}"

    def get_diagnostic_report(self) -> str:
        """Generates a detailed report for a syntax error in an expression."""
        return format_diagnostic_report(
            error_type="Invalid Expression Syntax",
            details=self.details,
            suggestion="Correct the syntax of the expression. Note that identifiers cannot be Python keywords (e.g., 'for', 'if', 'class').",
            context={
                'fqn': self.owner_fqn,
                'source_file': self.source_yaml_path,
                'user_input': self.user_input,
            }
        )


@dataclass(frozen=True)
class CircularParameterDependencyError(ParameterError):
    """Raised when a circular dependency is detected among parameters."""
    cycle: List[str]

    def __str__(self):
        # Create a display-friendly version of the cycle, e.g., A -> B -> A
        cycle_display = self.cycle + [self.cycle[0]] if self.cycle and self.cycle[0] != self.cycle[-1] else self.cycle
        return f"Circular dependency detected: {' -> '.join(cycle_display)}"

    def get_diagnostic_report(self) -> str:
        """Generates a detailed report showing the detected dependency cycle."""
        return format_diagnostic_report(
            error_type="Circular Parameter Dependency",
            details=f"A circular reference was detected involving the following parameters:\n{' -> '.join(self.cycle)}",
            suggestion="Break the dependency cycle by redefining one of the parameters to not depend on the others in the loop.",
            context={'fqn': self.cycle[0]}  # Use the first parameter in the cycle as the primary context.
        )


@dataclass(frozen=True)
class ParameterEvaluationError(ParameterError):
    """Raised when a runtime error occurs during the numerical evaluation of a valid expression."""
    fqn: str
    details: str
    error_indices: Optional[np.ndarray]
    frequencies: Optional[np.ndarray]
    input_values: Dict[str, Any]

    def __str__(self):
        freq_str = ""
        if self.frequencies is not None and self.frequencies.size > 0:
            freq_str = f" at frequency {self.frequencies[0]:.4e} Hz"
        return f"Evaluation error for '{self.fqn}'{freq_str}: {self.details}"

    def get_diagnostic_report(self) -> str:
        """
        Generates a highly detailed report, including the specific frequency and
        input values that caused the first failure in a vectorized evaluation.
        """
        # Attempt to find the first point of failure for a rich diagnostic.
        if self.error_indices is not None and self.frequencies is not None and self.error_indices.size > 0:
            first_fail_idx = self.error_indices[0]
            first_fail_freq = self.frequencies[first_fail_idx]

            # Extract the specific input values at the point of failure.
            input_vals_at_first_failure = {
                name: val[first_fail_idx] if isinstance(val, np.ndarray) and val.ndim > 0 and val.size > first_fail_idx else val
                for name, val in self.input_values.items()
            }
            input_details = "\n".join(f"  - {name} = {val}" for name, val in input_vals_at_first_failure.items())

            details_str = (
                f"{self.details}\n\n"
                f"The error first occurred at sweep index {first_fail_idx} (frequency = {first_fail_freq:.4e} Hz) "
                f"with the following input values:\n{input_details}"
            )
            if self.error_indices.size > 1:
                details_str += f"\n\nNote: This error occurred at {self.error_indices.size} total frequency points."
        else:
            # Fallback if specific index information is not available.
            details_str = self.details

        return format_diagnostic_report(
            error_type="Parameter Evaluation Error",
            details=details_str,
            suggestion="Check the expression for operations that may be invalid at the given frequency (e.g., division by zero, logarithms of non-positive numbers, or invalid arguments to functions).",
            context={'fqn': self.fqn}
        )