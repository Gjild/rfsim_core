# src/rfsim_core/parameters/exceptions.py
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

from ..errors import Diagnosable, format_diagnostic_report


class ParameterError(ValueError):
    """Base class for all parameter-related errors."""
    pass


@dataclass(frozen=True)
class ParameterDefinitionError(ParameterError, Diagnosable):
    """Error during the definition or parsing of a single parameter."""
    fqn: str
    user_input: str
    source_yaml_path: Path
    details: str

    def __str__(self):
        return f"Parameter '{self.fqn}': {self.details} (input: '{self.user_input}', file: '{self.source_yaml_path}')"

    def get_diagnostic_report(self) -> str:
        return format_diagnostic_report(
            error_type="Invalid Parameter Definition",
            details=self.details,
            suggestion="Ensure the parameter value has a valid syntax and its units are correct.",
            context={
                'fqn': self.fqn,
                'source_file': self.source_yaml_path,
                'user_input': self.user_input,
            }
        )


@dataclass(frozen=True)
class ParameterScopeError(ParameterError, Diagnosable):
    """Raised when a symbol cannot be resolved in its scope."""
    owner_fqn: str
    unresolved_symbol: str
    user_input: str
    source_yaml_path: Path
    resolution_path_details: str

    def __str__(self):
        return f"Unresolved symbol '{self.unresolved_symbol}' in expression for '{self.owner_fqn}' ('{self.user_input}')"

    def get_diagnostic_report(self) -> str:
        main_details = f"The symbol '{self.unresolved_symbol}' could not be resolved within the scope of parameter '{self.owner_fqn}'.\n\n{self.resolution_path_details}"
        return format_diagnostic_report(
            error_type="Unresolved Symbol in Expression",
            details=main_details,
            suggestion="Check the spelling of the symbol or define it in an accessible scope (e.g., in the 'parameters' block of the relevant YAML file).",
            context={
                'fqn': self.owner_fqn,
                'source_file': self.source_yaml_path,
                'user_input': self.user_input,
            }
        )


@dataclass(frozen=True)
class ParameterSyntaxError(ParameterError, Diagnosable):
    """Raised for invalid syntax in a parameter expression."""
    owner_fqn: str
    user_input: str
    source_yaml_path: Path
    details: str

    def __str__(self):
        return f"Invalid syntax for parameter '{self.owner_fqn}' ('{self.user_input}'): {self.details}"

    def get_diagnostic_report(self) -> str:
        return format_diagnostic_report(
            error_type="Invalid Expression Syntax",
            details=self.details,
            suggestion="Correct the syntax of the expression.",
            context={
                'fqn': self.owner_fqn,
                'source_file': self.source_yaml_path,
                'user_input': self.user_input,
            }
        )


@dataclass(frozen=True)
class CircularParameterDependencyError(ParameterError, Diagnosable):
    """Raised when a circular dependency is detected."""
    cycle: List[str]

    def __str__(self):
        cycle_display = self.cycle + [self.cycle[0]] if self.cycle and self.cycle[0] != self.cycle[-1] else self.cycle
        return f"Circular dependency detected: {' -> '.join(cycle_display)}"

    def get_diagnostic_report(self) -> str:
        return format_diagnostic_report(
            error_type="Circular Parameter Dependency",
            details=f"A circular reference was detected involving the following parameters:\n{' -> '.join(self.cycle)}",
            suggestion="Break the dependency cycle by redefining one of the parameters to not depend on the others in the loop.",
            context={'fqn': self.cycle[0]}
        )


@dataclass(frozen=True)
class ParameterEvaluationError(ParameterError, Diagnosable):
    """Raised when a runtime error occurs during parameter expression evaluation."""
    fqn: str
    details: str
    error_indices: Optional[np.ndarray]
    frequencies: Optional[np.ndarray]
    input_values: Dict[str, Any]

    def __str__(self):
        freq_str = f" at frequency {self.frequencies[0]:.4e} Hz" if self.frequencies is not None and len(self.frequencies) > 0 else ""
        return f"Evaluation error for '{self.fqn}'{freq_str}: {self.details}"

    def get_diagnostic_report(self) -> str:
        if self.error_indices is not None and self.frequencies is not None and self.error_indices.size > 0:
            first_fail_idx = self.error_indices[0]
            first_fail_freq = self.frequencies[first_fail_idx]
            input_vals_at_first_failure = {
                name: val[first_fail_idx] if isinstance(val, np.ndarray) and val.ndim > 0 else val
                for name, val in self.input_values.items()
            }
            input_details = "\n".join(f"  - {name} = {val}" for name, val in input_vals_at_first_failure.items())
            
            details_str = (
                f"{self.details}\n\n"
                f"The error first occurred at index {first_fail_idx} (frequency = {first_fail_freq:.4e} Hz) "
                f"with the following input values:\n{input_details}"
            )
            if self.error_indices.size > 1:
                details_str += f"\n\nNote: This error occurred at {self.error_indices.size} total frequency points."
        else:
            details_str = self.details
            
        return format_diagnostic_report(
            error_type="Parameter Evaluation Error",
            details=details_str,
            suggestion="Check the expression for operations that may be invalid at the given frequency (e.g., division by zero, logarithms of non-positive numbers).",
            context={'fqn': self.fqn}
        )