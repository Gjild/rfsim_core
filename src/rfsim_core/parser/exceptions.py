# src/rfsim_core/parser/exceptions.py
"""
Defines custom, diagnosable exceptions for the parsing and schema validation stage.

This module establishes the specific error contracts for the NetlistParser. It adheres
to the definitive exception handling architecture by:

1.  **Inheriting from a Common Base:** All exceptions defined here ultimately derive
    from the global `DiagnosableError` base class. This allows high-level error
    handlers (like in `CircuitBuilder`) to catch a single, concrete type for all
    known, reportable application errors.

2.  **Providing Explicit Contracts:** Each exception class is a self-contained contract.
    `ParsingError` is for file-level or syntax issues, while `SchemaValidationError`
    is for structural issues against the Cerberus schema.

3.  **Enforcing Actionable Diagnostics:** Each concrete exception class (`ParsingError`,
    `SchemaValidationError`) implements the `get_diagnostic_report` method, as
    mandated by the `DiagnosableError` abstract base class. This guarantees that every
    parsing-related error can be translated into a user-friendly, actionable report,
    fulfilling a core design mandate of the project.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

# Import the new, concrete, catchable base class and the formatting utility.
# This creates a clear dependency on the core error-handling architecture.
from ..errors import DiagnosableError, format_diagnostic_report


class BaseParsingError(DiagnosableError):
    """
    A local, concrete base class for all YAML parsing and schema validation errors.

    This class inherits from DiagnosableError, hooking the entire family of parsing
    exceptions into the global, catchable exception hierarchy.
    """
    def get_diagnostic_report(self) -> str:
        """
        Provides a fallback diagnostic report for any BaseParsingError subclasses
        that might not implement their own. This ensures the contract from
        DiagnosableError is always met.
        """
        return format_diagnostic_report(
            error_type="Generic Parsing Error",
            details=str(self),
            suggestion="Please check the format and content of the relevant YAML netlist file.",
            context={}
        )


@dataclass(frozen=True)
class ParsingError(BaseParsingError):
    """
    Custom exception for logical errors or file-system issues during parsing.
    This covers cases like a file not being found, having incorrect permissions,
    or containing fundamentally invalid YAML syntax that prevents loading.
    """
    details: str
    file_path: Path

    def __str__(self):
        return f"Parsing error in file '{self.file_path}': {self.details}"

    def get_diagnostic_report(self) -> str:
        """
        Generates a detailed report specific to file or syntax-level errors.
        """
        return format_diagnostic_report(
            error_type="YAML Parsing or File Error",
            details=self.details,
            suggestion="Ensure the file exists, has the correct read permissions, and contains valid YAML syntax.",
            context={'source_file': self.file_path}
        )


@dataclass(frozen=True)
class SchemaValidationError(BaseParsingError):
    """
    Custom exception for failures during Cerberus schema validation.
    This is raised when the YAML is syntactically valid but does not conform to the
    required structure of an RFSim Core netlist (e.g., missing keys, invalid
    identifiers, duplicate component IDs).
    """
    errors: Dict[str, Any]
    file_path: Path

    def __str__(self):
        """Provides a concise, multi-line summary suitable for logging."""
        error_lines = [
            f"  - In field '{'.'.join(map(str, k))}': {v[0]}"
            for k, v in sorted(self.errors.items())
        ]
        return (
            f"YAML schema validation failed for file '{self.file_path}':\n"
            + "\n".join(error_lines)
        )

    def get_diagnostic_report(self) -> str:
        """
        Generates a rich, user-friendly report that lists every schema violation
        found in the file, providing precise guidance for correction.
        """
        # Build a detailed, readable list of all errors found by Cerberus.
        error_list_str = "\n".join(
            f"  - Field '{'.'.join(map(str, k))}': {v[0]}"
            for k, v in sorted(self.errors.items())
        )
        details = (
            "The structure of the YAML file does not conform to the required schema.\n"
            f"See details for {len(self.errors)} issue(s) below:\n\n{error_list_str}"
        )

        return format_diagnostic_report(
            error_type="YAML Schema Validation Error",
            details=details,
            suggestion="Correct the specified fields in the YAML file to match the documented format. Check for invalid identifiers (e.g., using '-' or '.'), duplicate component IDs, or missing required sections like 'components'.",
            context={'source_file': self.file_path}
        )