# src/rfsim_core/parser/exceptions.py
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

from ..errors import Diagnosable, format_diagnostic_report

class BaseParsingError(ValueError):
    """Base class for parsing and schema validation errors."""
    pass

@dataclass(frozen=True)
class ParsingError(BaseParsingError, Diagnosable):
    """Custom exception for logical errors or file issues during parsing."""
    details: str
    file_path: Path

    def __str__(self):
        return f"Parsing error in file '{self.file_path}': {self.details}"

    def get_diagnostic_report(self) -> str:
        return format_diagnostic_report(
            error_type="YAML Parsing or File Error",
            details=self.details,
            suggestion="Ensure the file exists, has correct permissions, and contains valid YAML syntax.",
            context={'source_file': self.file_path}
        )

@dataclass(frozen=True)
class SchemaValidationError(BaseParsingError, Diagnosable):
    """Custom exception for YAML schema validation failures."""
    errors: Dict[str, Any]
    file_path: Path

    def __str__(self):
        error_lines = [
            f"  - In field '{'.'.join(map(str, k))}': {v[0]}"
            for k, v in sorted(self.errors.items())
        ]
        return (
            f"YAML schema validation failed for file '{self.file_path}':\n"
            + "\n".join(error_lines)
        )

    def get_diagnostic_report(self) -> str:
        details = "The structure of the YAML file does not conform to the required schema.\nSee details below:\n" + "\n".join(
            f"  - Field '{'.'.join(map(str, k))}': {v[0]}" for k, v in sorted(self.errors.items())
        )
        return format_diagnostic_report(
            error_type="YAML Schema Validation Error",
            details=details,
            suggestion="Correct the specified fields in the YAML file to match the documented format.",
            context={'source_file': self.file_path}
        )