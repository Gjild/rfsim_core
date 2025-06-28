# src/rfsim_core/validation/exceptions.py
from typing import List

from .issues import ValidationIssue
from ..errors import Diagnosable, format_diagnostic_report

class SemanticValidationError(ValueError, Diagnosable):
    """
    Custom exception raised when semantic validation detects one or more errors.
    Contains a list of all error-level ValidationIssue objects and is diagnosable.
    """
    def __init__(self, issues: List[ValidationIssue]):
        self.issues: List[ValidationIssue] = [
            issue for issue in issues if issue.level == "ERROR"
        ]
        error_lines = [str(issue) for issue in self.issues]
        summary_message = (
            f"Semantic validation failed with {len(self.issues)} error(s):\n"
            + "\n".join(f"  - {line}" for line in error_lines)
        )
        super().__init__(summary_message)
    
    def get_diagnostic_report(self) -> str:
        """Generates a report that lists all validation errors."""
        error_lines = [str(issue) for issue in self.issues]
        details = (
            f"One or more logical errors were found in the circuit definition.\n"
            f"Found {len(self.issues)} error(s). See details below:\n\n"
            + "\n".join(f"  - {line}" for line in error_lines)
        )
        
        first_issue = self.issues[0] if self.issues else None
        context = {}
        if first_issue:
            context['fqn'] = first_issue.component_fqn or first_issue.hierarchical_context or 'Multiple'
            # Assuming ValidationIssue.details might contain the source path
            if hasattr(first_issue, 'details') and (source_path := first_issue.details.get('source_yaml_path')):
                 context['source_file'] = source_path
        
        return format_diagnostic_report(
            error_type="Circuit Semantic Validation Error",
            details=details,
            suggestion="Review and correct all validation errors listed above in the circuit netlist(s).",
            context=context
        )