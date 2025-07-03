# src/rfsim_core/validation/exceptions.py
"""
Defines the custom, diagnosable exception for the semantic validation process.

This module provides the `SemanticValidationError`, which is the formal, reportable
error raised when the `SemanticValidator` finds one or more logical errors in a
synthesized circuit model.

**Architectural Refinement (Phase 10):**
This exception now inherits from the `DiagnosableError` base class. This is a critical
change that aligns with the project's core principles:

1.  **Catchable by Type:** As a subclass of `DiagnosableError` (which inherits from
    `Exception`), it can be caught explicitly in `try...except` blocks, fulfilling
    the "Explicit Contracts" mandate.
2.  **Guaranteed Diagnosable:** The `DiagnosableError` abstract base class requires
    the implementation of the `get_diagnostic_report()` method, ensuring that this
    exception is always capable of producing a user-friendly, actionable report.
"""
from typing import List

# Import the contract for a single issue.
from .issues import ValidationIssue, ValidationIssueLevel

# Import the new, concrete base class and the standard report formatter.
from ..errors import DiagnosableError, format_diagnostic_report


class SemanticValidationError(DiagnosableError):
    """
    Custom exception raised when semantic validation detects one or more errors.

    This exception is a container for all error-level `ValidationIssue` objects
    found during a validation pass. It is responsible for formatting these issues
    into a single, comprehensive diagnostic report for the user.
    """
    def __init__(self, issues: List[ValidationIssue]):
        """
        Initializes the exception with the list of validation issues.

        Args:
            issues: The complete list of issues found by the SemanticValidator.
                    This constructor will filter this list to include only those
                    with a level of `ERROR`.
        """
        self.issues: List[ValidationIssue] = [
            issue for issue in issues if issue.level == ValidationIssueLevel.ERROR
        ]
        if not self.issues:
            # This case should theoretically not be reached if the caller checks for errors,
            # but it is a safeguard against misuse.
            summary_message = "SemanticValidationError was raised with no error-level issues."
        else:
            error_lines = [str(issue) for issue in self.issues]
            summary_message = (
                f"Semantic validation failed with {len(self.issues)} error(s):\n"
                + "\n".join(f"  - {line}" for line in error_lines)
            )
        super().__init__(summary_message)

    def get_diagnostic_report(self) -> str:
        """
        Generates a complete, user-friendly, multi-line diagnostic report string
        that details every validation error found.

        This method fulfills the `Diagnosable` contract.

        Returns:
            A formatted, user-friendly diagnostic report string.
        """
        error_lines = [str(issue) for issue in self.issues]
        details = (
            f"One or more logical errors were found in the circuit definition.\n"
            f"Found {len(self.issues)} error(s). See details below:\n\n"
            + "\n".join(f"  - {line}" for line in error_lines)
        )

        # For the top-level report context, use information from the first error
        # as a reasonable heuristic for the primary point of failure.
        first_issue = self.issues[0] if self.issues else None
        context = {}
        if first_issue:
            # Prefer the most specific FQN available.
            context['fqn'] = first_issue.component_fqn or first_issue.hierarchical_context or 'Multiple'
            # Extract source file path from the issue's details dictionary if present.
            if hasattr(first_issue, 'details') and (source_path := first_issue.details.get('source_yaml_path')):
                 context['source_file'] = source_path

        return format_diagnostic_report(
            error_type="Circuit Semantic Validation Error",
            details=details,
            suggestion="Review and correct all validation errors listed above in the relevant circuit netlist(s).",
            context=context
        )