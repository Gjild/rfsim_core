# src/rfsim_core/validation/issues.py
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ValidationIssueLevel(Enum):
    """Severity level of a validation issue."""
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"

    def __str__(self):
        return self.value


@dataclass
class ValidationIssue:
    """
    Represents a single validation issue found during semantic checks.
    Includes full hierarchical context for actionable diagnostics.
    """
    level: ValidationIssueLevel
    code: str
    message: str
    component_fqn: Optional[str] = None
    hierarchical_context: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        parts = [f"[{self.level.name} - {self.code}]"]
        if self.hierarchical_context:
            parts.append(f"Context: {self.hierarchical_context}")
        if self.component_fqn:
            parts.append(f"Component: {self.component_fqn}")
        parts.append(f"Message: {self.message}")

        if self.details:
            filtered_details = {
                k: v for k, v in self.details.items()
                if k not in ['instance_fqn', 'hierarchical_context', 'component_fqn']
            }
            if filtered_details:
                details_str = ", ".join(f"{k}={v}" for k, v in sorted(filtered_details.items()))
                parts.append(f"Details: ({details_str})")

        return " ".join(parts)