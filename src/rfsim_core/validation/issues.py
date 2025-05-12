# --- src/rfsim_core/validation/issues.py ---
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
    """
    level: ValidationIssueLevel
    code: str  # Unique string code for the issue type, e.g., "NET_CONN_001"
    message: str  # Human-readable, formatted message describing the issue
    component_id: Optional[str] = None
    net_name: Optional[str] = None
    parameter_name: Optional[str] = None
    details: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __str__(self) -> str:
        parts = [
            f"[{self.level.name} - {self.code}]: {self.message}"
        ]
        if self.component_id:
            parts.append(f"Component: {self.component_id}")
        if self.net_name:
            parts.append(f"Net: {self.net_name}")
        if self.parameter_name:
            parts.append(f"Parameter: {self.parameter_name}")
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            parts.append(f"Details: ({details_str})")
        return " ".join(parts)