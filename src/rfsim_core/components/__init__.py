# --- Modify: src/rfsim_core/components/__init__.py ---
import logging
logger = logging.getLogger(__name__)

# Import base first to define registry and decorator
from .base import (
    ComponentBase, COMPONENT_REGISTRY, register_component, DCBehaviorType
)
from .exceptions import ComponentError 
# Import concrete elements to trigger registration
from .elements import Resistor, Capacitor, Inductor

# --- Import constants if needed directly from components namespace (Optional) ---
# from ..constants import LARGE_ADMITTANCE_SIEMENS # Keep this commented unless explicitly desired

logger.info(f"Available component types: {list(COMPONENT_REGISTRY.keys())}")

__all__ = [
    "ComponentBase",
    "COMPONENT_REGISTRY",
    "register_component",
    "Resistor",
    "Capacitor",
    "Inductor",
    "ComponentError",
    "DCBehaviorType",
]