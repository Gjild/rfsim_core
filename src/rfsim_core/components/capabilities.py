# src/rfsim_core/components/capabilities.py
"""
Defines the foundational capability architecture for RFSim Core components.

This module introduces the concept of "capabilities" using `typing.Protocol`.
This decouples analysis engines (like MnaAssembler, DCAnalyzer) from the
concrete component implementations. Instead of requiring components to inherit
from a monolithic base class with many abstract methods, analysis engines can
query a component instance to see if it provides a specific, required capability
(e.g., `IMnaContributor`).

This architecture is mandated for its extensibility and maintainability, allowing
the future addition of new analysis domains (e.g., noise, harmonic balance)
without requiring breaking changes to the existing component API.

Key elements:
- ComponentCapability: A marker protocol for all capabilities.
- IMnaContributor, IDcContributor: Specific protocols defining the contracts for
  contributing to MNA and DC analysis, respectively.
- @provides: A class decorator for declaratively registering a class as an
  implementation of a specific capability. This automates discovery and
  improves the developer experience for plugin authors.
- TCapability: A TypeVar for precise type-hinting of capability queries.
"""

import logging
from typing import Protocol, Type, Dict, Optional, List, Tuple, TypeVar, TYPE_CHECKING
import numpy as np

# These imports are essential to define the method signatures correctly.
from ..units import Quantity
from .base_enums import DCBehaviorType  # Import from new, separated file.

# Use TYPE_CHECKING to import ComponentBase only for type analysis,
# preventing a circular import at runtime.
if TYPE_CHECKING:
    from .base import ComponentBase

logger = logging.getLogger(__name__)


class ComponentCapability(Protocol):
    """
    A marker protocol for all component capabilities. Any class that provides
    a specific functionality to an analysis engine should conform to a
    protocol that inherits from this one.
    """
    pass


# TCapability is a TypeVar bound to ComponentCapability. This allows for precise
# type-hinting in `ComponentBase.get_capability`, ensuring that a request for
# `IMnaContributor` is known by the type checker to return an `IMnaContributor`.
TCapability = TypeVar("TCapability", bound=ComponentCapability)


class IMnaContributor(ComponentCapability):
    """
    Defines the capability of a component to contribute to a linear,
    frequency-domain MNA system.

    ARCHITECTURAL CONTRACT:
    This method MUST be a vectorized operation. It accepts the full array of
    simulation frequencies and is expected to return the MNA stamp contributions
    for all frequencies in a single, efficient computation. This is critical for
    maintaining simulation performance. The simulation executive will call this
    method ONCE, before the per-frequency loop begins.
    """
    def get_mna_stamps(
        self,
        component: 'ComponentBase',  # The component instance provides context.
        freq_hz_array: np.ndarray,
        all_evaluated_params: Dict[str, Quantity]
    ) -> List[Tuple[Quantity, List[str | int]]]:
        ...


class IDcContributor(ComponentCapability):
    """
    Defines the capability of a component to contribute to a DC (F=0) analysis.
    The signature MUST match the data-passing requirements of the DCAnalyzer.
    """
    def get_dc_behavior(
        self,
        component: 'ComponentBase',  # The component instance provides context.
        all_dc_params: Dict[str, Quantity]
    ) -> Tuple[DCBehaviorType, Optional[Quantity]]:
        ...


def provides(capability_protocol: Type[ComponentCapability]):
    """
    A class decorator to register a class as an implementation for a capability.

    This decorator attaches a private attribute, `_implements_capability`, to the
    decorated class. The `ComponentBase.declare_capabilities` method will use this
    attribute for automatic discovery. This implements the 'Declarative Registration'
    mandate, making the API robust and easy to use for plugin authors.

    Args:
        capability_protocol: The capability Protocol (e.g., IMnaContributor)
                             that this class implements.
    """
    def decorator(cls: Type) -> Type:
        if not issubclass(capability_protocol, ComponentCapability):
             raise TypeError(
                f"Decorator argument for @provides must be a ComponentCapability "
                f"Protocol, but got {capability_protocol}."
             )
        # Attach the metadata directly to the implementation class.
        cls._implements_capability = capability_protocol
        logger.debug(f"Class '{cls.__name__}' registered as providing capability '{capability_protocol.__name__}'.")
        return cls
    return decorator