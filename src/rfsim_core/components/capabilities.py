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
- IMnaContributor, IDcContributor, ITopologyContributor: Specific protocols defining
  the contracts for contributing to MNA, DC, and Topology analysis, respectively.
- IConnectivityProvider: A protocol defining the contract for reporting internal
  port-to-port connectivity, making analysis engines truly type-agnostic.
- @provides: A class decorator for declaratively registering a class as an
  implementation of a specific capability. This automates discovery and
  improves the developer experience for plugin authors.
- TCapability: A TypeVar for precise type-hinting of capability queries.
"""

import logging
from typing import (
    Union,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    TYPE_CHECKING,
    runtime_checkable,
)

import numpy as np

# These imports are essential to define the method signatures correctly.
from ..units import Quantity
from .base_enums import DCBehaviorType

# Use TYPE_CHECKING to import ComponentBase only for type analysis,
# preventing a circular import at runtime.
if TYPE_CHECKING:
    from .base import ComponentBase

logger = logging.getLogger(__name__)


@runtime_checkable
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

@runtime_checkable
class IMnaContributor(ComponentCapability, Protocol):
    """
    Defines the capability of a component to contribute to a linear,
    frequency-domain MNA system.

    ARCHITECTURAL CONTRACT:
    1.  **Vectorization:** This method MUST be a vectorized operation. It accepts the full array of
        simulation frequencies and is expected to return the MNA stamp contribution
        for all frequencies in a single, efficient computation. The simulation executive will call this
        method ONCE, before the per-frequency loop begins.

    2.  **Simplified Return Type:** This method MUST return a single tuple containing the
        vectorized admittance `Quantity` and a list of the corresponding string-based port names.
        Returning a list of stamps is forbidden.
    """

    def get_mna_stamps(
        self,
        component: "ComponentBase",  # The component instance provides context.
        freq_hz_array: np.ndarray,
        all_evaluated_params: Dict[str, Quantity],
    ) -> Tuple[Quantity, List[str]]:
        """
        Computes and returns the component's MNA stamp contribution.

        Args:
            component: The parent component instance, providing context.
            freq_hz_array: The 1D NumPy array of frequencies for the sweep.
            all_evaluated_params: A dictionary of all resolved parameters for the sweep.

        Returns:
            A single tuple containing:
            - A `pint.Quantity` with `[admittance]` dimensions. Its magnitude
              must be a NumPy array of shape (num_freqs, N, N), where N is the
              number of ports.
            - A list of N strings, where each string is a canonical port name
              (e.g., `['p1', 'p2']`) corresponding to the stamp's axes.
        """
        ...

# Define a type alias for the admittance part of the return value
DcAdmittancePayload = Union[
    Quantity,  # For scalar admittance
    Tuple[Quantity, List[str]] # For matrix admittance + port order
]

@runtime_checkable
class IDcContributor(ComponentCapability, Protocol):
    """
    Defines the capability of a component to contribute to a DC (F=0) analysis.
    The signature MUST match the data-passing requirements of the DCAnalyzer.
    """

    def get_dc_behavior(
        self,
        component: "ComponentBase",  # The component instance provides context.
        all_dc_params: Dict[str, Quantity],
    ) -> Tuple[DCBehaviorType, Optional[DcAdmittancePayload]]:
        ...


@runtime_checkable
class ITopologyContributor(ComponentCapability, Protocol):
    """
    Defines the capability of a component to report its structural topology
    based on its constant-valued parameters. This is used by the TopologyAnalyzer
    to identify components that are permanently open-circuited.
    """

    def is_structurally_open(
        self,
        component: "ComponentBase",  # The component instance provides context.
        resolved_constant_params: Dict[str, Quantity],
    ) -> bool:
        ...


@runtime_checkable
class IConnectivityProvider(ComponentCapability, Protocol):
    """
    Defines the capability of a component to report its effective external
    port-to-port connectivity as a list of string pairs.

    This capability is essential for making analysis engines like TopologyAnalyzer
    and MnaAssembler truly agnostic to component types (leaf vs. hierarchical),
    thereby purifying the component model and eradicating special-case `isinstance`
    checks in the simulation core.
    """
    def get_connectivity(
        self,
        component: "ComponentBase",  # The component instance provides context.
    ) -> List[Tuple[str, str]]:
        """
        Returns a list of tuples, where each tuple represents a pair of
        the component's external port names that are conductively connected.
        For a simple 2-port resistor, this would be [('p1', 'p2')].
        For a subcircuit, this is dynamically determined via recursive analysis.
        """
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
        logger.debug(
            f"Class '{cls.__name__}' registered as providing capability '{capability_protocol.__name__}'."
        )
        return cls

    return decorator