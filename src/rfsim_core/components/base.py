# src/rfsim_core/components/base.py

import logging
import inspect
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, ClassVar, Optional, Type

import numpy as np

from ..units import ureg, Quantity
from ..parameters import ParameterManager
from .capabilities import (
    ComponentCapability, TCapability, IMnaContributor, IDcContributor,
    ITopologyContributor, IConnectivityProvider, provides
)
from .base_enums import DCBehaviorType


logger = logging.getLogger(__name__)

# The port identifiers in this tuple are now EXCLUSIVELY strings,
# a non-negotiable contract for type safety and clarity.
StampInfo = Tuple[Quantity, List[str]]


class ComponentBase(ABC):
    """
    The abstract base class for all circuit components in RFSim Core.

    This class establishes the fundamental contract for component identity,
    parameter declaration, and, most importantly, provides the queryable
    capability system that decouples components from analysis engines. It serves
    as a provider of capabilities rather than a monolithic implementation of
    all possible analysis behaviors.
    """
    component_type_str: ClassVar[str] = "BaseComponent"

    def __init__(
        self,
        instance_id: str,
        component_type_str: str,
        parameter_manager: ParameterManager,
        parent_hierarchical_id: str,
        port_net_map: Dict[str, str],
    ):
        """
        Initializes the base attributes of a component instance.

        Args:
            instance_id: The unique ID of this component instance (e.g., 'R1').
            component_type_str: The component type identifier string (e.g., 'Resistor').
                                This is an explicit part of the contract, provided
                                by the CircuitBuilder.
            parameter_manager: The single, global ParameterManager for the simulation.
            parent_hierarchical_id: The FQN of the circuit containing this component.
            port_net_map: A dictionary mapping this instance's canonical port names
                          (e.g., 'p1') to the net names of the circuit it is placed
                          in (e.g., 'net_in'). This is a non-negotiable part of the
                          component contract.
        """
        self.instance_id: str = instance_id
        self.parameter_manager: ParameterManager = parameter_manager
        self.parent_hierarchical_id: str = parent_hierarchical_id

        # The port-to-net mapping is the sole source of connectivity truth.
        self._port_net_map = port_net_map

        # The component's type string is now an explicit, constructor-provided contract.
        self.component_type: str = component_type_str

        self.ureg = ureg

        # This instance-level cache is a critical performance optimization.
        # It ensures that for a given component instance, each capability object
        # is created only ONCE on its first request.
        self._capability_cache: Dict[Type[ComponentCapability], ComponentCapability] = {}
        logger.debug(f"Initialized {type(self).__name__} '{self.fqn}'")

    @property
    def fqn(self) -> str:
        """The canonical, fully qualified name (FQN) of this component instance."""
        if self.parent_hierarchical_id == "top":
            return f"top.{self.instance_id}"
        return f"{self.parent_hierarchical_id}.{self.instance_id}"

    @property
    def parameter_fqns(self) -> List[str]:
        """A list of the fully qualified names for this component's parameters."""
        return [f"{self.fqn}.{base_name}" for base_name in self.declare_parameters()]

    def get_port_net_mapping(self) -> Dict[str, str]:
        """
        Returns a dictionary mapping this instance's canonical port names
        (e.g., 'p1') to the net names of the circuit it is placed in
        (e.g., 'net_in'). This is the formal contract for retrieving
        connectivity information, populated by the CircuitBuilder.
        """
        return self._port_net_map

    @provides(IConnectivityProvider)
    class ConnectivityProvider:
        """
        Default implementation of the IConnectivityProvider capability.

        **ARCHITECTURAL REFINEMENT (Finding 2):**
        This implementation has been made stricter to enforce correctness.
        - For 2-port components, it correctly returns pairwise connectivity.
        - For components with >2 ports, it returns NO connectivity (`[]`) and logs
          an ERROR. This is a non-negotiable change that FORCES authors of N-port
          components (e.g., couplers, circulators) to provide their own, correct
          `IConnectivityProvider` capability. This prevents silent topological
          errors and upholds the "Correctness by Construction" mandate.
        """
        def get_connectivity(self, component: "ComponentBase") -> List[Tuple[str, str]]:
            ports = type(component).declare_ports()
            if len(ports) == 2:
                return [(ports[0], ports[1])]
            
            # For 0, 1, or >2 ports, return no connectivity by default.
            # This FORCES N-port component authors to be explicit.
            if len(ports) > 2:
                logger.error(
                    f"Component type '{type(component).component_type_str}' has > 2 ports but uses the default "
                    f"IConnectivityProvider, which assumes no internal connections. You MUST provide a custom "
                    f"capability implementation for correct N-port topology."
                )
            return []

    @classmethod
    def declare_capabilities(cls) -> Dict[Type[ComponentCapability], Type]:
        """
        Automatically discovers and returns the capabilities map by robustly
        inspecting the class hierarchy (MRO).

        This method fulfills the 'Declarative and Robust Registration' mandate.
        It introspects the component class's full Method Resolution Order (MRO)
        to find nested classes decorated with `@provides`. This correctly handles
        capabilities defined in parent classes, making the plugin API robust,
        predictable, and intuitive for authors.

        Returns:
            A dictionary mapping a capability Protocol (e.g., IMnaContributor) to the
            nested class that provides its implementation.
        """
        discovered_capabilities = {}
        # We iterate through the MRO to correctly handle inheritance.
        for base_class in cls.__mro__:
            for _, member_obj in inspect.getmembers(base_class):
                # Check for the magic attribute set by the @provides decorator.
                if hasattr(member_obj, '_implements_capability'):
                    protocol = member_obj._implements_capability
                    # Only add it if we haven't already found a more specific one.
                    if protocol not in discovered_capabilities:
                        discovered_capabilities[protocol] = member_obj
        return discovered_capabilities

    def get_capability(self, capability_type: Type[TCapability]) -> Optional[TCapability]:
        """
        Queries the component instance for a specific capability.

        This is the sole, public-facing entry point for all analysis engines.
        It implements a lazy-loading and caching pattern.

        Args:
            capability_type: The Protocol class representing the desired capability
                             (e.g., `IMnaContributor`).

        Returns:
            An instance of the capability implementation if supported, otherwise `None`.
        """
        if capability_type in self._capability_cache:
            return self._capability_cache[capability_type]

        declared = type(self).declare_capabilities()
        impl_class = declared.get(capability_type)

        if impl_class:
            instance = impl_class()
            self._capability_cache[capability_type] = instance
            return instance

        return None

    @classmethod
    @abstractmethod
    def declare_parameters(cls) -> Dict[str, str]:
        """Declare parameter names and their expected physical dimensions as strings."""
        pass

    @classmethod
    @abstractmethod
    def declare_ports(cls) -> List[str]:
        """
        Declare the canonical, string-based names of the component's connection ports.

        This is a non-negotiable contract that enforces the "Universal String-Based
        Identifier" mandate. Allowing mixed types (e.g., integers) is forbidden as
        it creates ambiguity and downstream complexity.

        For a standard 2-port element, this method MUST return, for example, `['p1', 'p2']`.
        """
        pass

    def __str__(self) -> str:
        return f"{type(self).__name__}('{self.fqn}')"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(fqn='{self.fqn}')"


# --- Global Component Registry and Decorator ---

COMPONENT_REGISTRY: Dict[str, type[ComponentBase]] = {}


def register_component(type_str: str):
    """
    A class decorator to register a component class in the global component registry,
    making it available to the netlist parser.
    """
    def decorator(cls: type[ComponentBase]):
        if not issubclass(cls, ComponentBase):
            raise TypeError(f"Class {cls.__name__} must inherit from ComponentBase.")

        # --- VALIDATION FOR 'declare_ports' ---
        try:
            ports = cls.declare_ports()
            if not isinstance(ports, list) or not all(isinstance(p, str) and p for p in ports):
                raise TypeError(
                    f"Component class '{cls.__name__}' violates API contract. "
                    f"declare_ports() must return a list of non-empty strings, but returned: {ports}."
                )
            # Enforce port name uniqueness at build time.
            if len(set(ports)) != len(ports):
                raise TypeError(
                    f"Component class '{cls.__name__}' violates API contract. "
                    f"declare_ports() must return a list of unique strings, but found duplicates in: {ports}."
                )
        except Exception as e:
            raise TypeError(
                f"A failure occurred while attempting to validate the API contract of "
                f"component class '{cls.__name__}'. Error during call to declare_ports(): {e}"
            ) from e

        # Enforce 'declare_parameters' contract at build time.
        try:
            params = cls.declare_parameters()
            if not isinstance(params, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in params.items()):
                 raise TypeError(
                    f"Component class '{cls.__name__}' violates API contract. "
                    f"declare_parameters() must return a Dict[str, str], but returned a value of type '{type(params).__name__}'."
                )
        except Exception as e:
            raise TypeError(
                f"A failure occurred while attempting to validate the API contract of "
                f"component class '{cls.__name__}'. Error during call to declare_parameters(): {e}"
            ) from e

        if type_str in COMPONENT_REGISTRY:
            logger.warning(f"Component type '{type_str}' is being redefined/overwritten.")
        cls.component_type_str = type_str
        COMPONENT_REGISTRY[type_str] = cls
        logger.info(f"Registered component type '{type_str}' -> {cls.__name__}")
        return cls
    return decorator