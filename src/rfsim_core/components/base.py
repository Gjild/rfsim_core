# src/rfsim_core/components/base.py

import logging
import inspect  # NEW: For robust class hierarchy inspection.
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, ClassVar, Optional, Type

import numpy as np

from ..units import ureg, Quantity
from ..parameters import ParameterManager
from ..parser.raw_data import ParsedComponentData
# --- NEW IMPORTS: The core of the capability system ---
from .capabilities import ComponentCapability, TCapability, IMnaContributor, IDcContributor
from .base_enums import DCBehaviorType  # Import from the new, separated file.


logger = logging.getLogger(__name__)


# A type alias for MNA stamp information, retained for clarity.
StampInfo = Tuple[Quantity, List[str | int]]


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
        parameter_manager: ParameterManager,
        parent_hierarchical_id: str,
        raw_ir_data: ParsedComponentData
    ):
        """
        Initializes the base attributes of a component instance.

        Args:
            instance_id: The unique ID of this component instance (e.g., 'R1').
            parameter_manager: The single, global ParameterManager for the simulation.
            parent_hierarchical_id: The FQN of the circuit containing this component.
            raw_ir_data: A link to the raw, parsed data for this specific instance,
                         essential for validation and diagnostics.
        """
        self.instance_id: str = instance_id
        self.parameter_manager: ParameterManager = parameter_manager
        self.parent_hierarchical_id: str = parent_hierarchical_id
        self.raw_ir_data: ParsedComponentData = raw_ir_data

        # --- MODIFICATION: Fulfill the instance attribute contract ---
        # This makes the component's type string directly available on the instance,
        # creating an explicit contract for validators and other tools.
        self.component_type: str = raw_ir_data.component_type
        # --- END OF MODIFICATION ---

        self.ureg = ureg

        # NEW: This instance-level cache is a critical performance optimization.
        # It ensures that for a given component instance, each capability object
        # is created only ONCE on its first request. This amortizes the cost
        # of discovery and instantiation across the entire simulation.
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

    # --- DELETED: The old, monolithic abstract methods have been removed. ---
    # @abstractmethod
    # def get_mna_stamps(...) -> List[StampInfo]: ...
    # @abstractmethod
    # def get_dc_behavior(...) -> Tuple[DCBehaviorType, Optional[Quantity]]: ...

    # --- NEW: The core methods of the capability system. ---

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
        # We iterate through the MRO to correctly handle inheritance. The MRO
        # is ordered from the class itself to its parents, so the first
        # implementation found for a given protocol is the most specific one.
        for base_class in cls.__mro__:
            # Use inspect.getmembers to find all members of the class.
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
        It implements a lazy-loading and caching pattern:
        1. Checks the instance-level cache for an existing capability object.
        2. If not found, it calls the class-level discovery method.
        3. If declared, it instantiates the stateless capability implementation class.
        4. The new instance is cached for future requests and returned.

        Args:
            capability_type: The Protocol class representing the desired capability
                             (e.g., `IMnaContributor`).

        Returns:
            An instance of the capability implementation if supported, otherwise `None`.
        """
        # 1. Check instance cache first for maximum performance.
        if capability_type in self._capability_cache:
            return self._capability_cache[capability_type]

        # 2. On cache miss, consult the class-level declaration via discovery.
        declared = type(self).declare_capabilities()
        impl_class = declared.get(capability_type)

        if impl_class:
            # 3. Instantiate the stateless capability object.
            # No arguments are passed; it's a stateless service object.
            instance = impl_class()

            # 4. Cache the singleton instance and return it.
            self._capability_cache[capability_type] = instance
            return instance

        # Return None if the component does not support this capability.
        return None

    # --- Abstract methods for core component definition (REMAINING) ---

    @classmethod
    @abstractmethod
    def declare_parameters(cls) -> Dict[str, str]:
        """Declare parameter names and their expected physical dimensions as strings."""
        pass

    @classmethod
    @abstractmethod
    def declare_ports(cls) -> List[str | int]:
        """Declare the names/indices of the component's connection ports."""
        pass

    @classmethod
    def declare_connectivity(cls) -> List[Tuple[str | int, str | int]]:
        """
        Declare internal connectivity between ports for MNA sparsity pattern prediction.
        The default implementation assumes full connectivity for components with >2 ports.
        Override for performance with sparsely-connected multi-port components.
        """
        ports = cls.declare_ports()
        if len(ports) == 2:
            return [(ports[0], ports[1])]
        elif len(ports) < 2:
            return []
        else:
            logger.warning(
                f"Component type '{cls.component_type_str}' has > 2 ports ({ports}) but "
                f"uses default pairwise connectivity. Override declare_connectivity() "
                f"for accurate sparsity."
            )
            from itertools import combinations
            return list(combinations(ports, 2))

    @abstractmethod
    def is_structurally_open(self, resolved_constant_params: Dict[str, Quantity]) -> bool:
        """
        Determine if the component is a structural open based on constant parameters.
        This is used for pre-simulation topological analysis to identify and remove
        floating sub-circuits, improving performance and robustness.

        Args:
            resolved_constant_params: A dictionary of this component's constant
                                      parameter values, already resolved.

        Returns:
            True if the component acts as a permanent open circuit, False otherwise.
        """
        pass

    # --- Concrete dunder methods for representation (REMAINING) ---

    def __str__(self) -> str:
        return f"{type(self).__name__}('{self.fqn}')"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(fqn='{self.fqn}')"


# --- Global Component Registry and Decorator (REMAINING) ---

COMPONENT_REGISTRY: Dict[str, type[ComponentBase]] = {}

def register_component(type_str: str):
    """
    A class decorator to register a component class in the global component registry,
    making it available to the netlist parser.
    """
    def decorator(cls: type[ComponentBase]):
        if not issubclass(cls, ComponentBase):
            raise TypeError(f"Class {cls.__name__} must inherit from ComponentBase.")
        if type_str in COMPONENT_REGISTRY:
            logger.warning(f"Component type '{type_str}' is being redefined/overwritten.")
        cls.component_type_str = type_str
        COMPONENT_REGISTRY[type_str] = cls
        logger.info(f"Registered component type '{type_str}' -> {cls.__name__}")
        return cls
    return decorator