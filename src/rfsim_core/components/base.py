# src/rfsim_core/components/base.py
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, ClassVar, Optional
from enum import Enum, auto
import numpy as np

from ..units import ureg, Quantity
from ..parameters import ParameterManager
from ..parser.raw_data import ParsedComponentData

logger = logging.getLogger(__name__)


class ComponentError(ValueError):
    """Custom exception for component-related errors."""
    pass


StampInfo = Tuple[Quantity, List[str | int]]


class DCBehaviorType(Enum):
    SHORT_CIRCUIT = auto()
    OPEN_CIRCUIT = auto()
    ADMITTANCE = auto()


class ComponentBase(ABC):
    component_type_str: ClassVar[str] = "BaseComponent"

    def __init__(
        self,
        instance_id: str,
        parameter_manager: ParameterManager,
        parent_hierarchical_id: str,
        raw_ir_data: ParsedComponentData
    ):
        self.instance_id: str = instance_id
        self.parameter_manager: ParameterManager = parameter_manager
        self.parent_hierarchical_id: str = parent_hierarchical_id
        self.raw_ir_data: ParsedComponentData = raw_ir_data
        self._ureg = ureg
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
        """Declare internal connectivity between ports for MNA sparsity pattern prediction."""
        ports = cls.declare_ports()
        if len(ports) == 2:
            return [(ports[0], ports[1])]
        elif len(ports) < 2:
            return []
        else:
            logger.warning(f"Component type '{cls.component_type_str}' has > 2 ports ({ports}) but uses default pairwise connectivity. Override declare_connectivity() for accurate sparsity.")
            from itertools import combinations
            return list(combinations(ports, 2))

    @abstractmethod
    def get_mna_stamps(self, freq_hz_array: np.ndarray, all_evaluated_params: Dict[str, Quantity]) -> List[StampInfo]:
        """
        Calculate MNA matrix contributions for a full frequency sweep. This method MUST
        be vectorized.

        Args:
            freq_hz_array: The NumPy array of all simulation frequencies in Hz.
            all_evaluated_params: A dictionary mapping every FQN in the circuit to its
                                  vectorized, resolved `pint.Quantity` object. The component
                                  is responsible for picking the parameters it needs by FQN.

        Returns:
            A list of StampInfo tuples. The Quantity in each tuple MUST be a vectorized
            admittance matrix, e.g., with shape (num_frequencies, num_ports, num_ports).
        """
        pass

    @abstractmethod
    def get_dc_behavior(self, all_dc_params: Dict[str, Quantity]) -> Tuple[DCBehaviorType, Optional[Quantity]]:
        """
        Determine the component's behavior at DC (F=0).

        Args:
            all_dc_params: A dictionary mapping every FQN in the circuit to its
                           scalar `pint.Quantity` object evaluated at F=0. The component
                           is responsible for picking the parameters it needs by FQN.

        Returns:
            A tuple describing the DC behavior.
        """
        pass

    @abstractmethod
    def is_structurally_open(self, resolved_constant_params: Dict[str, Quantity]) -> bool:
        """
        Determine if the component is a structural open based on constant parameters.
        """
        pass

    def __str__(self) -> str:
        return f"{type(self).__name__}('{self.fqn}')"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(fqn='{self.fqn}')"


COMPONENT_REGISTRY: Dict[str, type[ComponentBase]] = {}
def register_component(type_str: str):
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