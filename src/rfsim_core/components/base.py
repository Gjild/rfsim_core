# --- src/rfsim_core/components/base.py ---
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, ClassVar, Optional
from enum import Enum, auto
import numpy as np

from ..units import ureg, pint, Quantity
from ..parameters import ParameterManager # For type hinting

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
        component_type: str,
        parameter_manager: Optional[ParameterManager], # Remains Optional
        parameter_internal_names: List[str]
    ):
        self.instance_id = instance_id
        self.component_type = component_type
        self.parameter_manager: Optional[ParameterManager] = parameter_manager
        self.parameter_internal_names = parameter_internal_names
        self._ureg = ureg
        logger.debug(f"Initialized {self.component_type} '{self.instance_id}' with param names: {self.parameter_internal_names}")

    @classmethod
    @abstractmethod
    def declare_parameters(cls) -> Dict[str, str]:
        pass

    @classmethod
    @abstractmethod
    def declare_ports(cls) -> List[str | int]:
        pass

    @classmethod
    def declare_connectivity(cls) -> List[Tuple[str | int, str | int]]:
        ports = cls.declare_ports()
        if len(ports) == 2:
            return [(ports[0], ports[1])]
        elif len(ports) < 2:
             return []
        else:
             logger.warning(f"Component type '{cls.component_type_str}' has > 2 ports ({ports}) but uses default pairwise connectivity declaration. Override declare_connectivity() for accurate sparsity.")
             from itertools import combinations
             return list(combinations(ports, 2)) # type: ignore

    @abstractmethod
    def get_mna_stamps(self, freq_hz: np.ndarray, resolved_params: Dict[str, Quantity], current_sweep_idx: Optional[int] = None) -> List[StampInfo]: # Added current_sweep_idx
        pass

    def __str__(self) -> str:
        return f"{self.component_type}('{self.instance_id}')"

    def __repr__(self) -> str:
        param_names_str = ", ".join(self.parameter_internal_names)
        return f"{self.component_type}(id='{self.instance_id}', params_managed=[{param_names_str}])"

    @abstractmethod
    def get_dc_behavior(self, resolved_params: Dict[str, Quantity]) -> Tuple[DCBehaviorType, Optional[Quantity]]:
        pass

    @abstractmethod
    def is_structurally_open(self, resolved_constant_params: Dict[str, Quantity]) -> bool:
        pass

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