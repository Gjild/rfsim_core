# src/rfsim_core/components/base.py
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

# Assuming data_structures and units are accessible
from ..data_structures import Component as ComponentData # Alias to avoid naming conflict
from ..units import ureg, pint

Quantity = ureg.Quantity
logger = logging.getLogger(__name__)

class ComponentError(ValueError):
    """Custom exception for component-related errors."""
    pass

class ComponentBase(ABC):
    """
    Abstract base class for all circuit components.
    Defines the interface for parameter declaration, validation,
    and calculation of simulation contributions.
    """
    # Class attribute to store component type string (e.g., 'Resistor')
    # Concrete classes should override this if auto-registration is used,
    # otherwise, it's set during registration.
    component_type_str: str = "BaseComponent"

    def __init__(self, instance_id: str, component_type: str, processed_params: Dict[str, Quantity]):
        """
        Initializes the simulation-ready component instance.

        Args:
            instance_id: Unique identifier for this component instance (e.g., "R1").
            component_type: The type string of the component (e.g., "Resistor").
            processed_params: A dictionary mapping parameter names to their
                              validated pint.Quantity values.
        """
        self.instance_id = instance_id
        self.component_type = component_type # Store the type string
        self._params: Dict[str, Quantity] = processed_params
        logger.debug(f"Initialized {self.component_type} '{self.instance_id}' with params: { {k: f'{v:~P}' for k, v in processed_params.items()} }")

    @classmethod
    @abstractmethod
    def declare_parameters(cls) -> Dict[str, str]:
        """
        Declares the parameters required by this component type and their
        expected physical dimensions as strings (parsable by Pint).

        Returns:
            A dictionary mapping parameter names (str) to their expected
            dimension strings (e.g., {'resistance': 'ohm', 'length': 'm'}).
        """
        pass

    @abstractmethod
    def get_admittance(self, freq_hz: np.ndarray) -> Quantity:
        """
        Calculates the complex admittance of the component at the given frequency(ies).

        Args:
            freq_hz: The frequency or frequencies in Hertz (as float or NumPy array).
                     Passed unitless, internal calculations must handle units correctly.

        Returns:
            The calculated admittance as a pint.Quantity with dimensions of siemens
            (e.g., Siemens). The magnitude can be complex. If freq_hz is an array,
            the returned Quantity should contain a corresponding array of admittances.
        """
        pass

    def get_parameter(self, name: str) -> Quantity:
        """Retrieves a processed parameter value."""
        try:
            return self._params[name]
        except KeyError:
            raise KeyError(f"Parameter '{name}' not found for component '{self.instance_id}' ({self.component_type}).")

    def __str__(self) -> str:
        return f"{self.component_type}('{self.instance_id}')"

    def __repr__(self) -> str:
        param_str = ", ".join(f"{k}={v:~P}" for k, v in self._params.items())
        return f"{self.component_type}(id='{self.instance_id}', params=[{param_str}])"

# --- Component Registry ---
# (Could be in a separate file or here, simple dict is fine for now)
COMPONENT_REGISTRY: Dict[str, type[ComponentBase]] = {}

def register_component(type_str: str):
    """Decorator to register a component class in the registry."""
    def decorator(cls: type[ComponentBase]):
        if not issubclass(cls, ComponentBase):
            raise TypeError(f"Class {cls.__name__} must inherit from ComponentBase.")
        if type_str in COMPONENT_REGISTRY:
            logger.warning(f"Component type '{type_str}' is being redefined/overwritten.")
        cls.component_type_str = type_str # Set class attribute for potential use
        COMPONENT_REGISTRY[type_str] = cls
        logger.info(f"Registered component type '{type_str}' -> {cls.__name__}")
        return cls
    return decorator