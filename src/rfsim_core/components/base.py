# --- src/rfsim_core/components/base.py ---
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, ClassVar
import numpy as np

from ..units import ureg, pint, Quantity
# Import ParameterManager for type hinting __init__
from ..parameters import ParameterManager

logger = logging.getLogger(__name__)

# Define a large admittance value to represent ideal shorts numerically (for F>0)
# Accessible via rfsim_core.components.LARGE_ADMITTANCE_SIEMENS
LARGE_ADMITTANCE_SIEMENS = 1e12 # Siemens (1 / microOhm)

class ComponentError(ValueError):
    """Custom exception for component-related errors."""
    pass

# Define a type alias for the return type of get_mna_stamps
# Tuple[AdmittanceMatrixQuantity, ListOfPortIDs]
# AdmittanceMatrixQuantity will be a Quantity whose magnitude is a complex numpy array
# For a 2-port, shape might be (num_freqs, 2, 2) or just (2, 2) if freq is scalar
StampInfo = Tuple[Quantity, List[str | int]]

class ComponentBase(ABC):
    """
    Abstract base class for all circuit components.
    Stores reference to ParameterManager and its own parameter internal names.
    """
    component_type_str: ClassVar[str] = "BaseComponent"

    def __init__(
        self,
        instance_id: str,
        component_type: str,
        parameter_manager: ParameterManager,
        parameter_internal_names: List[str]
    ):
        """
        Initializes the simulation-ready component instance.

        Args:
            instance_id: Unique identifier for this component instance.
            component_type: String identifier (e.g., "Resistor").
            parameter_manager: Reference to the built ParameterManager.
            parameter_internal_names: List of fully qualified internal names
                                      (e.g., ['R1.resistance']) relevant to this instance.
        """
        self.instance_id = instance_id
        self.component_type = component_type
        self.parameter_manager = parameter_manager
        self.parameter_internal_names = parameter_internal_names
        self._ureg = ureg
        logger.debug(f"Initialized {self.component_type} '{self.instance_id}' with param names: {self.parameter_internal_names}")

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

    @classmethod
    @abstractmethod
    def declare_ports(cls) -> List[str | int]:
        """
        Declares the mandatory port identifiers (names or numbers) for this component type.
        Used for validation during circuit building.
        """
        pass

    @classmethod
    def declare_connectivity(cls) -> List[Tuple[str | int, str | int]]:
        """
        Declares pairs of port IDs that have direct admittance paths between them,
        used for calculating the MNA matrix sparsity pattern *before* simulation.
        Return list of tuples, e.g., [(port1, port2), (port2, port3)] for a 3-terminal device.
        For a simple 2-terminal device, return e.g., [(0, 1)].
        """
        # Default implementation assumes pairwise connection between all declared ports
        # Subclasses with more complex internal structures should override this.
        ports = cls.declare_ports()
        if len(ports) == 2:
            return [(ports[0], ports[1])]
        elif len(ports) < 2:
             return []
        else:
             # Warning: Default pairwise assumption might add too many non-zeros
             # for components like transformers or ideal op-amps.
             logger.warning(f"Component type '{cls.component_type_str}' has > 2 ports ({ports}) but uses default pairwise connectivity declaration. Override declare_connectivity() for accurate sparsity.")
             from itertools import combinations
             return list(combinations(ports, 2)) # type: ignore

    @abstractmethod
    def get_mna_stamps(self, freq_hz: np.ndarray, resolved_params: Dict[str, Quantity]) -> List[StampInfo]: # MODIFIED SIGNATURE
        """
        Calculates the MNA contribution(s) of the component at the given frequency(ies).
        Implementations use the 'resolved_params' dictionary to get their parameter values as Quantities.
        The 'freq_hz' argument will be a 1D NumPy array, typically containing a single frequency
        when called from MnaAssembler.assemble().
        Magnitudes of Quantities in 'resolved_params' and of 'freq_hz' will typically be 1-element arrays.
        The returned AdmittanceMatrixQuantity.magnitude should be shaped (num_freqs, N, N),
        e.g., (1, N, N) if freq_hz is (1,).

        Args:
            freq_hz: NumPy array of frequencies in Hertz (unitless). Shape (num_eval_freqs,).
            resolved_params: Dictionary mapping base parameter names (e.g., "resistance")
                             to their resolved pint.Quantity objects for the given frequency(ies).
                             The magnitude of these Quantities will be NumPy arrays, typically
                             matching the shape of freq_hz.

        Returns:
            A list of StampInfo tuples representing the component's contributions.

        Raises:
            ComponentError: If calculation fails.
            TypeError: If freq_hz is not a NumPy array or resolved_params is incorrect.
            pint.DimensionalityError: If internal calculations produce wrong dimensions or
                                    the returned Quantity is not [admittance].
            ParameterError: If parameter resolution fails (though most resolution errors
                            should be caught before this point by MnaAssembler).
        """
        pass

    def __str__(self) -> str:
        return f"{self.component_type}('{self.instance_id}')"

    def __repr__(self) -> str:
        param_names_str = ", ".join(self.parameter_internal_names)
        return f"{self.component_type}(id='{self.instance_id}', params_managed=[{param_names_str}])"

# --- Component Registry ---
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