# --- src/rfsim_core/components/elements.py ---
import logging
import numpy as np

from typing import Dict, List
from .base import (
    ComponentBase, register_component, ComponentError, StampInfo,
    LARGE_ADMITTANCE_SIEMENS
)
from ..units import ureg, pint, Quantity
# Import for type hints and temporary shim
from ..parameters import ParameterManager, ParameterError

logger = logging.getLogger(__name__)

# Standard port names/indices for simple 2-terminal elements
PORT_1 = 0
PORT_2 = 1  

@register_component("Resistor")
class Resistor(ComponentBase):
    """Ideal linear resistor."""

    @classmethod
    def declare_parameters(cls) -> Dict[str, str]:
        return {"resistance": "ohm"}
    
    @classmethod
    def declare_ports(cls) -> List[str | int]:
        return [PORT_1, PORT_2]

    def __init__(
        self,
        instance_id: str,
        component_type: str,
        parameter_manager: ParameterManager,
        parameter_internal_names: List[str]
    ):
        super().__init__(instance_id, component_type, parameter_manager, parameter_internal_names)
        # Parameter validation (e.g., non-negative) might move to get_mna_stamps after resolution

    def get_mna_stamps(self, freq_hz: np.ndarray) -> List[StampInfo]:
        """Calculates the 2x2 admittance matrix stamp."""
        if not isinstance(freq_hz, np.ndarray):
            raise TypeError(f"{self.instance_id}: freq_hz must be a NumPy array.")

        # --- TEMPORARY SHIM for Task 5.1 completion ---
        # Assumes 'resistance' is defined as a constant for now.
        # This will be replaced by resolve_parameter in Task 5.4.
        resistance_internal_name = f"{self.instance_id}.resistance"
        try:
            # Use get_constant_value which was populated during PM build
            resistance = self.parameter_manager.get_constant_value(resistance_internal_name)
        except ParameterError as e:
            raise ComponentError(f"Failed to get constant resistance for {self.instance_id}: {e}. Is it defined as an expression?") from e
        except KeyError:
             raise ComponentError(f"Internal Error: Constant resistance '{resistance_internal_name}' not found for {self.instance_id}.")

        # Validate value *after* retrieval (optional, could wait for expression support)
        if not np.isreal(resistance.magnitude) or resistance.magnitude < 0:
            raise ComponentError(f"Resistance must be real and non-negative for {self.instance_id}. Got {resistance:~P}")
        if resistance.magnitude == 0:
            logger.warning(f"Component '{self.instance_id}' has zero resistance. Treated as ideal short (large finite admittance) for F>0 analysis.")
        # --- End Temporary Shim ---

        R_mag = resistance.magnitude

        if R_mag == 0:
            y_val = LARGE_ADMITTANCE_SIEMENS + 0j
        else:
            y_scalar_qty = (1.0 / resistance).to(ureg.siemens)
            y_val = complex(y_scalar_qty.magnitude)

        num_freqs = len(freq_hz) if freq_hz.ndim > 0 and freq_hz.size > 0 else 1
        if num_freqs > 1:
             stamp_mag = np.zeros((num_freqs, 2, 2), dtype=np.complex128)
             stamp_mag[:, 0, 0] = y_val
             stamp_mag[:, 0, 1] = -y_val
             stamp_mag[:, 1, 0] = -y_val
             stamp_mag[:, 1, 1] = y_val
        elif num_freqs == 1 and freq_hz.size > 0:
            stamp_mag = np.array([[y_val, -y_val], [-y_val, y_val]], dtype=np.complex128)
        else:
             stamp_mag = np.empty((0, 2, 2), dtype=np.complex128)

        admittance_matrix_qty = Quantity(stamp_mag, ureg.siemens)
        return [(admittance_matrix_qty, [PORT_1, PORT_2])]


@register_component("Capacitor")
class Capacitor(ComponentBase):
    """Ideal linear capacitor."""

    @classmethod
    def declare_parameters(cls) -> Dict[str, str]:
        return {"capacitance": "farad"}

    @classmethod
    def declare_ports(cls) -> List[str | int]:
        return [PORT_1, PORT_2]

    # Updated __init__
    def __init__(
        self,
        instance_id: str,
        component_type: str,
        parameter_manager: ParameterManager,
        parameter_internal_names: List[str]
    ):
        super().__init__(instance_id, component_type, parameter_manager, parameter_internal_names)
        # No longer store self.capacitance

    def get_mna_stamps(self, freq_hz: np.ndarray) -> List[StampInfo]:
        if not isinstance(freq_hz, np.ndarray):
             raise TypeError(f"{self.instance_id}: freq_hz must be a NumPy array.")

        # --- TEMPORARY SHIM ---
        capacitance_internal_name = f"{self.instance_id}.capacitance"
        try:
            capacitance = self.parameter_manager.get_constant_value(capacitance_internal_name)
        except ParameterError as e:
            raise ComponentError(f"Failed to get constant capacitance for {self.instance_id}: {e}. Is it defined as an expression?") from e
        except KeyError:
             raise ComponentError(f"Internal Error: Constant capacitance '{capacitance_internal_name}' not found for {self.instance_id}.")

        # Validate
        if not np.isreal(capacitance.magnitude) or capacitance.magnitude < 0:
             if not np.isinf(capacitance.magnitude):
                 raise ComponentError(f"Capacitance must be real and non-negative (or infinite) for {self.instance_id}. Got {capacitance:~P}")
        if np.isinf(capacitance.magnitude):
            logger.warning(f"Component '{self.instance_id}' has infinite capacitance. Treated as ideal short (large finite admittance) for F>0 analysis.")
        # --- End Shim ---

        C_mag = capacitance.magnitude
        y_val_array: np.ndarray
        is_dc = (freq_hz == 0)
        is_infinite_cap = np.isinf(C_mag)

        if is_infinite_cap:
            y_scalar = LARGE_ADMITTANCE_SIEMENS + 0j
            y_val_array = np.full_like(freq_hz, y_scalar, dtype=np.complex128)
        else:
            omega = (2 * np.pi * freq_hz) * (ureg.rad / ureg.second)
            y_qty = (1j * omega * capacitance).to(ureg.siemens) # Use retrieved capacitance
            y_val_array_temp = np.asarray(y_qty.magnitude, dtype=np.complex128)
            if np.any(is_dc):
                y_val_array = np.where(is_dc, 0.0 + 0.0j, y_val_array_temp)
            else:
                y_val_array = y_val_array_temp

        if y_val_array.ndim > 0 and y_val_array.size > 0:
            num_freqs = len(y_val_array)
            stamp_mag = np.zeros((num_freqs, 2, 2), dtype=np.complex128)
            stamp_mag[:, 0, 0] = y_val_array
            stamp_mag[:, 0, 1] = -y_val_array
            stamp_mag[:, 1, 0] = -y_val_array
            stamp_mag[:, 1, 1] = y_val_array
        elif y_val_array.ndim == 0 and y_val_array.size == 1:
             y_scalar = complex(y_val_array)
             stamp_mag = np.array([[y_scalar, -y_scalar], [-y_scalar, y_scalar]], dtype=np.complex128)
        else:
             stamp_mag = np.empty((0, 2, 2), dtype=np.complex128)

        admittance_matrix_qty = Quantity(stamp_mag, ureg.siemens)
        return [(admittance_matrix_qty, [PORT_1, PORT_2])]


@register_component("Inductor")
class Inductor(ComponentBase):
    """Ideal linear inductor."""

    @classmethod
    def declare_parameters(cls) -> Dict[str, str]:
        return {"inductance": "henry"}

    @classmethod
    def declare_ports(cls) -> List[str | int]:
        return [PORT_1, PORT_2]

    # Updated __init__
    def __init__(
        self,
        instance_id: str,
        component_type: str,
        parameter_manager: ParameterManager,
        parameter_internal_names: List[str]
    ):
        super().__init__(instance_id, component_type, parameter_manager, parameter_internal_names)
        # No longer store self.inductance

    def get_mna_stamps(self, freq_hz: np.ndarray) -> List[StampInfo]:
        if not isinstance(freq_hz, np.ndarray):
             raise TypeError(f"{self.instance_id}: freq_hz must be a NumPy array.")

        # --- TEMPORARY SHIM ---
        inductance_internal_name = f"{self.instance_id}.inductance"
        try:
            inductance = self.parameter_manager.get_constant_value(inductance_internal_name)
        except ParameterError as e:
            raise ComponentError(f"Failed to get constant inductance for {self.instance_id}: {e}. Is it defined as an expression?") from e
        except KeyError:
             raise ComponentError(f"Internal Error: Constant inductance '{inductance_internal_name}' not found for {self.instance_id}.")

        # Validate
        if not np.isreal(inductance.magnitude) or inductance.magnitude < 0:
             raise ComponentError(f"Inductance must be real and non-negative for {self.instance_id}. Got {inductance:~P}")
        if inductance.magnitude == 0:
             logger.warning(f"Component '{self.instance_id}' has zero inductance. Treated as ideal short (large finite admittance) for F>=0 analysis.")
        # --- End Shim ---

        L_mag = inductance.magnitude
        y_val_array: np.ndarray
        is_dc = (freq_hz == 0)
        is_zero_inductance = (L_mag == 0)

        if is_zero_inductance:
            y_scalar = LARGE_ADMITTANCE_SIEMENS + 0j
            y_val_array = np.full_like(freq_hz, y_scalar, dtype=np.complex128)
        else:
             omega = (2 * np.pi * freq_hz) * (ureg.rad / ureg.second)
             impedance = (1j * omega * inductance) # Use retrieved inductance

             with np.errstate(divide='ignore', invalid='ignore'):
                 y_qty = (1.0 / impedance).to(ureg.siemens)
                 y_val_array_temp = np.asarray(y_qty.magnitude, dtype=np.complex128)

             if np.any(is_dc):
                 y_scalar_dc = LARGE_ADMITTANCE_SIEMENS + 0j
                 y_val_array = np.where(is_dc, y_scalar_dc, y_val_array_temp)
                 y_val_array = np.nan_to_num(y_val_array, nan=y_scalar_dc, posinf=y_scalar_dc, neginf=y_scalar_dc)
             else:
                 y_val_array = y_val_array_temp

        if y_val_array.ndim > 0 and y_val_array.size > 0:
            num_freqs = len(y_val_array)
            stamp_mag = np.zeros((num_freqs, 2, 2), dtype=np.complex128)
            stamp_mag[:, 0, 0] = y_val_array
            stamp_mag[:, 0, 1] = -y_val_array
            stamp_mag[:, 1, 0] = -y_val_array
            stamp_mag[:, 1, 1] = y_val_array
        elif y_val_array.ndim == 0 and y_val_array.size == 1:
             y_scalar = complex(y_val_array)
             stamp_mag = np.array([[y_scalar, -y_scalar], [-y_scalar, y_scalar]], dtype=np.complex128)
        else:
             stamp_mag = np.empty((0, 2, 2), dtype=np.complex128)

        admittance_matrix_qty = Quantity(stamp_mag, ureg.siemens)
        return [(admittance_matrix_qty, [PORT_1, PORT_2])]