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

    def get_mna_stamps(self, freq_hz: np.ndarray, resolved_params: Dict[str, Quantity]) -> List[StampInfo]:
        if not isinstance(freq_hz, np.ndarray):
            raise TypeError(f"{self.instance_id}: freq_hz must be a NumPy array.")
        if not isinstance(resolved_params, dict):
            raise TypeError(f"{self.instance_id}: resolved_params must be a dictionary.")

        try:
            resistance_qty = resolved_params['resistance']
        except KeyError:
            # This should ideally not happen if CircuitBuilder ensures all declared params are resolved.
            raise ComponentError(f"Parameter 'resistance' not found in resolved_params for {self.instance_id}. Available: {list(resolved_params.keys())}")

        # resistance_qty.magnitude is expected to be a NumPy array, e.g., shape (1,) if freq_hz is (1,)
        # We need the scalar resistance value for the current frequency point(s).
        # If freq_hz (and thus resistance_qty.magnitude) has shape (1,), then resistance_qty[0] gives a scalar Quantity.
        
        # Assuming freq_hz (and therefore resistance_qty.magnitude) is shape (1,) here,
        # as MnaAssembler.assemble calls with freq_hz = np.array([current_scalar_freq])
        if resistance_qty.magnitude.shape[0] != 1 :
             logger.warning(f"{self.instance_id}: Resistance quantity magnitude shape {resistance_qty.magnitude.shape} unexpected, expected (1,). Using first element.")
        
        current_R_qty_scalar = resistance_qty[0] # Get Quantity for the single frequency point

        if not np.isreal(current_R_qty_scalar.magnitude) or current_R_qty_scalar.magnitude < 0:
            raise ComponentError(f"Resistance must be real and non-negative for {self.instance_id}. Got {current_R_qty_scalar:~P}")
        
        if current_R_qty_scalar.magnitude == 0:
            logger.warning(f"Component '{self.instance_id}' has zero resistance. Treated as ideal short (large finite admittance) for F>0 analysis.")
            y_scalar_complex = LARGE_ADMITTANCE_SIEMENS + 0j
        else:
            admittance_scalar_qty = (1.0 / current_R_qty_scalar).to(ureg.siemens)
            y_scalar_complex = complex(admittance_scalar_qty.magnitude)

        # y_val_array should contain the admittance for each frequency in freq_hz.
        # Since freq_hz is (1,) here, y_val_array will be (1,).
        y_val_array = np.array([y_scalar_complex], dtype=np.complex128)
        
        num_freqs_in_stamp = y_val_array.shape[0] # Should be 1
        stamp_mag = np.zeros((num_freqs_in_stamp, 2, 2), dtype=np.complex128)
        stamp_mag[:, 0, 0] = y_val_array
        stamp_mag[:, 0, 1] = -y_val_array
        stamp_mag[:, 1, 0] = -y_val_array
        stamp_mag[:, 1, 1] = y_val_array
        
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

    def get_mna_stamps(self, freq_hz: np.ndarray, resolved_params: Dict[str, Quantity]) -> List[StampInfo]:
        if not isinstance(freq_hz, np.ndarray):
             raise TypeError(f"{self.instance_id}: freq_hz must be a NumPy array.")
        if not isinstance(resolved_params, dict):
            raise TypeError(f"{self.instance_id}: resolved_params must be a dictionary.")

        try:
            capacitance_qty = resolved_params['capacitance'] # Magnitude is shape (1,)
        except KeyError:
            raise ComponentError(f"Parameter 'capacitance' not found in resolved_params for {self.instance_id}.")

        # Validate (using the scalar value for the current frequency)
        current_C_scalar_mag = capacitance_qty.magnitude[0]
        if not np.isreal(current_C_scalar_mag) or current_C_scalar_mag < 0:
             if not np.isinf(current_C_scalar_mag):
                 raise ComponentError(f"Capacitance must be real and non-negative (or infinite) for {self.instance_id}. Got {capacitance_qty[0]:~P}")
        
        is_infinite_cap = np.isinf(current_C_scalar_mag)
        if is_infinite_cap:
            logger.warning(f"Component '{self.instance_id}' has infinite capacitance. Treated as ideal short (large finite admittance) for F>0 analysis.")
            y_val_array = np.array([LARGE_ADMITTANCE_SIEMENS + 0j], dtype=np.complex128)
        else:
            # freq_hz is (1,), capacitance_qty.magnitude is (1,)
            # Pint handles element-wise operations for Quantities with array magnitudes.
            omega = (2 * np.pi * freq_hz) * (ureg.rad / ureg.second) # omega.magnitude is (1,)
            admittance_qty = (1j * omega * capacitance_qty).to(ureg.siemens) # admittance_qty.magnitude is (1,)
            y_val_array_temp = np.asarray(admittance_qty.magnitude, dtype=np.complex128) # Shape (1,)
            
            # DC handling: freq_hz is np.array([f_scalar])
            is_dc_array = (freq_hz == 0) # is_dc_array is np.array([True/False]), shape (1,)
            y_val_array = np.where(is_dc_array, 0.0 + 0.0j, y_val_array_temp) # Shape (1,)

        num_freqs_in_stamp = y_val_array.shape[0] # Should be 1
        stamp_mag = np.zeros((num_freqs_in_stamp, 2, 2), dtype=np.complex128)
        stamp_mag[:, 0, 0] = y_val_array
        stamp_mag[:, 0, 1] = -y_val_array
        stamp_mag[:, 1, 0] = -y_val_array
        stamp_mag[:, 1, 1] = y_val_array
        
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

    def get_mna_stamps(self, freq_hz: np.ndarray, resolved_params: Dict[str, Quantity]) -> List[StampInfo]:
        if not isinstance(freq_hz, np.ndarray):
             raise TypeError(f"{self.instance_id}: freq_hz must be a NumPy array.")
        if not isinstance(resolved_params, dict):
            raise TypeError(f"{self.instance_id}: resolved_params must be a dictionary.")
        
        try:
            inductance_qty = resolved_params['inductance'] # Magnitude is shape (1,)
        except KeyError:
            raise ComponentError(f"Parameter 'inductance' not found in resolved_params for {self.instance_id}.")

        current_L_scalar_mag = inductance_qty.magnitude[0]
        if not np.isreal(current_L_scalar_mag) or current_L_scalar_mag < 0:
             raise ComponentError(f"Inductance must be real and non-negative for {self.instance_id}. Got {inductance_qty[0]:~P}")
        
        is_zero_inductance = (current_L_scalar_mag == 0)
        if is_zero_inductance:
             logger.warning(f"Component '{self.instance_id}' has zero inductance. Treated as ideal short (large finite admittance) for F>=0 analysis.")
             y_val_array = np.array([LARGE_ADMITTANCE_SIEMENS + 0j], dtype=np.complex128)
        else:
            omega = (2 * np.pi * freq_hz) * (ureg.rad / ureg.second) # omega.magnitude is (1,)
            impedance_qty = (1j * omega * inductance_qty) # impedance_qty.magnitude is (1,)
            
            # Handle division by zero for DC case (impedance is zero)
            # Pint's 1.0 / impedance_qty might handle units correctly.
            # We need to be careful with magnitudes.
            # impedance_qty.magnitude will be array like [0j] at DC.
            
            y_val_list = []
            for i in range(freq_hz.shape[0]): # Loop over frequencies (here, just 1)
                imp_scalar_qty_for_freq_i = impedance_qty[i]
                if np.abs(imp_scalar_qty_for_freq_i.magnitude) < 1e-18: # Effectively zero impedance (DC or L=0 at AC)
                    y_val_list.append(LARGE_ADMITTANCE_SIEMENS + 0j)
                else:
                    adm_scalar_qty_for_freq_i = (1.0 / imp_scalar_qty_for_freq_i).to(ureg.siemens)
                    y_val_list.append(adm_scalar_qty_for_freq_i.magnitude)
            
            y_val_array = np.array(y_val_list, dtype=np.complex128) # Shape (1,)

        num_freqs_in_stamp = y_val_array.shape[0] # Should be 1
        stamp_mag = np.zeros((num_freqs_in_stamp, 2, 2), dtype=np.complex128)
        stamp_mag[:, 0, 0] = y_val_array
        stamp_mag[:, 0, 1] = -y_val_array
        stamp_mag[:, 1, 0] = -y_val_array
        stamp_mag[:, 1, 1] = y_val_array
        
        admittance_matrix_qty = Quantity(stamp_mag, ureg.siemens)
        return [(admittance_matrix_qty, [PORT_1, PORT_2])]