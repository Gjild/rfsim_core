# src/rfsim_core/components/elements.py
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

from .base import (
    ComponentBase, register_component, ComponentError, StampInfo,
    DCBehaviorType
)
from ..constants import LARGE_ADMITTANCE_SIEMENS
from ..units import ureg, Quantity

logger = logging.getLogger(__name__)

PORT_1 = 0
PORT_2 = 1

def _extract_dc_real_scalar_value(
    all_dc_params: Dict[str, Quantity],
    param_fqn: str,
    component_fqn: str
) -> float:
    """Extracts the scalar real value from a resolved DC parameter Quantity."""
    try:
        # DC evaluation returns a Quantity with a 1-element array, so we extract the scalar.
        qty = all_dc_params[param_fqn][0]
        mag = qty.magnitude
        
        if np.iscomplexobj(mag) and mag.imag != 0:
            raise ComponentError(f"Parameter '{param_fqn}' for DC analysis for '{component_fqn}' must be real. Got {qty:~P}.")
        
        real_val = float(mag.real if np.iscomplexobj(mag) else mag)
        return real_val
    except KeyError:
        raise ComponentError(f"Parameter '{param_fqn}' not found for DC analysis of {component_fqn}.")
    except (TypeError, ValueError) as e:
        raise ComponentError(f"Could not convert DC parameter '{param_fqn}' of '{component_fqn}' to a scalar float: {e}") from e


@register_component("Resistor")
class Resistor(ComponentBase):
    @classmethod
    def declare_parameters(cls) -> Dict[str, str]:
        return {"resistance": "ohm"}

    @classmethod
    def declare_ports(cls) -> List[str | int]:
        return [PORT_1, PORT_2]

    def get_mna_stamps(self, freq_hz_array: np.ndarray, all_evaluated_params: Dict[str, Quantity]) -> List[StampInfo]:
        r_qty = all_evaluated_params[self.parameter_fqns[0]]
        r_mag_si = r_qty.to(ureg.ohm).magnitude
        
        if np.any(np.iscomplex(r_mag_si)) and np.any(r_mag_si.imag != 0):
            raise ComponentError(f"Resistance must be real for {self.fqn}. Got a complex value.")
        
        r_real_mag = r_mag_si.real if np.any(np.iscomplex(r_mag_si)) else r_mag_si
        if np.any(r_real_mag < 0):
            raise ComponentError(f"Resistance must be non-negative for {self.fqn}. Got a negative value.")

        y_mag_si = np.empty_like(r_real_mag, dtype=np.complex128)
        
        # Handle non-ideal case first for numerical stability
        non_ideal_mask = (r_real_mag > 0) & np.isfinite(r_real_mag)
        y_mag_si[non_ideal_mask] = 1.0 / r_real_mag[non_ideal_mask]
        
        # Patch ideal cases
        y_mag_si[r_real_mag == 0] = LARGE_ADMITTANCE_SIEMENS
        y_mag_si[np.isposinf(r_real_mag)] = 0.0

        num_freqs = len(freq_hz_array)
        stamp_mag = np.zeros((num_freqs, 2, 2), dtype=np.complex128)
        stamp_mag[:, 0, 0] = y_mag_si
        stamp_mag[:, 1, 1] = y_mag_si
        stamp_mag[:, 0, 1] = -y_mag_si
        stamp_mag[:, 1, 0] = -y_mag_si
        
        return [(Quantity(stamp_mag, ureg.siemens), [PORT_1, PORT_2])]

    def get_dc_behavior(self, all_dc_params: Dict[str, Quantity]) -> Tuple[DCBehaviorType, Optional[Quantity]]:
        param_fqn = self.parameter_fqns[0]
        r_real_val = _extract_dc_real_scalar_value(all_dc_params, param_fqn, self.fqn)
        
        if np.isnan(r_real_val) or np.isposinf(r_real_val):
            return (DCBehaviorType.OPEN_CIRCUIT, None)
        if r_real_val == 0.0:
            return (DCBehaviorType.SHORT_CIRCUIT, None)
        if r_real_val < 0:
            raise ComponentError(f"Negative resistance R={all_dc_params[param_fqn][0]:~P} not supported for DC analysis for '{self.fqn}'.")
        
        return (DCBehaviorType.ADMITTANCE, (1.0 / all_dc_params[param_fqn][0]).to(ureg.siemens))

    def is_structurally_open(self, resolved_constant_params: Dict[str, Quantity]) -> bool:
        if 'resistance' not in resolved_constant_params: return False
        return bool(np.isposinf(resolved_constant_params['resistance'].magnitude))


@register_component("Capacitor")
class Capacitor(ComponentBase):
    @classmethod
    def declare_parameters(cls) -> Dict[str, str]:
        return {"capacitance": "farad"}

    @classmethod
    def declare_ports(cls) -> List[str | int]:
        return [PORT_1, PORT_2]

    def get_mna_stamps(self, freq_hz_array: np.ndarray, all_evaluated_params: Dict[str, Quantity]) -> List[StampInfo]:
        c_qty = all_evaluated_params[self.parameter_fqns[0]]
        
        if np.any(np.iscomplex(c_qty.magnitude)):
            raise ComponentError(f"Capacitance must be real for {self.fqn}.")
        if np.any(c_qty.magnitude < 0):
            raise ComponentError(f"Capacitance must be non-negative for {self.fqn}.")

        omega = (2 * np.pi * freq_hz_array) * (ureg.rad / ureg.second)
        y_mag_si = (1j * omega * c_qty).to(ureg.siemens).magnitude

        # Patch ideal cases
        c_mag = c_qty.magnitude
        y_mag_si = np.where(np.isposinf(c_mag), LARGE_ADMITTANCE_SIEMENS + 0j, y_mag_si)
        y_mag_si = np.where(c_mag == 0, 0j, y_mag_si)

        num_freqs = len(freq_hz_array)
        stamp_mag = np.zeros((num_freqs, 2, 2), dtype=np.complex128)
        stamp_mag[:, 0, 0] = y_mag_si
        stamp_mag[:, 1, 1] = y_mag_si
        stamp_mag[:, 0, 1] = -y_mag_si
        stamp_mag[:, 1, 0] = -y_mag_si

        return [(Quantity(stamp_mag, ureg.siemens), [PORT_1, PORT_2])]

    def get_dc_behavior(self, all_dc_params: Dict[str, Quantity]) -> Tuple[DCBehaviorType, Optional[Quantity]]:
        param_fqn = self.parameter_fqns[0]
        c_real_val = _extract_dc_real_scalar_value(all_dc_params, param_fqn, self.fqn)
        
        if np.isposinf(c_real_val):
            return (DCBehaviorType.SHORT_CIRCUIT, None)
        if c_real_val < 0:
            raise ComponentError(f"Negative capacitance C={all_dc_params[param_fqn][0]:~P} not supported for DC analysis for '{self.fqn}'.")
        
        # C=0, finite C > 0, or C=NaN are all open circuits at DC
        return (DCBehaviorType.OPEN_CIRCUIT, None)

    def is_structurally_open(self, resolved_constant_params: Dict[str, Quantity]) -> bool:
        if 'capacitance' not in resolved_constant_params: return False
        return resolved_constant_params['capacitance'].magnitude == 0


@register_component("Inductor")
class Inductor(ComponentBase):
    @classmethod
    def declare_parameters(cls) -> Dict[str, str]:
        return {"inductance": "henry"}

    @classmethod
    def declare_ports(cls) -> List[str | int]:
        return [PORT_1, PORT_2]

    def get_mna_stamps(self, freq_hz_array: np.ndarray, all_evaluated_params: Dict[str, Quantity]) -> List[StampInfo]:
        l_qty = all_evaluated_params[self.parameter_fqns[0]]

        if np.any(np.iscomplex(l_qty.magnitude)):
            raise ComponentError(f"Inductance must be real for {self.fqn}.")
        if np.any(l_qty.magnitude < 0):
            raise ComponentError(f"Inductance must be non-negative for {self.fqn}.")
        
        omega = (2 * np.pi * freq_hz_array) * (ureg.rad / ureg.second)
        impedance_qty = (1j * omega * l_qty).to(ureg.ohm)
        
        y_mag_si = np.empty_like(impedance_qty.magnitude, dtype=np.complex128)
        
        # Handle non-ideal case first for numerical stability
        # Use a mask to avoid division by zero for ideal shorts at F=0
        l_mag = l_qty.magnitude
        non_ideal_mask = (impedance_qty.magnitude != 0) & (np.isfinite(l_mag))
        with np.errstate(divide='ignore', invalid='ignore'):
             y_mag_si[non_ideal_mask] = 1.0 / impedance_qty.magnitude[non_ideal_mask]
        
        # Patch ideal cases
        y_mag_si[impedance_qty.magnitude == 0] = LARGE_ADMITTANCE_SIEMENS + 0j
        y_mag_si[np.isposinf(l_mag)] = 0j
        
        num_freqs = len(freq_hz_array)
        stamp_mag = np.zeros((num_freqs, 2, 2), dtype=np.complex128)
        stamp_mag[:, 0, 0] = y_mag_si
        stamp_mag[:, 1, 1] = y_mag_si
        stamp_mag[:, 0, 1] = -y_mag_si
        stamp_mag[:, 1, 0] = -y_mag_si
        
        return [(Quantity(stamp_mag, ureg.siemens), [PORT_1, PORT_2])]

    def get_dc_behavior(self, all_dc_params: Dict[str, Quantity]) -> Tuple[DCBehaviorType, Optional[Quantity]]:
        param_fqn = self.parameter_fqns[0]
        l_real_val = _extract_dc_real_scalar_value(all_dc_params, param_fqn, self.fqn)

        if np.isposinf(l_real_val) or np.isnan(l_real_val):
            return (DCBehaviorType.OPEN_CIRCUIT, None)
        if l_real_val < 0:
            raise ComponentError(f"Negative inductance L={all_dc_params[param_fqn][0]:~P} not supported for DC analysis for '{self.fqn}'.")
        
        # L=0 or finite L > 0 are all short circuits at DC
        return (DCBehaviorType.SHORT_CIRCUIT, None)

    def is_structurally_open(self, resolved_constant_params: Dict[str, Quantity]) -> bool:
        if 'inductance' not in resolved_constant_params: return False
        return bool(np.isposinf(resolved_constant_params['inductance'].magnitude))