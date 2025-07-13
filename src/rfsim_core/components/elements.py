# src/rfsim_core/components/elements.py
"""
This module provides the concrete implementations for the fundamental, "leaf-level"
passive circuit elements: Resistor, Capacitor, and Inductor.
"""

import logging
import numpy as np
import pint
from typing import Dict, List, Optional, Tuple

# --- Foundational Imports ---
from ..units import ureg, Quantity
from ..constants import LARGE_ADMITTANCE_SIEMENS

# --- Core Component Model Imports ---
# StampInfo is now directly the tuple type returned by get_mna_stamps.
from .base import ComponentBase, register_component
from .base_enums import DCBehaviorType

# --- Capability System Imports ---
from .capabilities import IMnaContributor, IDcContributor, ITopologyContributor, provides
from .exceptions import ComponentError


logger = logging.getLogger(__name__)


def _extract_dc_real_scalar_value(
    all_dc_params: Dict[str, Quantity],
    param_fqn: str,
    component_fqn: str
) -> float:
    """
    A stateless helper to robustly extract a single, real, scalar float value
    from a resolved DC parameter. This enforces the contract required by the DCAnalyzer.
    """
    try:
        qty = all_dc_params[param_fqn][0]  # DC evaluation always yields a single-element array
        mag = qty.magnitude

        if np.iscomplexobj(mag) and mag.imag != 0:
            raise ComponentError(
                component_fqn=component_fqn,
                details=f"Parameter '{param_fqn}' for DC analysis must be real. Got {qty:~P}."
            )

        return float(mag.real if np.iscomplexobj(mag) else mag)
    except KeyError:
        raise ComponentError(
            component_fqn=component_fqn,
            details=f"Parameter '{param_fqn}' not found for DC analysis."
        ) from None
    except (TypeError, ValueError) as e:
        raise ComponentError(
            component_fqn=component_fqn,
            details=f"Could not convert DC parameter '{param_fqn}' to a scalar float: {e}"
        ) from e


def _validate_and_get_real_non_negative_magnitude(
    qty: Quantity,
    component_fqn: str,
    param_name: str,
    expected_dim: str
) -> np.ndarray:
    """
    A centralized, stateless utility to enforce all physical constraints on a parameter.
    It validates dimensionality, realness, and non-negativity, raising a canonical
    `ComponentError` for any violation. This is a cornerstone of the "Correctness by
    Construction" mandate for leaf components.
    """
    try:
        if not qty.is_compatible_with(expected_dim):
            raise pint.DimensionalityError(qty.units, ureg.Unit(expected_dim))

        raw_mag = np.asarray(qty.magnitude)

        if np.iscomplexobj(raw_mag):
            if np.any(raw_mag.imag != 0):
                raise ComponentError(
                    component_fqn=component_fqn,
                    details=f"Parameter '{param_name}' must be real, but received a complex value."
                )
            raw_mag = raw_mag.real

        if np.any(raw_mag < 0):
            raise ComponentError(
                component_fqn=component_fqn,
                details=f"Parameter '{param_name}' must be non-negative."
            )

        return raw_mag

    except (pint.DimensionalityError, TypeError, ValueError) as e:
        raise ComponentError(
            component_fqn=component_fqn,
            details=f"Validation failed for parameter '{param_name}': {e}"
        ) from e


@register_component("Resistor")
class Resistor(ComponentBase):
    """Represents an ideal Resistor component."""

    @provides(IMnaContributor)
    class MnaContributor:
        def get_mna_stamps(
            self, component: 'Resistor', freq_hz_array: np.ndarray, all_evaluated_params: Dict[str, Quantity]
        ) -> Tuple[Quantity, List[str]]:
            param_fqn = component.parameter_fqns[0]
            r_real_mag = _validate_and_get_real_non_negative_magnitude(
                all_evaluated_params[param_fqn], component.fqn, "resistance", "ohm")

            if r_real_mag.ndim == 0:
                r_real_mag = np.full_like(freq_hz_array, r_real_mag.item(), dtype=float)

            y_mag_si = np.empty_like(r_real_mag, dtype=np.complex128)
            non_ideal_mask = (r_real_mag > 0) & np.isfinite(r_real_mag)
            y_mag_si[non_ideal_mask] = 1.0 / r_real_mag[non_ideal_mask]
            y_mag_si[r_real_mag == 0] = LARGE_ADMITTANCE_SIEMENS
            y_mag_si[np.isposinf(r_real_mag)] = 0.0

            stamp_mag = np.zeros((len(freq_hz_array), 2, 2), dtype=np.complex128)
            stamp_mag[:, 0, 0] = stamp_mag[:, 1, 1] = y_mag_si
            stamp_mag[:, 0, 1] = stamp_mag[:, 1, 0] = -y_mag_si

            return (Quantity(stamp_mag, ureg.siemens), ['p1', 'p2'])

    @provides(IDcContributor)
    class DcContributor:
        def get_dc_behavior(self, component: 'Resistor', all_dc_params: Dict[str, Quantity]) -> Tuple[DCBehaviorType, Optional[Quantity]]:
            param_fqn = component.parameter_fqns[0]
            # Use the designated helper to safely extract the scalar float value,
            # thus adhering to the contract with the ParameterManager.
            r_real_val = _extract_dc_real_scalar_value(all_dc_params, param_fqn, component.fqn)
            
            if np.isnan(r_real_val) or np.isposinf(r_real_val): return (DCBehaviorType.OPEN_CIRCUIT, None)
            if r_real_val == 0.0: return (DCBehaviorType.SHORT_CIRCUIT, None)
            if r_real_val < 0: raise ComponentError(component_fqn=component.fqn, details=f"Negative resistance R={all_dc_params[param_fqn][0]:~P} not supported for DC analysis.")
            
            # FIXED: Perform the division on the now-guaranteed scalar value, then create the Quantity.
            # This prevents the TypeError from `1.0 / Quantity([50.0], 'ohm')`.
            return (DCBehaviorType.ADMITTANCE, Quantity(1.0 / r_real_val, ureg.siemens))

    @provides(ITopologyContributor)
    class TopologyContributor:
        def is_structurally_open(self, component: 'Resistor', resolved_constant_params: Dict[str, Quantity]) -> bool:
            if 'resistance' in resolved_constant_params:
                return bool(np.isposinf(resolved_constant_params['resistance'].magnitude))
            return False

    @classmethod
    def declare_parameters(cls) -> Dict[str, str]: return {"resistance": "ohm"}

    @classmethod
    def declare_ports(cls) -> List[str]: return ['p1', 'p2']


@register_component("Capacitor")
class Capacitor(ComponentBase):
    """Represents an ideal Capacitor component."""

    @provides(IMnaContributor)
    class MnaContributor:
        def get_mna_stamps(self, component: 'Capacitor', freq_hz_array: np.ndarray, all_evaluated_params: Dict[str, Quantity]) -> Tuple[Quantity, List[str]]:
            param_fqn = component.parameter_fqns[0]
            c_mag_si = _validate_and_get_real_non_negative_magnitude(
                all_evaluated_params[param_fqn], component.fqn, "capacitance", "farad")

            omega = (2 * np.pi * freq_hz_array)

            if c_mag_si.ndim == 0:
                c_mag_si = np.full_like(omega, c_mag_si.item(), dtype=float)

            y_mag_si = np.full_like(omega, 0j, dtype=np.complex128)

            finite_mask = (c_mag_si > 0) & np.isfinite(c_mag_si)
            if np.any(finite_mask):
                y_mag_si[finite_mask] = 1j * omega[finite_mask] * c_mag_si[finite_mask]

            inf_mask = np.isposinf(c_mag_si)
            if np.any(inf_mask):
                y_mag_si[inf_mask] = LARGE_ADMITTANCE_SIEMENS + 0j

            stamp_mag = np.zeros((len(freq_hz_array), 2, 2), dtype=np.complex128)
            stamp_mag[:, 0, 0] = stamp_mag[:, 1, 1] = y_mag_si
            stamp_mag[:, 0, 1] = stamp_mag[:, 1, 0] = -y_mag_si

            return (Quantity(stamp_mag, ureg.siemens), ['p1', 'p2'])

    @provides(IDcContributor)
    class DcContributor:
        def get_dc_behavior(self, component: 'Capacitor', all_dc_params: Dict[str, Quantity]) -> Tuple[DCBehaviorType, Optional[Quantity]]:
            param_fqn = component.parameter_fqns[0]
            # This implementation correctly uses the helper and is architecturally sound.
            c_real_val = _extract_dc_real_scalar_value(all_dc_params, param_fqn, component.fqn)

            # The logic here is robust because it operates on the extracted float.
            if np.isposinf(c_real_val): return (DCBehaviorType.SHORT_CIRCUIT, None)
            if c_real_val < 0: raise ComponentError(component_fqn=component.fqn, details=f"Negative capacitance C={all_dc_params[param_fqn][0]:~P} not supported for DC analysis.")
            return (DCBehaviorType.OPEN_CIRCUIT, None)

    @provides(ITopologyContributor)
    class TopologyContributor:
        def is_structurally_open(self, component: 'Capacitor', resolved_constant_params: Dict[str, Quantity]) -> bool:
            # A capacitor with C=0 is a structural open circuit.
            if 'capacitance' in resolved_constant_params:
                return resolved_constant_params['capacitance'].magnitude == 0
            return False

    @classmethod
    def declare_parameters(cls) -> Dict[str, str]: return {"capacitance": "farad"}

    @classmethod
    def declare_ports(cls) -> List[str]: return ['p1', 'p2']


@register_component("Inductor")
class Inductor(ComponentBase):
    """Represents an ideal Inductor component."""

    @provides(IMnaContributor)
    class MnaContributor:
        def get_mna_stamps(self, component: 'Inductor', freq_hz_array: np.ndarray, all_evaluated_params: Dict[str, Quantity]) -> Tuple[Quantity, List[str]]:
            param_fqn = component.parameter_fqns[0]
            l_mag_si = _validate_and_get_real_non_negative_magnitude(
                all_evaluated_params[param_fqn], component.fqn, "inductance", "henry")

            omega = (2 * np.pi * freq_hz_array)

            if l_mag_si.ndim == 0:
                l_mag_si = np.full_like(omega, l_mag_si.item(), dtype=float)

            z_mag_si = 1j * omega * l_mag_si
            y_mag_si = np.empty_like(z_mag_si, dtype=np.complex128)

            non_zero_z_mask = (z_mag_si != 0)
            with np.errstate(divide='ignore', invalid='ignore'):
                y_mag_si[non_zero_z_mask] = 1.0 / z_mag_si[non_zero_z_mask]

            y_mag_si[~non_zero_z_mask] = LARGE_ADMITTANCE_SIEMENS + 0j
            
            inf_l_mask = np.isposinf(l_mag_si)
            if np.any(inf_l_mask):
                y_mag_si[inf_l_mask] = 0j

            stamp_mag = np.zeros((len(freq_hz_array), 2, 2), dtype=np.complex128)
            stamp_mag[:, 0, 0] = stamp_mag[:, 1, 1] = y_mag_si
            stamp_mag[:, 0, 1] = stamp_mag[:, 1, 0] = -y_mag_si
            
            return (Quantity(stamp_mag, ureg.siemens), ['p1', 'p2'])

    @provides(IDcContributor)
    class DcContributor:
        def get_dc_behavior(self, component: 'Inductor', all_dc_params: Dict[str, Quantity]) -> Tuple[DCBehaviorType, Optional[Quantity]]:
            param_fqn = component.parameter_fqns[0]
            # This implementation correctly uses the helper and is architecturally sound.
            l_real_val = _extract_dc_real_scalar_value(all_dc_params, param_fqn, component.fqn)
            
            # The logic here is robust because it operates on the extracted float.
            if np.isposinf(l_real_val) or np.isnan(l_real_val): return (DCBehaviorType.OPEN_CIRCUIT, None)
            if l_real_val < 0: raise ComponentError(component_fqn=component.fqn, details=f"Negative inductance L={all_dc_params[param_fqn][0]:~P} not supported for DC analysis.")
            return (DCBehaviorType.SHORT_CIRCUIT, None)

    @provides(ITopologyContributor)
    class TopologyContributor:
        def is_structurally_open(self, component: 'Inductor', resolved_constant_params: Dict[str, Quantity]) -> bool:
            # An inductor with L=inf is a structural open circuit.
            if 'inductance' in resolved_constant_params:
                return bool(np.isposinf(resolved_constant_params['inductance'].magnitude))
            return False

    @classmethod
    def declare_parameters(cls) -> Dict[str, str]: return {"inductance": "henry"}

    @classmethod
    def declare_ports(cls) -> List[str]: return ['p1', 'p2']