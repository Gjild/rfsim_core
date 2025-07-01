# src/rfsim_core/components/elements.py

"""
This module provides the concrete implementations for the fundamental, "leaf-level"
passive circuit elements: Resistor, Capacitor, and Inductor.

**Architectural Refactoring (Phase 9):**

This module has been refactored to align with the capability-based component model.
The key architectural changes are as follows:

1.  **Cohesion of Implementation:** All logic related to a specific analysis domain
    (e.g., MNA, DC) is now encapsulated within a dedicated, stateless, nested class
    inside its parent component class (e.g., `Resistor.MnaContributor`). This makes
    the component's total functionality self-contained and highly discoverable.

2.  **Declarative Registration:** The new `@provides` decorator is used on these
    nested classes to declaratively register them as implementers of a specific
    capability protocol (e.g., `IMnaContributor`). This automates discovery and
    removes the burden of manual registration from the component author.

3.  **Decoupled Interface:** The top-level component classes (`Resistor`, `Capacitor`,
    `Inductor`) no longer contain direct implementations of `get_mna_stamps` or
    `get_dc_behavior`. Instead, analysis engines will query for these capabilities
    using `component.get_capability(IMnaContributor)`. This change completes the
    decoupling of analysis logic from component implementation.

4.  **Stateless Capability with Context:** The nested capability implementation classes
    are stateless. When their methods are called by the framework, they receive the
    parent component instance as an explicit first argument (e.g., `component: 'Resistor'`).
    This provides all necessary context (like the component's FQN or parameter list)
    for the calculation, while maintaining a clean, service-oriented design.

The logic within the capability methods (`get_mna_stamps`, `get_dc_behavior`) remains
identical to the pre-refactoring implementation, merely relocated to its new,
architecturally-sound home. This ensures that the component's behavior is preserved
while its structure is vastly improved for future extensibility.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

# --- Foundational Imports ---
from ..units import ureg, Quantity
from ..constants import LARGE_ADMITTANCE_SIEMENS

# --- Core Component Model Imports ---
# Import the base classes and enums required for component definition.
from .base import (
    ComponentBase, register_component, ComponentError, StampInfo
)
from .base_enums import DCBehaviorType

# --- Capability System Imports (The Heart of the New Architecture) ---
# Import the capability protocols and the registration decorator.
from .capabilities import IMnaContributor, IDcContributor, provides


logger = logging.getLogger(__name__)

# Define port constants for clarity and to prevent magic numbers.
PORT_1 = 0
PORT_2 = 1

def _extract_dc_real_scalar_value(
    all_dc_params: Dict[str, Quantity],
    param_fqn: str,
    component_fqn: str
) -> float:
    """
    A stateless helper to robustly extract a single, real, scalar float value
    from a resolved DC parameter. This function remains unchanged as it serves
    as a utility for the DCContributor capabilities.
    """
    try:
        # DC evaluation returns a Quantity with a 1-element array; extract the scalar.
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
    """Represents an ideal Resistor component."""

    # --- Capability Implementations are Nested Here ---
    # This section contains the implementations for all capabilities this component provides.

    @provides(IMnaContributor)
    class MnaContributor:
        """Stateless implementation of the MNA contribution capability for a Resistor."""
        def get_mna_stamps(
            self,
            component: 'Resistor',  # The component instance provides context.
            freq_hz_array: np.ndarray,
            all_evaluated_params: Dict[str, Quantity]
        ) -> List[StampInfo]:
            """
            Calculates the vectorized MNA stamp for the Resistor. The logic is
            moved verbatim from the old Resistor.get_mna_stamps method.
            """
            param_fqn = component.parameter_fqns[0]
            r_qty = all_evaluated_params[param_fqn]
            r_mag_si = r_qty.to(ureg.ohm).magnitude

            if np.any(np.iscomplex(r_mag_si)) and np.any(r_mag_si.imag != 0):
                raise ComponentError(f"Resistance must be real for {component.fqn}. Got a complex value.")

            r_real_mag = r_mag_si.real if np.any(np.iscomplex(r_mag_si)) else r_mag_si
            if np.any(r_real_mag < 0):
                raise ComponentError(f"Resistance must be non-negative for {component.fqn}. Got a negative value.")

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

    @provides(IDcContributor)
    class DcContributor:
        """Stateless implementation of the DC contribution capability for a Resistor."""
        def get_dc_behavior(
            self,
            component: 'Resistor',  # The component instance provides context.
            all_dc_params: Dict[str, Quantity]
        ) -> Tuple[DCBehaviorType, Optional[Quantity]]:
            """
            Determines the Resistor's behavior at F=0. The logic is moved
            verbatim from the old Resistor.get_dc_behavior method.
            """
            param_fqn = component.parameter_fqns[0]
            r_real_val = _extract_dc_real_scalar_value(all_dc_params, param_fqn, component.fqn)

            if np.isnan(r_real_val) or np.isposinf(r_real_val):
                return (DCBehaviorType.OPEN_CIRCUIT, None)
            if r_real_val == 0.0:
                return (DCBehaviorType.SHORT_CIRCUIT, None)
            if r_real_val < 0:
                raise ComponentError(f"Negative resistance R={all_dc_params[param_fqn][0]:~P} not supported for DC analysis for '{component.fqn}'.")

            return (DCBehaviorType.ADMITTANCE, (1.0 / all_dc_params[param_fqn][0]).to(ureg.siemens))

    # --- Component's Own Declarations Follow ---
    # These methods define the component's static interface and remain on the main class.

    @classmethod
    def declare_parameters(cls) -> Dict[str, str]:
        return {"resistance": "ohm"}

    @classmethod
    def declare_ports(cls) -> List[str | int]:
        return [PORT_1, PORT_2]

    def is_structurally_open(self, resolved_constant_params: Dict[str, Quantity]) -> bool:
        if 'resistance' not in resolved_constant_params: return False
        return bool(np.isposinf(resolved_constant_params['resistance'].magnitude))


@register_component("Capacitor")
class Capacitor(ComponentBase):
    """Represents an ideal Capacitor component."""

    # --- Capability Implementations are Nested Here ---

    @provides(IMnaContributor)
    class MnaContributor:
        """Stateless implementation of the MNA contribution capability for a Capacitor."""
        def get_mna_stamps(
            self, component: 'Capacitor', freq_hz_array: np.ndarray, all_evaluated_params: Dict[str, Quantity]
        ) -> List[StampInfo]:
            param_fqn = component.parameter_fqns[0]
            c_qty = all_evaluated_params[param_fqn]

            if np.any(np.iscomplex(c_qty.magnitude)):
                raise ComponentError(f"Capacitance must be real for {component.fqn}.")
            if np.any(c_qty.magnitude < 0):
                raise ComponentError(f"Capacitance must be non-negative for {component.fqn}.")

            omega = (2 * np.pi * freq_hz_array) * (ureg.rad / ureg.second)
            y_qty = (1j * omega * c_qty)

            # Create a complex array for admittances in Siemens.
            y_mag_si = np.full_like(freq_hz_array, 0j, dtype=np.complex128)
            
            # Vectorized calculation for finite, non-zero capacitance
            c_mag = c_qty.magnitude
            finite_mask = (c_mag > 0) & np.isfinite(c_mag)
            y_mag_si[finite_mask] = y_qty.to(ureg.siemens).magnitude[finite_mask]

            # Patch ideal cases
            y_mag_si[np.isposinf(c_mag)] = LARGE_ADMITTANCE_SIEMENS + 0j
            y_mag_si[c_mag == 0] = 0j

            num_freqs = len(freq_hz_array)
            stamp_mag = np.zeros((num_freqs, 2, 2), dtype=np.complex128)
            stamp_mag[:, 0, 0] = y_mag_si
            stamp_mag[:, 1, 1] = y_mag_si
            stamp_mag[:, 0, 1] = -y_mag_si
            stamp_mag[:, 1, 0] = -y_mag_si

            return [(Quantity(stamp_mag, ureg.siemens), [PORT_1, PORT_2])]

    @provides(IDcContributor)
    class DcContributor:
        """Stateless implementation of the DC contribution capability for a Capacitor."""
        def get_dc_behavior(
            self, component: 'Capacitor', all_dc_params: Dict[str, Quantity]
        ) -> Tuple[DCBehaviorType, Optional[Quantity]]:
            param_fqn = component.parameter_fqns[0]
            c_real_val = _extract_dc_real_scalar_value(all_dc_params, param_fqn, component.fqn)

            if np.isposinf(c_real_val):
                # An infinite capacitor is a perfect short at DC (stores infinite charge for any voltage).
                return (DCBehaviorType.SHORT_CIRCUIT, None)
            if c_real_val < 0:
                raise ComponentError(f"Negative capacitance C={all_dc_params[param_fqn][0]:~P} not supported for DC analysis for '{component.fqn}'.")

            # C=0, finite C > 0, or C=NaN are all perfect open circuits at DC.
            return (DCBehaviorType.OPEN_CIRCUIT, None)

    # --- Component's Own Declarations Follow ---

    @classmethod
    def declare_parameters(cls) -> Dict[str, str]:
        return {"capacitance": "farad"}

    @classmethod
    def declare_ports(cls) -> List[str | int]:
        return [PORT_1, PORT_2]

    def is_structurally_open(self, resolved_constant_params: Dict[str, Quantity]) -> bool:
        if 'capacitance' not in resolved_constant_params: return False
        # A zero-value capacitor is a permanent open circuit across all frequencies.
        return resolved_constant_params['capacitance'].magnitude == 0


@register_component("Inductor")
class Inductor(ComponentBase):
    """Represents an ideal Inductor component."""

    # --- Capability Implementations are Nested Here ---

    @provides(IMnaContributor)
    class MnaContributor:
        """Stateless implementation of the MNA contribution capability for an Inductor."""
        def get_mna_stamps(
            self, component: 'Inductor', freq_hz_array: np.ndarray, all_evaluated_params: Dict[str, Quantity]
        ) -> List[StampInfo]:
            param_fqn = component.parameter_fqns[0]
            l_qty = all_evaluated_params[param_fqn]

            if np.any(np.iscomplex(l_qty.magnitude)):
                raise ComponentError(f"Inductance must be real for {component.fqn}.")
            if np.any(l_qty.magnitude < 0):
                raise ComponentError(f"Inductance must be non-negative for {component.fqn}.")

            omega = (2 * np.pi * freq_hz_array) * (ureg.rad / ureg.second)
            impedance_qty = (1j * omega * l_qty).to(ureg.ohm)
            y_mag_si = np.empty_like(impedance_qty.magnitude, dtype=np.complex128)

            # Use a mask to avoid division by zero for ideal shorts at F>0
            # and to handle the F=0 case separately.
            l_mag = l_qty.magnitude
            non_zero_impedance_mask = (impedance_qty.magnitude != 0) & (np.isfinite(l_mag))
            with np.errstate(divide='ignore', invalid='ignore'):
                 y_mag_si[non_zero_impedance_mask] = 1.0 / impedance_qty.magnitude[non_zero_impedance_mask]

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

    @provides(IDcContributor)
    class DcContributor:
        """Stateless implementation of the DC contribution capability for an Inductor."""
        def get_dc_behavior(
            self, component: 'Inductor', all_dc_params: Dict[str, Quantity]
        ) -> Tuple[DCBehaviorType, Optional[Quantity]]:
            param_fqn = component.parameter_fqns[0]
            l_real_val = _extract_dc_real_scalar_value(all_dc_params, param_fqn, component.fqn)

            if np.isposinf(l_real_val) or np.isnan(l_real_val):
                # An infinite inductor is a perfect open circuit.
                return (DCBehaviorType.OPEN_CIRCUIT, None)
            if l_real_val < 0:
                raise ComponentError(f"Negative inductance L={all_dc_params[param_fqn][0]:~P} not supported for DC analysis for '{component.fqn}'.")

            # L=0 or any finite L > 0 are all perfect short circuits at DC.
            return (DCBehaviorType.SHORT_CIRCUIT, None)

    # --- Component's Own Declarations Follow ---

    @classmethod
    def declare_parameters(cls) -> Dict[str, str]:
        return {"inductance": "henry"}

    @classmethod
    def declare_ports(cls) -> List[str | int]:
        return [PORT_1, PORT_2]

    def is_structurally_open(self, resolved_constant_params: Dict[str, Quantity]) -> bool:
        if 'inductance' not in resolved_constant_params: return False
        # An infinite-value inductor is a permanent open circuit across all frequencies.
        return bool(np.isposinf(resolved_constant_params['inductance'].magnitude))