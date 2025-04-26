# src/rfsim_core/components/elements.py
import logging
import numpy as np

from typing import Dict
from .base import ComponentBase, register_component, ComponentError
from ..units import ureg, pint

Quantity = ureg.Quantity
logger = logging.getLogger(__name__)

@register_component("Resistor")
class Resistor(ComponentBase):
    """Ideal linear resistor."""

    @classmethod
    def declare_parameters(cls) -> Dict[str, str]:
        return {"resistance": "ohm"}

    def __init__(self, instance_id: str, component_type: str, processed_params: Dict[str, Quantity]):
        super().__init__(instance_id, component_type, processed_params)
        self.resistance = self.get_parameter("resistance")
        # Basic check (more robust handling, esp. R=0, comes with DC analysis)
        if self.resistance.magnitude < 0:
             raise ComponentError(f"Resistance cannot be negative for {self.instance_id}. Got {self.resistance:~P}")
        if self.resistance.magnitude == 0:
             logger.warning(f"Component '{self.instance_id}' has zero resistance. This implies a DC short.")

    def get_admittance(self, freq_hz: np.ndarray) -> Quantity:
        """Calculates admittance Y = 1/R, broadcast across frequencies."""
        if not isinstance(freq_hz, np.ndarray):
            raise TypeError("freq_hz must be a NumPy array.")

        try:
            admittance_scalar = 1.0 / self.resistance
            # Broadcast the scalar admittance to the shape of freq_hz
            admittance_value = np.full_like(freq_hz, admittance_scalar.magnitude, dtype=np.complex128)
            return Quantity(admittance_value, ureg.siemens)

        except ZeroDivisionError:
            logger.debug(f"Zero resistance detected for {self.instance_id}, returning infinite admittance array.")
            inf_adm_value = np.full_like(freq_hz, np.inf + 0j, dtype=np.complex128)
            return Quantity(inf_adm_value, ureg.siemens)


@register_component("Capacitor")
class Capacitor(ComponentBase):
    """Ideal linear capacitor."""

    @classmethod
    def declare_parameters(cls) -> Dict[str, str]:
        return {"capacitance": "farad"} # Expects Farads

    def __init__(self, instance_id: str, component_type: str, processed_params: Dict[str, Quantity]):
        super().__init__(instance_id, component_type, processed_params)
        self.capacitance = self.get_parameter("capacitance")
        if self.capacitance.magnitude < 0:
             raise ComponentError(f"Capacitance cannot be negative for {self.instance_id}. Got {self.capacitance:~P}")
        # C=0 is numerically okay (zero admittance), maybe warn later if needed

    def get_admittance(self, freq_hz: np.ndarray) -> Quantity:
        """Calculates admittance Y = j * omega * C vectorized."""
        if not isinstance(freq_hz, np.ndarray):
            raise TypeError("freq_hz must be a NumPy array.")

        omega = (2 * np.pi * freq_hz) * ureg.rad / ureg.second
        admittance = 1j * omega * self.capacitance
        admittance_siemens = admittance.to(ureg.siemens)
        # Ensure complex data type for the array
        admittance_value = admittance_siemens.magnitude.astype(np.complex128)
        return Quantity(admittance_value, ureg.siemens)


@register_component("Inductor")
class Inductor(ComponentBase):
    """Ideal linear inductor."""

    @classmethod
    def declare_parameters(cls) -> Dict[str, str]:
        return {"inductance": "henry"} # Expects Henry ([inductance])

    def __init__(self, instance_id: str, component_type: str, processed_params: Dict[str, Quantity]):
        super().__init__(instance_id, component_type, processed_params)
        self.inductance = self.get_parameter("inductance")
        if self.inductance.magnitude < 0:
             raise ComponentError(f"Inductance cannot be negative for {self.instance_id}. Got {self.inductance:~P}")
        if self.inductance.magnitude == 0:
             logger.warning(f"Component '{self.instance_id}' has zero inductance. This implies a DC short.")
             # L=0 -> infinite admittance at F>0

    def get_admittance(self, freq_hz: np.ndarray) -> Quantity: # <--- Updated signature
        """Calculates admittance Y = 1 / (j * omega * L) vectorized."""
        if not isinstance(freq_hz, np.ndarray):
            raise TypeError("freq_hz must be a NumPy array.")

        admittance_value = np.empty_like(freq_hz, dtype=np.complex128)

        if self.inductance.magnitude == 0:
            # L = 0 case -> Infinite admittance for all F
            logger.debug(f"Zero inductance detected for {self.instance_id}, returning infinite admittance array.")
            admittance_value[:] = np.inf + 0j
            # Log specific warning for F=0 if present, though behavior is inf anyway
            if np.any(freq_hz == 0):
                 logger.warning(f"L=0 at F=0 for {self.instance_id}. Admittance treated as Inf. DC analysis should handle this.")
            return Quantity(admittance_value, ureg.siemens)

        # Proceed for L > 0
        # Handle AC and DC parts separately using masks
        dc_mask = (freq_hz == 0)
        ac_mask = ~dc_mask

        # DC case (F=0, L>0) -> Infinite admittance (ideal short)
        if np.any(dc_mask):
            admittance_value[dc_mask] = np.inf + 0j
            logger.warning(f"Inductor '{self.instance_id}' treated as ideal short (infinite admittance) at F=0 Hz points within AC analysis path. Use dedicated DC analysis for accurate results.")

        # AC case (F>0, L>0)
        if np.any(ac_mask):
            omega = (2 * np.pi * freq_hz[ac_mask]) * ureg.rad / ureg.second
            impedance = 1j * omega * self.inductance
            # Use numpy division for element-wise calculation
            admittance = (1.0 / impedance).to(ureg.siemens)
            admittance_value[ac_mask] = admittance.magnitude.astype(np.complex128)

        return Quantity(admittance_value, ureg.siemens)


# Ensure components/__init__.py exists and potentially imports these to trigger registration
# Create src/rfsim_core/components/__init__.py
# Content:
# from .base import ComponentBase, COMPONENT_REGISTRY
# from .elements import Resistor, Capacitor, Inductor