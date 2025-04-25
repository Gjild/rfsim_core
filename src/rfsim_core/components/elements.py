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

    def get_admittance(self, freq_hz: np.ndarray | float) -> Quantity:
        """Calculates admittance Y = 1/R."""
        try:
            # Pint handles division, result should have dimension 1/ohm = admittance
            admittance = 1.0 / self.resistance
            # Ensure result is complex, matching expected MNA type, broadcast if needed
            if isinstance(freq_hz, np.ndarray):
                # Broadcast the scalar admittance to the shape of freq_hz
                admittance_value = np.full(freq_hz.shape, admittance.magnitude, dtype=np.complex128)
            else:
                admittance_value = np.complex128(admittance.magnitude) # Single complex value

            # Return as Quantity with correct units (Siemens)
            return Quantity(admittance_value, ureg.siemens)
        except ZeroDivisionError:
            # R = 0 case -> Infinite admittance (handled numerically later in MNA for F>0)
            logger.debug(f"Zero resistance detected for {self.instance_id}, returning infinite admittance placeholder.")
            # Represent infinity numerically for now, MNA stage will handle substitution
            inf_adm = np.inf + 0j # Represent as complex infinity
            if isinstance(freq_hz, np.ndarray):
                 inf_adm_value = np.full(freq_hz.shape, inf_adm, dtype=np.complex128)
            else:
                 inf_adm_value = np.complex128(inf_adm)
            # Still return as a Quantity, MNA assembler must handle np.inf
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

    def get_admittance(self, freq_hz: np.ndarray | float) -> Quantity:
        """Calculates admittance Y = j * omega * C."""
        # Ensure frequency is treated correctly unit-wise
        omega = (2 * np.pi * freq_hz) * ureg.rad / ureg.second
        admittance = 1j * omega * self.capacitance

        # Ensure the result has the correct dimensions (admittance) - Pint handles this
        # Convert to base units (Siemens) and ensure complex type
        admittance_siemens = admittance.to(ureg.siemens)

        # Ensure complex data type
        if isinstance(admittance_siemens.magnitude, np.ndarray):
            admittance_value = admittance_siemens.magnitude.astype(np.complex128)
        else:
            admittance_value = np.complex128(admittance_siemens.magnitude)

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

    def get_admittance(self, freq_hz: np.ndarray | float) -> Quantity:
        """Calculates admittance Y = 1 / (j * omega * L)."""
        if self.inductance.magnitude == 0:
            # L = 0 case -> Infinite admittance for F > 0
            logger.debug(f"Zero inductance detected for {self.instance_id}, returning infinite admittance placeholder.")
            inf_adm = np.inf + 0j
            if isinstance(freq_hz, np.ndarray):
                dc_mask = (freq_hz == 0)
                ac_mask = ~dc_mask
                admittance_value = np.empty_like(freq_hz, dtype=np.complex128)

                if np.any(dc_mask):
                    # For L>0 at F=0, admittance is infinite (ideal short) based on limit
                    admittance_value[dc_mask] = np.inf + 0j
                    logger.warning(f"Inductor '{self.instance_id}' treated as ideal short (infinite admittance) at F=0 Hz within AC analysis path. Use dedicated DC analysis for accurate results.")

                if np.any(ac_mask):
                    omega = (2 * np.pi * freq_hz[ac_mask]) * ureg.rad / ureg.second
                    impedance = 1j * omega * self.inductance
                    admittance = (1.0 / impedance).to(ureg.siemens)
                    admittance_value[ac_mask] = admittance.magnitude.astype(np.complex128)

            elif freq_hz == 0:
                # Handle single DC frequency
                logger.warning(f"Inductor '{self.instance_id}' treated as ideal short (infinite admittance) at F=0 Hz within AC analysis path. Use dedicated DC analysis for accurate results.")
                admittance_value = np.complex128(np.inf + 0j)
            else:
                # Handle single AC frequency
                omega = (2 * np.pi * freq_hz) * ureg.rad / ureg.second
                impedance = 1j * omega * self.inductance
                admittance = (1.0 / impedance).to(ureg.siemens)
                admittance_value = np.complex128(admittance.magnitude)

            return Quantity(admittance_value, ureg.siemens)

        # Proceed for L > 0
        omega = (2 * np.pi * freq_hz) * ureg.rad / ureg.second
        impedance = 1j * omega * self.inductance
        admittance = 1.0 / impedance

        # Ensure the result has the correct dimensions (admittance) - Pint handles this
        # Convert to base units (Siemens) and ensure complex type
        admittance_siemens = admittance.to(ureg.siemens)

        # Ensure complex data type and handle potential division by zero if freq_hz=0
        if isinstance(admittance_siemens.magnitude, np.ndarray):
            # Handle freq=0 case resulting in potential inf/nan
            valid_idx = omega.magnitude != 0
            admittance_value = np.full_like(omega.magnitude, np.nan + 0j, dtype=np.complex128) # Default to NaN for F=0
            if np.any(valid_idx):
                 admittance_value[valid_idx] = admittance_siemens.magnitude[valid_idx].astype(np.complex128)
            if np.any(~valid_idx): # If F=0 was present
                 logger.warning(f"Inductor '{self.instance_id}' has infinite impedance (zero admittance) at F=0 Hz. Returning NaN admittance for F=0 points.")
                 # Technically Y=0 at DC. MNA handles zero stamps fine. Let's return 0.
                 admittance_value[~valid_idx] = 0.0 + 0.0j
                 logger.info(f"Corrected F=0 admittance for inductor '{self.instance_id}' to 0 Siemens.")
        else:
            # Handle single frequency case
            if omega.magnitude == 0: # Check for F=0
                logger.warning(f"Inductor '{self.instance_id}' has infinite impedance (zero admittance) at F=0 Hz. Returning 0 Siemens.")
                admittance_value = np.complex128(0.0)
            else:
                admittance_value = np.complex128(admittance_siemens.magnitude)

        return Quantity(admittance_value, ureg.siemens)


# Ensure components/__init__.py exists and potentially imports these to trigger registration
# Create src/rfsim_core/components/__init__.py
# Content:
# from .base import ComponentBase, COMPONENT_REGISTRY
# from .elements import Resistor, Capacitor, Inductor