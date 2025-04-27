# --- src/rfsim_core/components/elements.py ---
import logging
import numpy as np

from typing import Dict, List
from .base import (
    ComponentBase, register_component, ComponentError, StampInfo,
    LARGE_ADMITTANCE_SIEMENS
)
from ..units import ureg, pint, Quantity

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
        return [PORT_1, PORT_2] # Use integer port IDs

    def __init__(self, instance_id: str, component_type: str, processed_params: Dict[str, Quantity]):
        super().__init__(instance_id, component_type, processed_params)
        self.resistance = self.get_parameter("resistance")
        if not np.isreal(self.resistance.magnitude) or self.resistance.magnitude < 0:
             raise ComponentError(f"Resistance must be real and non-negative for {self.instance_id}. Got {self.resistance:~P}")
        if self.resistance.magnitude == 0:
             logger.warning(f"Component '{self.instance_id}' has zero resistance. Treated as ideal short (large finite admittance) for F>0 analysis.")

    def get_mna_stamps(self, freq_hz: np.ndarray) -> List[StampInfo]:
        """Calculates the 2x2 admittance matrix stamp."""
        if not isinstance(freq_hz, np.ndarray):
            raise TypeError(f"{self.instance_id}: freq_hz must be a NumPy array.")

        R_mag = self.resistance.magnitude

        # Calculate scalar admittance value (potentially large for R=0)
        if R_mag == 0:
            y_val = LARGE_ADMITTANCE_SIEMENS + 0j
        else:
            # Ensure calculation is done with units for safety, then extract magnitude
            y_scalar_qty = (1.0 / self.resistance).to(ureg.siemens)
            y_val = complex(y_scalar_qty.magnitude) # Convert to complex float

        # Create the 2x2 stamp matrix magnitude
        # Need to broadcast if freq_hz is an array
        num_freqs = len(freq_hz) if freq_hz.ndim > 0 else 1
        if num_freqs > 1:
            # Create a (num_freqs, 2, 2) array
            stamp_mag = np.zeros((num_freqs, 2, 2), dtype=np.complex128)
            stamp_mag[:, 0, 0] = y_val
            stamp_mag[:, 0, 1] = -y_val
            stamp_mag[:, 1, 0] = -y_val
            stamp_mag[:, 1, 1] = y_val
        else:
             # Create a (2, 2) array
            stamp_mag = np.array([[y_val, -y_val], [-y_val, y_val]], dtype=np.complex128)


        # Package into a Quantity with the correct dimension and shape
        admittance_matrix_qty = Quantity(stamp_mag, ureg.siemens)

        # Return the stamp info: (matrix quantity, port list)
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

    def __init__(self, instance_id: str, component_type: str, processed_params: Dict[str, Quantity]):
        super().__init__(instance_id, component_type, processed_params)
        self.capacitance = self.get_parameter("capacitance")
        if not np.isreal(self.capacitance.magnitude) or self.capacitance.magnitude < 0:
             raise ComponentError(f"Capacitance must be real and non-negative for {self.instance_id}. Got {self.capacitance:~P}")

    def get_mna_stamps(self, freq_hz: np.ndarray) -> List[StampInfo]:
        """Calculates the 2x2 admittance matrix stamp Y = j*omega*C * [[1, -1], [-1, 1]]."""
        if not isinstance(freq_hz, np.ndarray):
             raise TypeError(f"{self.instance_id}: freq_hz must be a NumPy array.")

        # Calculate frequency-dependent admittance value y = j*omega*C
        omega = (2 * np.pi * freq_hz) * (ureg.rad / ureg.second)
        y_qty = (1j * omega * self.capacitance).to(ureg.siemens)
        y_val_array = np.asarray(y_qty.magnitude, dtype=np.complex128) # This will be scalar or array matching freq_hz

        # Create the 2x2 stamp matrix magnitude (potentially broadcast over frequencies)
        if y_val_array.ndim > 0: # freq_hz was an array
            num_freqs = len(y_val_array)
            stamp_mag = np.zeros((num_freqs, 2, 2), dtype=np.complex128)
            stamp_mag[:, 0, 0] = y_val_array
            stamp_mag[:, 0, 1] = -y_val_array
            stamp_mag[:, 1, 0] = -y_val_array
            stamp_mag[:, 1, 1] = y_val_array
        else: # freq_hz was scalar
             y_scalar = complex(y_val_array) # Ensure complex
             stamp_mag = np.array([[y_scalar, -y_scalar], [-y_scalar, y_scalar]], dtype=np.complex128)

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

    def __init__(self, instance_id: str, component_type: str, processed_params: Dict[str, Quantity]):
        super().__init__(instance_id, component_type, processed_params)
        self.inductance = self.get_parameter("inductance")
        if not np.isreal(self.inductance.magnitude) or self.inductance.magnitude < 0:
             raise ComponentError(f"Inductance must be real and non-negative for {self.instance_id}. Got {self.inductance:~P}")
        if self.inductance.magnitude == 0:
             logger.warning(f"Component '{self.instance_id}' has zero inductance. Treated as ideal short (large finite admittance) for F>0 analysis.")

    def get_mna_stamps(self, freq_hz: np.ndarray) -> List[StampInfo]:
        """Calculates the 2x2 admittance matrix stamp Y = 1/(j*omega*L) * [[1, -1], [-1, 1]]."""
        if not isinstance(freq_hz, np.ndarray):
             raise TypeError(f"{self.instance_id}: freq_hz must be a NumPy array.")

        L_mag = self.inductance.magnitude

        # Calculate frequency-dependent admittance value y = 1 / (j*omega*L)
        y_val_array: np.ndarray # Declare type

        if L_mag == 0:
            # Special case: L=0 -> Large admittance, frequency independent for F>0
            y_scalar = LARGE_ADMITTANCE_SIEMENS + 0j
            # Ensure shape matches freq_hz if it's an array
            y_val_array = np.full_like(freq_hz, y_scalar, dtype=np.complex128)
        else:
            # Standard case: L > 0
             if np.any(freq_hz <= 0):
                  raise ComponentError(f"{self.instance_id}: AC analysis frequency must be > 0 Hz. Found {freq_hz[freq_hz <= 0]}.")
             omega = (2 * np.pi * freq_hz) * (ureg.rad / ureg.second)
             impedance = (1j * omega * self.inductance)
             y_qty = (1.0 / impedance).to(ureg.siemens)
             y_val_array = np.asarray(y_qty.magnitude, dtype=np.complex128)

        # Create the 2x2 stamp matrix magnitude (potentially broadcast over frequencies)
        if y_val_array.ndim > 0: # freq_hz was an array
            num_freqs = len(y_val_array)
            stamp_mag = np.zeros((num_freqs, 2, 2), dtype=np.complex128)
            stamp_mag[:, 0, 0] = y_val_array
            stamp_mag[:, 0, 1] = -y_val_array
            stamp_mag[:, 1, 0] = -y_val_array
            stamp_mag[:, 1, 1] = y_val_array
        else: # freq_hz was scalar
             y_scalar = complex(y_val_array) # Ensure complex
             stamp_mag = np.array([[y_scalar, -y_scalar], [-y_scalar, y_scalar]], dtype=np.complex128)

        admittance_matrix_qty = Quantity(stamp_mag, ureg.siemens)
        return [(admittance_matrix_qty, [PORT_1, PORT_2])]