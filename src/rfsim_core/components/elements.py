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
        # This is valid for F=0 as well.
        if R_mag == 0:
            y_val = LARGE_ADMITTANCE_SIEMENS + 0j
        else:
            # Ensure calculation is done with units for safety, then extract magnitude
            y_scalar_qty = (1.0 / self.resistance).to(ureg.siemens)
            y_val = complex(y_scalar_qty.magnitude) # Convert to complex float

        # Create the 2x2 stamp matrix magnitude
        # Need to broadcast if freq_hz is an array
        num_freqs = len(freq_hz) if freq_hz.ndim > 0 and freq_hz.size > 0 else 1 # Handle empty array case
        if num_freqs > 1:
             # Create a (num_freqs, 2, 2) array
             stamp_mag = np.zeros((num_freqs, 2, 2), dtype=np.complex128)
             stamp_mag[:, 0, 0] = y_val
             stamp_mag[:, 0, 1] = -y_val
             stamp_mag[:, 1, 0] = -y_val
             stamp_mag[:, 1, 1] = y_val
        elif num_freqs == 1 and freq_hz.size > 0: # Single frequency point
            stamp_mag = np.array([[y_val, -y_val], [-y_val, y_val]], dtype=np.complex128)
        else: # Handle empty freq_hz array case
             stamp_mag = np.empty((0, 2, 2), dtype=np.complex128)

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
             # Allow infinity, but not negative or complex capacitance
             if not np.isinf(self.capacitance.magnitude):
                 raise ComponentError(f"Capacitance must be real and non-negative (or infinite) for {self.instance_id}. Got {self.capacitance:~P}")
        if np.isinf(self.capacitance.magnitude):
            logger.warning(f"Component '{self.instance_id}' has infinite capacitance. Treated as ideal short (large finite admittance) for F>0 analysis.")


    def get_mna_stamps(self, freq_hz: np.ndarray) -> List[StampInfo]:
        """Calculates the 2x2 admittance matrix stamp Y = j*omega*C * [[1, -1], [-1, 1]]. Handles F=0 and C=inf."""
        if not isinstance(freq_hz, np.ndarray):
             raise TypeError(f"{self.instance_id}: freq_hz must be a NumPy array.")

        C_mag = self.capacitance.magnitude
        y_val_array: np.ndarray

        # Handle F=0 case explicitly**
        # At DC (F=0), capacitor is an open circuit (zero admittance), unless C=inf
        # Need to handle array input for freq_hz
        is_dc = (freq_hz == 0)
        is_infinite_cap = np.isinf(C_mag)

        if is_infinite_cap:
            # **REVISED (Point 2): Handle C=inf -> Ideal short for F>0**
            # For F=0, C=inf is also a short (infinite admittance)
            y_scalar = LARGE_ADMITTANCE_SIEMENS + 0j
            y_val_array = np.full_like(freq_hz, y_scalar, dtype=np.complex128)
            logger.debug(f"{self.instance_id}: C=inf treated as large admittance ({y_scalar:.2e} S).")
        else:
            # Standard case C finite and non-negative
            omega = (2 * np.pi * freq_hz) * (ureg.rad / ureg.second)
            y_qty = (1j * omega * self.capacitance).to(ureg.siemens)
            y_val_array_temp = np.asarray(y_qty.magnitude, dtype=np.complex128)

            # Ensure DC points are exactly zero admittance if C is finite
            if np.any(is_dc):
                y_val_array = np.where(is_dc, 0.0 + 0.0j, y_val_array_temp)
                logger.debug(f"{self.instance_id}: Finite C at F=0 treated as zero admittance.")
            else:
                y_val_array = y_val_array_temp

        # Create the 2x2 stamp matrix magnitude (potentially broadcast over frequencies)
        if y_val_array.ndim > 0 and y_val_array.size > 0: # freq_hz was a non-empty array
            num_freqs = len(y_val_array)
            stamp_mag = np.zeros((num_freqs, 2, 2), dtype=np.complex128)
            stamp_mag[:, 0, 0] = y_val_array
            stamp_mag[:, 0, 1] = -y_val_array
            stamp_mag[:, 1, 0] = -y_val_array
            stamp_mag[:, 1, 1] = y_val_array
        elif y_val_array.ndim == 0 and y_val_array.size == 1: # freq_hz was scalar
             y_scalar = complex(y_val_array) # Ensure complex
             stamp_mag = np.array([[y_scalar, -y_scalar], [-y_scalar, y_scalar]], dtype=np.complex128)
        else: # Handle empty freq_hz array case
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

    def __init__(self, instance_id: str, component_type: str, processed_params: Dict[str, Quantity]):
        super().__init__(instance_id, component_type, processed_params)
        self.inductance = self.get_parameter("inductance")
        if not np.isreal(self.inductance.magnitude) or self.inductance.magnitude < 0:
             raise ComponentError(f"Inductance must be real and non-negative for {self.instance_id}. Got {self.inductance:~P}")
        if self.inductance.magnitude == 0:
             logger.warning(f"Component '{self.instance_id}' has zero inductance. Treated as ideal short (large finite admittance) for F>=0 analysis.")

    def get_mna_stamps(self, freq_hz: np.ndarray) -> List[StampInfo]:
        """Calculates the 2x2 admittance matrix stamp Y = 1/(j*omega*L) * [[1, -1], [-1, 1]]. Handles F=0 and L=0."""
        if not isinstance(freq_hz, np.ndarray):
             raise TypeError(f"{self.instance_id}: freq_hz must be a NumPy array.")

        L_mag = self.inductance.magnitude
        y_val_array: np.ndarray # Declare type

        # **REVISED (Point 6): Handle F=0 case explicitly**
        # At DC (F=0), inductor is a short circuit (infinite admittance), unless L=inf (not handled yet)
        is_dc = (freq_hz == 0)
        is_zero_inductance = (L_mag == 0)

        if is_zero_inductance:
            # L=0 -> Large admittance, frequency independent for F>=0
            y_scalar = LARGE_ADMITTANCE_SIEMENS + 0j
            y_val_array = np.full_like(freq_hz, y_scalar, dtype=np.complex128)
            logger.debug(f"{self.instance_id}: L=0 treated as large admittance ({y_scalar:.2e} S).")
        else:
            # Standard case: L > 0
             # Create omega, avoiding division by zero for DC points initially
             omega = (2 * np.pi * freq_hz) * (ureg.rad / ureg.second)
             impedance = (1j * omega * self.inductance)

             # Calculate admittance, handle division by zero for DC points
             with np.errstate(divide='ignore', invalid='ignore'): # Suppress warning for 1/0
                 y_qty = (1.0 / impedance).to(ureg.siemens)
                 y_val_array_temp = np.asarray(y_qty.magnitude, dtype=np.complex128)

             # For DC points where L>0, impedance is 0, admittance is infinite (use LARGE_ADMITTANCE)
             if np.any(is_dc):
                 y_scalar_dc = LARGE_ADMITTANCE_SIEMENS + 0j
                 y_val_array = np.where(is_dc, y_scalar_dc, y_val_array_temp)
                 # Check for NaNs resulting from 0/0 if omega calculation had issues (shouldn't happen here)
                 y_val_array = np.nan_to_num(y_val_array, nan=y_scalar_dc, posinf=y_scalar_dc, neginf=y_scalar_dc)
                 logger.debug(f"{self.instance_id}: L>0 at F=0 treated as large admittance ({y_scalar_dc:.2e} S).")
             else:
                 y_val_array = y_val_array_temp

             # Handle potential frequency points that are *very* close to zero but not exactly zero
             # Might result in very large admittance values numerically
             # No explicit clamping here, relies on LARGE_ADMITTANCE for exact zero cases.

        # Create the 2x2 stamp matrix magnitude (potentially broadcast over frequencies)
        if y_val_array.ndim > 0 and y_val_array.size > 0: # freq_hz was a non-empty array
            num_freqs = len(y_val_array)
            stamp_mag = np.zeros((num_freqs, 2, 2), dtype=np.complex128)
            stamp_mag[:, 0, 0] = y_val_array
            stamp_mag[:, 0, 1] = -y_val_array
            stamp_mag[:, 1, 0] = -y_val_array
            stamp_mag[:, 1, 1] = y_val_array
        elif y_val_array.ndim == 0 and y_val_array.size == 1: # freq_hz was scalar
             y_scalar = complex(y_val_array) # Ensure complex
             stamp_mag = np.array([[y_scalar, -y_scalar], [-y_scalar, y_scalar]], dtype=np.complex128)
        else: # Handle empty freq_hz array case
             stamp_mag = np.empty((0, 2, 2), dtype=np.complex128)

        admittance_matrix_qty = Quantity(stamp_mag, ureg.siemens)
        return [(admittance_matrix_qty, [PORT_1, PORT_2])]