# --- Modify: src/rfsim_core/components/elements.py ---
# Add implementation for 'is_structurally_open'

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Set # Added Set

from .base import (
    ComponentBase, register_component, ComponentError, StampInfo,
    # LARGE_ADMITTANCE_SIEMENS, # <-- REMOVE THIS IMPORT
    DCBehaviorType
)
# --- ADD Import from constants ---
from ..constants import LARGE_ADMITTANCE_SIEMENS

from ..units import ureg, pint, Quantity
# Import for type hints and temporary shim
from ..parameters import ParameterManager, ParameterError

logger = logging.getLogger(__name__)

# Standard port names/indices for simple 2-terminal elements
PORT_1 = 0
PORT_2 = 1

# --- Helper for common DC parameter extraction and initial validation ---
# Note: This assumes the caller provides Quantities with 1-element ndarray magnitudes.
def _extract_dc_real_scalar_value(
    resolved_params: Dict[str, Quantity],
    param_name: str,
    expected_units: str, # For error message context, not strict checking here
    instance_id: str
) -> float:
    """
    Extracts the scalar real value from a resolved DC parameter Quantity.

    Assumes input Quantity has a 1-element ndarray magnitude.
    Checks for non-real values and raises ComponentError.

    Returns:
        The scalar real value (can be float, including np.inf, np.nan).

    Raises:
        ComponentError: If the parameter is missing, not a Quantity, or has a non-zero imaginary part.
    """
    try:
        qty_vec = resolved_params[param_name]
    except KeyError:
        raise ComponentError(f"Parameter '{param_name}' not found in resolved_params for DC analysis of {instance_id}.")

    # Assume input structure is Quantity with 1-element ndarray magnitude, as per get_dc_behavior contract.
    # Minimal check for safety:
    if not isinstance(qty_vec, Quantity) or not hasattr(qty_vec, 'magnitude'):
         raise ComponentError(f"Resolved '{param_name}' for {instance_id} is not a Quantity as expected for DC analysis. Got: {type(qty_vec)}")

    # Indexing a Quantity containing a 1-element array extracts the scalar Quantity
    # Add a check for the expected 1-element array structure before indexing
    if not isinstance(qty_vec.magnitude, np.ndarray) or qty_vec.magnitude.size != 1:
        raise ComponentError(f"Resolved '{param_name}' for {instance_id} does not have a 1-element NumPy array magnitude as expected for DC analysis. Got shape: {getattr(qty_vec.magnitude, 'shape', type(qty_vec.magnitude))}")

    qty_scalar = qty_vec[0]
    mag_scalar = qty_scalar.magnitude # Extracts numerical value (Python or NumPy scalar)

    # Check for non-real values
    if np.iscomplexobj(mag_scalar) and mag_scalar.imag != 0: # Use iscomplexobj for robustness
        raise ComponentError(f"Parameter '{param_name}' for DC analysis for '{instance_id}' must be real. Got {qty_scalar:~P} (expected units ~{expected_units}).")

    real_val = mag_scalar.real if np.iscomplexobj(mag_scalar) else mag_scalar
    # Ensure standard float type, handles np types (float, float32, float64 etc.)
    # This also ensures that downstream comparisons work correctly (e.g., == 0)
    try:
        return float(real_val)
    except TypeError as e:
        raise ComponentError(f"Could not convert extracted real value '{real_val}' (type: {type(real_val)}) to float for parameter '{param_name}' of '{instance_id}'. Error: {e}")


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
        # --- (Implementation unchanged, uses LARGE_ADMITTANCE_SIEMENS correctly) ---
        if not isinstance(freq_hz, np.ndarray):
            raise TypeError(f"{self.instance_id}: freq_hz must be a NumPy array.")
        if not isinstance(resolved_params, dict):
            raise TypeError(f"{self.instance_id}: resolved_params must be a dictionary.")

        try:
            resistance_qty = resolved_params['resistance']
        except KeyError:
            raise ComponentError(f"Parameter 'resistance' not found in resolved_params for {self.instance_id}. Available: {list(resolved_params.keys())}")

        # Validate shape just in case (though MnaAssembler should ensure 1 freq for each call)
        if not isinstance(resistance_qty.magnitude, np.ndarray) or resistance_qty.magnitude.shape != (1,):
             logger.warning(f"{self.instance_id}: Resistance quantity magnitude shape {getattr(resistance_qty.magnitude, 'shape', type(resistance_qty.magnitude))} unexpected, expected (1,). Using first element or failing if not possible.")
             if not isinstance(resistance_qty.magnitude, np.ndarray) or resistance_qty.magnitude.size < 1:
                 raise ComponentError(f"Resistance parameter '{self.instance_id}' has invalid magnitude shape/size for MNA stamping.")
             current_R_qty_scalar = resistance_qty[0] # Attempt extraction anyway
        else:
            current_R_qty_scalar = resistance_qty[0] # Extract scalar Quantity

        # get_mna_stamps handles F>0. Use ComponentError for negative R here too.
        if np.iscomplexobj(current_R_qty_scalar.magnitude) and current_R_qty_scalar.magnitude.imag != 0:
             raise ComponentError(f"Resistance must be real for {self.instance_id} MNA stamping. Got {current_R_qty_scalar:~P}")

        r_real_mag = current_R_qty_scalar.magnitude.real if np.iscomplexobj(current_R_qty_scalar.magnitude) else current_R_qty_scalar.magnitude

        if r_real_mag < 0:
            raise ComponentError(f"Resistance must be non-negative for {self.instance_id} MNA stamping. Got {current_R_qty_scalar:~P}")

        if r_real_mag == 0:
            logger.warning(f"Component '{self.instance_id}' has zero resistance. Treated as ideal short (large finite admittance) for F>=0 analysis in get_mna_stamps().")
            y_scalar_complex = LARGE_ADMITTANCE_SIEMENS + 0j # Correctly uses imported constant
        elif np.isposinf(r_real_mag): # Handle R=inf case for stamping
             logger.debug(f"Component '{self.instance_id}' has infinite resistance. Treated as open (zero admittance) for F>0 analysis in get_mna_stamps().")
             y_scalar_complex = 0j
        else:
            admittance_scalar_qty = (1.0 / current_R_qty_scalar).to(ureg.siemens)
            y_scalar_complex = complex(admittance_scalar_qty.magnitude)

        # Reshape scalar complex value to match freq_hz shape (which is (1,) here)
        y_val_array = np.array([y_scalar_complex], dtype=np.complex128)

        num_freqs_in_stamp = y_val_array.shape[0] # Will be 1
        stamp_mag = np.zeros((num_freqs_in_stamp, 2, 2), dtype=np.complex128)
        stamp_mag[:, 0, 0] = y_val_array
        stamp_mag[:, 0, 1] = -y_val_array
        stamp_mag[:, 1, 0] = -y_val_array
        stamp_mag[:, 1, 1] = y_val_array

        admittance_matrix_qty = Quantity(stamp_mag, ureg.siemens)
        return [(admittance_matrix_qty, [PORT_1, PORT_2])]

    def get_dc_behavior(self, resolved_params: Dict[str, Quantity]) -> Tuple[DCBehaviorType, Optional[Quantity]]:
        # --- (Implementation unchanged) ---
        """
        Determines the Resistor's behavior at DC (F=0) based on its resolved parameters.

        Assumes 'resolved_params' contains a 'resistance' Quantity with a
        1-element NumPy array magnitude, as provided by the caller (e.g., DCAnalyzer).

        Handles specific parameter values as follows:
        - R = 0.0: Returns `(DCBehaviorType.SHORT_CIRCUIT, None)`.
        - R = +inf or R = NaN: Returns `(DCBehaviorType.OPEN_CIRCUIT, None)`.
        - R < 0 (real part): Raises `ComponentError`.
        - R has imag != 0: Raises `ComponentError` (via helper).
        - R is finite positive real: Returns `(DCBehaviorType.ADMITTANCE, Quantity(1/R, 'siemens'))`.
        """
        # Extract scalar real value, checking for non-real components
        r_real_val = _extract_dc_real_scalar_value(
            resolved_params, 'resistance', 'ohm', self.instance_id
        )

        # Classify DC behavior based on the real value
        if np.isnan(r_real_val) or np.isposinf(r_real_val): # Only positive Inf is open
            logger.debug(f"Resistor '{self.instance_id}' DC behavior: OPEN_CIRCUIT (R is NaN or +Inf: {r_real_val}).")
            return (DCBehaviorType.OPEN_CIRCUIT, None)

        if np.isneginf(r_real_val): # Negative Inf resistance is non-physical
             raise ComponentError(f"Negative infinite resistance R={r_real_val} ohm not supported for DC analysis for '{self.instance_id}'.")

        if r_real_val == 0.0:
            logger.debug(f"Resistor '{self.instance_id}' DC behavior: SHORT_CIRCUIT (R == 0.0).")
            return (DCBehaviorType.SHORT_CIRCUIT, None)

        if r_real_val < 0:
            # Fetch original quantity for precise error message formatting
            r_qty_scalar_orig = resolved_params['resistance'][0]
            raise ComponentError(f"Negative resistance R={r_qty_scalar_orig:~P} not supported for DC analysis for '{self.instance_id}'.")

        # Else: finite, positive resistance
        # Retrieve original scalar quantity for unit handling in calculation
        r_qty_scalar = resolved_params['resistance'][0]
        # Perform calculation using the validated real value, but construct Quantity with units
        dc_admittance_val = (1.0 / Quantity(r_real_val, r_qty_scalar.units)).to(ureg.siemens)
        logger.debug(f"Resistor '{self.instance_id}' DC behavior: ADMITTANCE Y_dc={dc_admittance_val:~P} (R={r_qty_scalar:~P} is finite positive).")
        return (DCBehaviorType.ADMITTANCE, dc_admittance_val)

    def is_structurally_open(self, resolved_constant_params: Dict[str, Quantity]) -> bool:
        """Checks if R is ideally infinite based on constant parameters."""
        if 'resistance' not in resolved_constant_params:
            logger.debug(f"Resistor '{self.instance_id}' is_structurally_open: 'resistance' not in resolved_constant_params (e.g., frequency-dependent). Not structurally open.")
            return False
        try:
            r_qty = resolved_constant_params['resistance']
            r_mag = r_qty.magnitude
            if isinstance(r_mag, np.ndarray):
                if r_mag.size == 0: return False 
                r_val = r_mag.item() 
            else:
                r_val = r_mag 

            is_open = np.isposinf(r_val)
            logger.debug(f"Resistor '{self.instance_id}' is_structurally_open check: R={r_qty:~P} -> {is_open}")
            return bool(is_open)
        except Exception as e: # Catch other potential errors, e.g. if r_val is not a number np.isposinf can handle
             raise ComponentError(f"Error checking structural open state for resistor '{self.instance_id}' with R='{resolved_constant_params.get('resistance', 'N/A'):~P}': {e}") from e


@register_component("Capacitor")
class Capacitor(ComponentBase):
    """Ideal linear capacitor."""

    @classmethod
    def declare_parameters(cls) -> Dict[str, str]:
        return {"capacitance": "farad"}

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
        # --- (Implementation unchanged, uses LARGE_ADMITTANCE_SIEMENS correctly) ---
        if not isinstance(freq_hz, np.ndarray):
             raise TypeError(f"{self.instance_id}: freq_hz must be a NumPy array.")
        if not isinstance(resolved_params, dict):
            raise TypeError(f"{self.instance_id}: resolved_params must be a dictionary.")

        try:
            capacitance_qty = resolved_params['capacitance']
        except KeyError:
            raise ComponentError(f"Parameter 'capacitance' not found in resolved_params for {self.instance_id}.")

        # Validate shape just in case
        if not isinstance(capacitance_qty.magnitude, np.ndarray) or capacitance_qty.magnitude.shape != (1,):
             logger.warning(f"{self.instance_id}: Capacitance quantity magnitude shape {getattr(capacitance_qty.magnitude, 'shape', type(capacitance_qty.magnitude))} unexpected, expected (1,). Using first element or failing if not possible.")
             if not isinstance(capacitance_qty.magnitude, np.ndarray) or capacitance_qty.magnitude.size < 1:
                 raise ComponentError(f"Capacitance parameter '{self.instance_id}' has invalid magnitude shape/size for MNA stamping.")
             current_C_qty_scalar = capacitance_qty[0] # Attempt extraction anyway
        else:
            current_C_qty_scalar = capacitance_qty[0]

        # Check realness and non-negativity, allowing +Inf for C
        if np.iscomplexobj(current_C_qty_scalar.magnitude) and current_C_qty_scalar.magnitude.imag != 0:
             raise ComponentError(f"Capacitance must be real for {self.instance_id} MNA stamping. Got {current_C_qty_scalar:~P}")

        c_real_mag = current_C_qty_scalar.magnitude.real if np.iscomplexobj(current_C_qty_scalar.magnitude) else current_C_qty_scalar.magnitude

        is_infinite_cap = np.isposinf(c_real_mag) # Only positive infinity is physical short

        if c_real_mag < 0: # Negative C (including -Inf) is non-physical
             raise ComponentError(f"Capacitance must be non-negative for {self.instance_id} MNA stamping. Got {current_C_qty_scalar:~P}")

        if is_infinite_cap:
            logger.warning(f"Component '{self.instance_id}' has positive infinite capacitance. Treated as ideal short (large finite admittance) for F>=0 analysis in get_mna_stamps().")
            y_scalar_complex = LARGE_ADMITTANCE_SIEMENS + 0j # Correctly uses imported constant
        elif c_real_mag == 0: # Handle C=0 for stamping (open circuit)
             logger.debug(f"Component '{self.instance_id}' has zero capacitance. Treated as open (zero admittance) for F>0 analysis in get_mna_stamps().")
             y_scalar_complex = 0j
        else:
            # Standard admittance calculation: Y = jwC
            omega = (2 * np.pi * freq_hz) * (ureg.rad / ureg.second)
            # Use the original Quantity vector (which is shape (1,) here) for element-wise multiplication
            admittance_qty_vector = (1j * omega * capacitance_qty).to(ureg.siemens)
            y_scalar_complex = complex(admittance_qty_vector.magnitude[0]) # Extract the single complex value

        # Reshape scalar complex value to match freq_hz shape (which is (1,) here)
        y_val_array = np.array([y_scalar_complex], dtype=np.complex128)

        num_freqs_in_stamp = y_val_array.shape[0] # Will be 1
        stamp_mag = np.zeros((num_freqs_in_stamp, 2, 2), dtype=np.complex128)
        stamp_mag[:, 0, 0] = y_val_array
        stamp_mag[:, 0, 1] = -y_val_array
        stamp_mag[:, 1, 0] = -y_val_array
        stamp_mag[:, 1, 1] = y_val_array

        admittance_matrix_qty = Quantity(stamp_mag, ureg.siemens)
        return [(admittance_matrix_qty, [PORT_1, PORT_2])]

    def get_dc_behavior(self, resolved_params: Dict[str, Quantity]) -> Tuple[DCBehaviorType, Optional[Quantity]]:
        # --- (Implementation unchanged) ---
        """
        Determines the Capacitor's behavior at DC (F=0) based on its resolved parameters.

        Assumes 'resolved_params' contains a 'capacitance' Quantity with a
        1-element NumPy array magnitude, as provided by the caller (e.g., DCAnalyzer).

        Handles specific parameter values as follows:
        - C = +inf: Returns `(DCBehaviorType.SHORT_CIRCUIT, None)`.
        - C = 0, C = finite > 0, or C = NaN: Returns `(DCBehaviorType.OPEN_CIRCUIT, None)`.
        - C = -inf or C < 0 (real part): Raises `ComponentError`.
        - C has imag != 0: Raises `ComponentError` (via helper).
        """
        # Extract scalar real value, checking for non-real components
        c_real_val = _extract_dc_real_scalar_value(
            resolved_params, 'capacitance', 'farad', self.instance_id
        )

        # Classify DC behavior based on the real value
        if np.isnan(c_real_val):
            logger.debug(f"Capacitor '{self.instance_id}' DC behavior: OPEN_CIRCUIT (C is NaN).")
            return (DCBehaviorType.OPEN_CIRCUIT, None)

        if np.isinf(c_real_val):
            if c_real_val > 0: # C = +inf
                logger.debug(f"Capacitor '{self.instance_id}' DC behavior: SHORT_CIRCUIT (C is +Inf).")
                return (DCBehaviorType.SHORT_CIRCUIT, None)
            else: # C = -inf, non-physical
                raise ComponentError(f"Negative infinite capacitance C={c_real_val} F not supported for DC analysis for '{self.instance_id}'.")

        if c_real_val < 0: # Finite negative capacitance
            c_qty_scalar_orig = resolved_params['capacitance'][0]
            raise ComponentError(f"Negative capacitance C={c_qty_scalar_orig:~P} not supported for DC analysis for '{self.instance_id}'.")

        # Else (C_real_val >= 0 and finite, including C=0) -> Open Circuit at DC
        c_qty_scalar_orig = resolved_params['capacitance'][0]
        logger.debug(f"Capacitor '{self.instance_id}' DC behavior: OPEN_CIRCUIT (C={c_qty_scalar_orig:~P} is finite and >= 0).")
        return (DCBehaviorType.OPEN_CIRCUIT, None)

    def is_structurally_open(self, resolved_constant_params: Dict[str, Quantity]) -> bool:
        """Checks if C is ideally zero based on constant parameters."""
        if 'capacitance' not in resolved_constant_params:
            logger.debug(f"Capacitor '{self.instance_id}' is_structurally_open: 'capacitance' not in resolved_constant_params (e.g., frequency-dependent). Not structurally open.")
            return False
        try:
            c_qty = resolved_constant_params['capacitance']
            c_mag = c_qty.magnitude
            if isinstance(c_mag, np.ndarray):
                 if c_mag.size == 0: return False 
                 c_val = c_mag.item()
            else:
                c_val = c_mag 

            is_open = (c_val == 0)
            logger.debug(f"Capacitor '{self.instance_id}' is_structurally_open check: C={c_qty:~P} -> {is_open}")
            return bool(is_open)
        except Exception as e:
             raise ComponentError(f"Error checking structural open state for capacitor '{self.instance_id}' with C='{resolved_constant_params.get('capacitance', 'N/A'):~P}': {e}") from e


@register_component("Inductor")
class Inductor(ComponentBase):
    """Ideal linear inductor."""

    @classmethod
    def declare_parameters(cls) -> Dict[str, str]:
        return {"inductance": "henry"}

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
        # --- (Implementation essentially unchanged, uses LARGE_ADMITTANCE_SIEMENS correctly) ---
        if not isinstance(freq_hz, np.ndarray):
             raise TypeError(f"{self.instance_id}: freq_hz must be a NumPy array.")
        if not isinstance(resolved_params, dict):
            raise TypeError(f"{self.instance_id}: resolved_params must be a dictionary.")

        try:
            inductance_qty = resolved_params['inductance']
        except KeyError:
            raise ComponentError(f"Parameter 'inductance' not found in resolved_params for {self.instance_id}.")

        # Validate shape just in case
        if not isinstance(inductance_qty.magnitude, np.ndarray) or inductance_qty.magnitude.shape != (1,):
             logger.warning(f"{self.instance_id}: Inductance quantity magnitude shape {getattr(inductance_qty.magnitude, 'shape', type(inductance_qty.magnitude))} unexpected, expected (1,). Using first element or failing if not possible.")
             if not isinstance(inductance_qty.magnitude, np.ndarray) or inductance_qty.magnitude.size < 1:
                 raise ComponentError(f"Inductance parameter '{self.instance_id}' has invalid magnitude shape/size for MNA stamping.")
             current_L_qty_scalar = inductance_qty[0] # Attempt extraction anyway
        else:
            current_L_qty_scalar = inductance_qty[0]

        # Check realness and non-negativity
        if np.iscomplexobj(current_L_qty_scalar.magnitude) and current_L_qty_scalar.magnitude.imag != 0:
             raise ComponentError(f"Inductance must be real for {self.instance_id} MNA stamping. Got {current_L_qty_scalar:~P}")

        l_real_mag = current_L_qty_scalar.magnitude.real if np.iscomplexobj(current_L_qty_scalar.magnitude) else current_L_qty_scalar.magnitude

        if l_real_mag < 0:
            raise ComponentError(f"Inductance must be non-negative for {self.instance_id} MNA stamping. Got {current_L_qty_scalar:~P}")

        is_zero_inductance = (l_real_mag == 0)
        is_infinite_inductance = np.isposinf(l_real_mag) # Handle L=inf for stamping

        if is_zero_inductance:
             logger.warning(f"Component '{self.instance_id}' has zero inductance. Treated as ideal short (large finite admittance) for F>=0 analysis in get_mna_stamps().")
             y_scalar_complex = LARGE_ADMITTANCE_SIEMENS + 0j # Correctly uses imported constant
        elif is_infinite_inductance:
             logger.debug(f"Component '{self.instance_id}' has infinite inductance. Treated as open (zero admittance) for F>0 analysis in get_mna_stamps().")
             y_scalar_complex = 0j
        else:
             # Calculate impedance Z = jwL
             omega = (2 * np.pi * freq_hz) * (ureg.rad / ureg.second) # omega is shape (1,)
             # inductance_qty is shape (1,)
             impedance_qty_vector = (1j * omega * inductance_qty) # Result shape (1,)
             impedance_scalar_qty = impedance_qty_vector[0] # Extract scalar Quantity

             # Calculate admittance Y = 1/Z, handling near-zero impedance (e.g., at F=0 if L > 0)
             # Use a small threshold for impedance magnitude to switch to large admittance
             # Note: If freq_hz is exactly 0, omega will be 0, impedance will be 0.
             if abs(impedance_scalar_qty.magnitude) < 1e-18:
                 y_scalar_complex = LARGE_ADMITTANCE_SIEMENS + 0j # Correctly uses imported constant
                 # Only log warning if F>0, F=0 case is expected for L>0
                 if freq_hz.size > 0 and freq_hz[0] > 0:
                     logger.warning(f"Inductor '{self.instance_id}' impedance {impedance_scalar_qty:~P} is near zero at F={freq_hz[0]:.2e} Hz. Using LARGE_ADMITTANCE.")
             else:
                 admittance_scalar_qty = (1.0 / impedance_scalar_qty).to(ureg.siemens)
                 y_scalar_complex = complex(admittance_scalar_qty.magnitude)

        # Reshape scalar complex value to match freq_hz shape (which is (1,) here)
        y_val_array = np.array([y_scalar_complex], dtype=np.complex128)

        num_freqs_in_stamp = y_val_array.shape[0] # Will be 1
        stamp_mag = np.zeros((num_freqs_in_stamp, 2, 2), dtype=np.complex128)
        stamp_mag[:, 0, 0] = y_val_array
        stamp_mag[:, 0, 1] = -y_val_array
        stamp_mag[:, 1, 0] = -y_val_array
        stamp_mag[:, 1, 1] = y_val_array

        admittance_matrix_qty = Quantity(stamp_mag, ureg.siemens)
        return [(admittance_matrix_qty, [PORT_1, PORT_2])]

    def get_dc_behavior(self, resolved_params: Dict[str, Quantity]) -> Tuple[DCBehaviorType, Optional[Quantity]]:
        # --- (Implementation unchanged) ---
        """
        Determines the Inductor's behavior at DC (F=0) based on its resolved parameters.

        Assumes 'resolved_params' contains an 'inductance' Quantity with a
        1-element NumPy array magnitude, as provided by the caller (e.g., DCAnalyzer).

        Handles specific parameter values as follows:
        - L = 0 or L = finite > 0: Returns `(DCBehaviorType.SHORT_CIRCUIT, None)`.
        - L = +inf or L = NaN: Returns `(DCBehaviorType.OPEN_CIRCUIT, None)`.
        - L = -inf or L < 0 (real part): Raises `ComponentError`.
        - L has imag != 0: Raises `ComponentError` (via helper).
        """
        # Extract scalar real value, checking for non-real components
        l_real_val = _extract_dc_real_scalar_value(
            resolved_params, 'inductance', 'henry', self.instance_id
        )

        # Classify DC behavior based on the real value
        if np.isnan(l_real_val) or np.isposinf(l_real_val): # L = +inf or L = NaN
            logger.debug(f"Inductor '{self.instance_id}' DC behavior: OPEN_CIRCUIT (L is NaN or +Inf: {l_real_val}).")
            return (DCBehaviorType.OPEN_CIRCUIT, None)

        if np.isneginf(l_real_val): # L = -inf, non-physical
             raise ComponentError(f"Negative infinite inductance L={l_real_val} H not supported for DC analysis for '{self.instance_id}'.")

        if l_real_val < 0: # Finite negative inductance
            l_qty_scalar_orig = resolved_params['inductance'][0]
            raise ComponentError(f"Negative inductance L={l_qty_scalar_orig:~P} not supported for DC analysis for '{self.instance_id}'.")

        # Else (L_real_val >= 0 and finite, including L=0) -> Short Circuit at DC
        l_qty_scalar_orig = resolved_params['inductance'][0]
        logger.debug(f"Inductor '{self.instance_id}' DC behavior: SHORT_CIRCUIT (L={l_qty_scalar_orig:~P} is finite and >= 0).")
        return (DCBehaviorType.SHORT_CIRCUIT, None)

    def is_structurally_open(self, resolved_constant_params: Dict[str, Quantity]) -> bool:
        """Checks if L is ideally infinite based on constant parameters."""
        if 'inductance' not in resolved_constant_params:
            logger.debug(f"Inductor '{self.instance_id}' is_structurally_open: 'inductance' not in resolved_constant_params (e.g., frequency-dependent). Not structurally open.")
            return False
        try:
            l_qty = resolved_constant_params['inductance']
            l_mag = l_qty.magnitude
            if isinstance(l_mag, np.ndarray):
                 if l_mag.size == 0: return False 
                 l_val = l_mag.item()
            else:
                l_val = l_mag 

            is_open = np.isposinf(l_val)
            logger.debug(f"Inductor '{self.instance_id}' is_structurally_open check: L={l_qty:~P} -> {is_open}")
            return bool(is_open)
        except Exception as e:
             raise ComponentError(f"Error checking structural open state for inductor '{self.instance_id}' with L='{resolved_constant_params.get('inductance', 'N/A'):~P}': {e}") from e