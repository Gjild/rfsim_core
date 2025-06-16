# --- Modify: src/rfsim_core/units.py ---
import pint
import logging

logger = logging.getLogger(__name__)
ureg = pint.UnitRegistry()
Quantity = ureg.Quantity
logger.info("Pint Unit Registry initialized.")


# Define [admittance] dimension to align with project terminology.
# This relates the name '[admittance]' to the physical dimension of current/voltage.
ureg.define("[admittance] = [current] / [voltage]")

# Define [impedance] dimension to align with project terminology.
ureg.define("[impedance] = [voltage] / [current]")

# --- Canonical dimensionality objects for explicit checks ---
# These store the frozendict representation of the dimensions.
ADMITTANCE_DIMENSIONALITY = ureg.parse_expression('siemens').dimensionality
IMPEDANCE_DIMENSIONALITY = ureg.parse_expression('ohm').dimensionality

logger.debug(f"Defined canonical dimensionalities: ADMITTANCE_DIMENSIONALITY, IMPEDANCE_DIMENSIONALITY")

# If __all__ is used in this file, add them:
# __all__ = ["ureg", "Quantity", "ADMITTANCE_DIMENSIONALITY", "IMPEDANCE_DIMENSIONALITY"]