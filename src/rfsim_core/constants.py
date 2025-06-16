# --- Create file: src/rfsim_core/constants.py ---
import logging
from .units import ureg, Quantity

logger = logging.getLogger(__name__)

# --- Numerical Constants for Simulation ---

#: Large finite admittance value used to numerically represent ideal shorts (e.g., R=0, C=inf, L=0 at F>=0)
#: in the MNA matrix for F>0 calculations, or for DC-shorted AC ports in results mapping.
#: Value: 1e12 Siemens (equivalent to 1 micro-ohm impedance).
LARGE_ADMITTANCE_SIEMENS: float = 1.0e12 # Siemens

#: Threshold for admittance values considered effectively zero during numerical checks.
#: Used for future numerical stability improvements (e.g., identifying near-zero connections
#: that might cause ill-conditioning but are not strictly structural opens).
#: Not actively used for ideal classification in Phase 7.
#: Value: 1e-18 Siemens (equivalent to 1 Tera-ohm impedance).
ZERO_ADMITTANCE_THRESHOLD_SIEMENS: float = 1.0e-18 # Siemens

# You could add physical constants here later if needed, e.g.:
# SPEED_OF_LIGHT = Quantity(299792458.0, 'm/s')
# VACUUM_PERMITTIVITY = Quantity(8.8541878128e-12, 'F/m')
# VACUUM_PERMEABILITY = Quantity(1.25663706212e-6, 'N/A**2') # approx 4*pi*1e-7 H/m

logger.debug("Defined core constants: LARGE_ADMITTANCE_SIEMENS, ZERO_ADMITTANCE_THRESHOLD_SIEMENS")