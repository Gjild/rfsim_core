# --- src/rfsim_core/units.py ---
import pint
import logging

logger = logging.getLogger(__name__)
ureg = pint.UnitRegistry()
Quantity = ureg.Quantity
logger.info("Pint Unit Registry initialized.")