# src/rfsim_core/units.py
import pint
import logging

logger = logging.getLogger(__name__)

# Create a single unit registry instance
# You can customize this later (e.g., add custom definitions)
ureg = pint.UnitRegistry()
logger.info("Pint Unit Registry initialized.")

# You could add Quantity type alias for convenience if desired
Quantity = ureg.Quantity