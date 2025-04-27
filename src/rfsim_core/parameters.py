# --- src/rfsim_core/parameters.py ---
# (No changes needed for Phase 4 fixes)
import logging
from typing import Dict, Any
import pint

from .units import ureg # Import the shared unit registry

logger = logging.getLogger(__name__)

class ParameterError(Exception):
    """Custom exception for parameter-related errors."""
    pass

class ParameterManager:
    """
    Manages global circuit parameters with unit handling.
    Phase 4: Handles only constant values with units.
    """
    def __init__(self, global_parameters: Dict[str, Any]):
        self._parameters: Dict[str, pint.Quantity] = {}
        self._ureg = ureg # Use the shared registry
        logger.info(f"Initializing ParameterManager with {len(global_parameters)} global params.")
        self._parse_global_parameters(global_parameters)

    def _parse_global_parameters(self, params: Dict[str, Any]):
        """Parses raw global parameters into Pint Quantities."""
        for name, value in params.items():
            if not isinstance(name, str) or not name:
                raise ParameterError("Global parameter name must be a non-empty string.")
            try:
                value_str = str(value)
                quantity = self._ureg.Quantity(value_str)
                self._parameters[name] = quantity
                logger.debug(f"Parsed global parameter '{name}': {quantity:~P}") # Pretty print
            except (pint.errors.UndefinedUnitError, pint.errors.DimensionalityError) as e:
                err_msg = f"Error parsing global parameter '{name}' with value '{value}': {e}"
                logger.error(err_msg)
                raise ParameterError(err_msg) from e
            except Exception as e:
                err_msg = f"Unexpected error parsing global parameter '{name}' with value '{value}': {e}"
                logger.error(err_msg)
                raise ParameterError(err_msg) from e

    def get_parameter(self, name: str) -> pint.Quantity:
        """ Retrieves a parsed global parameter as a Pint Quantity. """
        if name not in self._parameters:
            raise KeyError(f"Global parameter '{name}' not found.")
        return self._parameters[name]

    def get_all_parameters(self) -> Dict[str, pint.Quantity]:
        """Returns a copy of the internal parameters dictionary."""
        return self._parameters.copy()

    def __str__(self) -> str:
        items = [f"  {name}: {qty:~P}" for name, qty in self._parameters.items()]
        return "ParameterManager:\n" + "\n".join(items)