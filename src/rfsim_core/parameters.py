# src/rfsim_core/parameters.py
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
    Initially handles only constant values with units.
    """
    def __init__(self, global_parameters: Dict[str, Any]):
        """
        Initializes the ParameterManager.

        Args:
            global_parameters: A dictionary of global parameter names to their
                               values (expected as strings or numbers for parsing).
        """
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
                # Convert value to string if it's a number, Pint handles unitless numbers too
                value_str = str(value)
                quantity = self._ureg.Quantity(value_str)
                self._parameters[name] = quantity
                logger.debug(f"Parsed global parameter '{name}': {quantity:~P}") # Pretty print
            except (pint.errors.UndefinedUnitError, pint.errors.DimensionalityError) as e:
                # Catch potential unit errors during parsing
                err_msg = f"Error parsing global parameter '{name}' with value '{value}': {e}"
                logger.error(err_msg)
                raise ParameterError(err_msg) from e
            except Exception as e:
                # Catch other unexpected errors during Quantity creation
                err_msg = f"Unexpected error parsing global parameter '{name}' with value '{value}': {e}"
                logger.error(err_msg)
                raise ParameterError(err_msg) from e

    def get_parameter(self, name: str) -> pint.Quantity:
        """
        Retrieves a parsed global parameter as a Pint Quantity.

        Args:
            name: The name of the parameter.

        Returns:
            The parameter value as a Pint Quantity.

        Raises:
            KeyError: If the parameter name is not found.
            ParameterError: For other parameter access issues (reserved for future).
        """
        if name not in self._parameters:
            raise KeyError(f"Global parameter '{name}' not found.")
        # In the future, this might involve evaluation if parameters were expressions
        return self._parameters[name]

    def get_all_parameters(self) -> Dict[str, pint.Quantity]:
        """Returns a copy of the internal parameters dictionary."""
        return self._parameters.copy()

    def __str__(self) -> str:
        items = [f"  {name}: {qty:~P}" for name, qty in self._parameters.items()]
        return "ParameterManager:\n" + "\n".join(items)