# src/rfsim_core/circuit_builder.py
import logging
from typing import Dict, Any, Optional

from .data_structures import Circuit as ParsedCircuitData # Input is parsed data
from .data_structures import Component as ComponentData
# Import the actual simulation-ready component base/registry
from .components.base import ComponentBase, COMPONENT_REGISTRY, ComponentError
from .parameters import ParameterManager, ParameterError
from .units import ureg, pint, Quantity

logger = logging.getLogger(__name__)

class CircuitBuildError(ValueError):
    """Custom exception for errors during circuit building/processing."""
    pass

class CircuitBuilder:
    """
    Takes parsed circuit data and builds a simulation-ready Circuit object
    containing processed components with validated parameters.
    """
    def __init__(self):
        self._ureg = ureg # Use the shared registry
        logger.info("CircuitBuilder initialized.")

    def build_circuit(self, parsed_data: ParsedCircuitData) -> ParsedCircuitData:
        """
        Processes the parsed circuit data, validates component parameters,
        and populates the circuit with simulation-ready component instances.

        Args:
            parsed_data: The Circuit object returned by NetlistParser, containing
                         raw component data.

        Returns:
            The same Circuit object, now potentially modified with a dictionary
            of simulation-ready component instances (e.g., in a new attribute).
            Or, we can replace the raw components dict. Let's add a new attribute.

        Raises:
            CircuitBuildError: If any errors occur during processing.
        """
        logger.info(f"Building simulation-ready circuit '{parsed_data.name}'...")
        param_manager = parsed_data.parameter_manager
        if param_manager is None:
             # Should not happen if parser worked, but defensively check
             raise CircuitBuildError("ParameterManager not found in parsed circuit data.")

        processed_components: Dict[str, ComponentBase] = {}
        errors = []

        for instance_id, comp_data in parsed_data.components.items():
            comp_type_str = comp_data.component_type
            logger.debug(f"Processing component '{instance_id}' of type '{comp_type_str}'")

            # 1. Find Component Class from Registry
            if comp_type_str not in COMPONENT_REGISTRY:
                errors.append(f"Unknown component type '{comp_type_str}' for instance '{instance_id}'. Registered types: {list(COMPONENT_REGISTRY.keys())}")
                continue # Skip this component
            ComponentClass = COMPONENT_REGISTRY[comp_type_str]

            # 2. Process and Validate Parameters
            try:
                declared_params = ComponentClass.declare_parameters()
                processed_params = self._process_component_parameters(
                    comp_data, declared_params, param_manager
                )
            except (ParameterError, ComponentError, pint.DimensionalityError) as e:
                errors.append(f"Parameter validation failed for component '{instance_id}': {e}")
                continue # Skip this component

            # 3. Instantiate Simulation-Ready Component
            try:
                sim_component = ComponentClass(
                    instance_id=instance_id,
                    component_type=comp_type_str,
                    processed_params=processed_params
                )
                processed_components[instance_id] = sim_component
                logger.debug(f"Successfully created simulation component: {sim_component!r}")
            except (ComponentError, ValueError) as e:
                errors.append(f"Instantiation failed for component '{instance_id}': {e}")
                continue # Skip this component

        if errors:
            error_msg = f"Circuit build failed for '{parsed_data.name}' with {len(errors)} errors:\n- " + "\n- ".join(errors)
            logger.error(error_msg)
            raise CircuitBuildError(error_msg)

        # Add the processed components to the circuit object
        # Let's add a new attribute to avoid ambiguity with the raw data
        setattr(parsed_data, 'sim_components', processed_components)
        logger.info(f"Successfully built circuit '{parsed_data.name}'. Found {len(processed_components)} simulation-ready components.")

        return parsed_data


    def _process_component_parameters(
        self,
        comp_data: ComponentData,
        declared_params: Dict[str, str],
        param_manager: ParameterManager
    ) -> Dict[str, Quantity]:
        """
        Processes raw parameters for a component instance, validates units,
        and resolves global references.
        """
        processed: Dict[str, Quantity] = {}
        raw_params = comp_data.parameters
        instance_id = comp_data.instance_id

        # Check for unexpected parameters provided in the netlist
        for raw_param_name in raw_params:
            if raw_param_name not in declared_params:
                logger.warning(f"Component '{instance_id}' provided parameter '{raw_param_name}' which is not declared by type '{comp_data.component_type}'. Ignoring.")

        # Process declared parameters
        for param_name, expected_dim_str in declared_params.items():
            if param_name not in raw_params:
                raise ParameterError(f"Required parameter '{param_name}' missing for component '{instance_id}'.")

            raw_value = raw_params[param_name]
            resolved_quantity: Optional[Quantity] = None

            # Try resolving as global parameter first if it's a string
            if isinstance(raw_value, str):
                try:
                    resolved_quantity = param_manager.get_parameter(raw_value)
                    logger.debug(f"Resolved parameter '{param_name}' for '{instance_id}' using global parameter '{raw_value}' -> {resolved_quantity:~P}")
                except KeyError:
                    # Not a global parameter name, treat as a literal value string
                    pass

            # If not resolved globally, parse as a literal value
            if resolved_quantity is None:
                try:
                    # Ensure value is string for Pint parsing
                    value_str = str(raw_value)
                    resolved_quantity = self._ureg.Quantity(value_str)
                    logger.debug(f"Parsed literal parameter '{param_name}' for '{instance_id}': '{value_str}' -> {resolved_quantity:~P}")
                except Exception as e:
                    raise ParameterError(f"Error parsing literal value for parameter '{param_name}' ('{raw_value}') in component '{instance_id}': {e}") from e

            # Validate dimensionality
            try:
                # Use check() for explicit dimension matching (or compatible dimensions)
                # Pint's check() raises DimensionalityError if not compatible.
                if not resolved_quantity.is_compatible_with(expected_dim_str):
                    expected_unit = self._ureg.parse_expression(expected_dim_str).units
                    raise pint.DimensionalityError(
                        resolved_quantity.units,
                        expected_unit,
                        resolved_quantity.dimensionality,
                        expected_unit.dimensionality,
                        (
                            f"Dimensionality mismatch for parameter '{param_name}' in "
                            f"component '{instance_id}'. Got {resolved_quantity:~P} "
                            f"but expected dimension compatible with '{expected_dim_str}'."
                        ),
                    )

                # Optional: Convert to the exact expected units if desired, but compatibility is key
                # resolved_quantity = resolved_quantity.to(expected_dim_str) # This might fail if units are compatible but not identical (e.g., km vs m)
                # Compatibility check is usually sufficient.

                processed[param_name] = resolved_quantity
                logger.debug(f"Validated parameter '{param_name}' for '{instance_id}'. Value: {resolved_quantity:~P}, Expected Dim: '{expected_dim_str}'")

            except pint.DimensionalityError as e:
                # Provide a more informative error message
                err_msg = (
                    f"Dimensionality mismatch for parameter '{param_name}' in component '{instance_id}'. "
                    f"Got value {resolved_quantity:~P} (dimensionality {resolved_quantity.dimensionality}), "
                    f"but expected dimension compatible with '{expected_dim_str}' ({self._ureg.parse_expression(expected_dim_str).dimensionality})."
                )
                logger.error(err_msg)
                # Raise a new error wrapping the Pint error for better context
                raise pint.DimensionalityError(e.units1, e.units2, e.dim1, e.dim2, err_msg) from e


        return processed