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
            The same Circuit object, now populated with a dictionary
            of simulation-ready component instances in the `sim_components` attribute.

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

        # --- Perform Build-Specific Validations and Instantiation ---
        for instance_id, comp_data in parsed_data.components.items():
            comp_type_str = comp_data.component_type
            logger.debug(f"Building component '{instance_id}' of type '{comp_type_str}'")

            # Component type already validated by parser, safe to access registry
            try:
                ComponentClass = COMPONENT_REGISTRY[comp_type_str]
            except KeyError:
                raise CircuitBuildError(f"Uknown Component Type: {comp_type_str}")

            # 1. Validate Ports Used vs Declared
            try:
                declared_ports_set = set(ComponentClass.declare_ports())
                used_ports_set = set(comp_data.ports.keys())

                # Check for ports used in instance but not declared by type
                extra_ports = used_ports_set - declared_ports_set
                if extra_ports:
                     errors.append(f"Component '{instance_id}' (type '{comp_type_str}') uses undeclared ports: {sorted(list(extra_ports))}. Declared ports are: {sorted(list(declared_ports_set))}.")

                # Check for declared ports that *must* be connected but aren't
                # For now, we only check if used ports are valid. A stricter check might be:
                # missing_ports = declared_ports_set - used_ports_set
                # if missing_ports:
                #    errors.append(f"Component '{instance_id}' (type '{comp_type_str}') is missing connections for declared ports: {missing_ports}.")
                # Let's keep it relaxed for now: only used ports must be declared.

            except Exception as e:
                errors.append(f"Error validating ports for component '{instance_id}': {e}")
                continue # Skip this component if port validation fails critically

            # 2. Process and Validate Parameters (Dimensionality Check)
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
                # Catch errors during component's __init__
                errors.append(f"Instantiation failed for component '{instance_id}': {e}")
                continue # Skip this component

        if errors:
            error_msg = f"Circuit build failed for '{parsed_data.name}' with {len(errors)} errors:\n- " + "\n- ".join(errors)
            logger.error(error_msg)
            raise CircuitBuildError(error_msg)
        
        # Add the processed components to the circuit object
        setattr(parsed_data, 'sim_components', processed_components) # Keep mutation approach for now
        logger.info(f"Successfully built circuit '{parsed_data.name}'. Created {len(processed_components)} simulation-ready components.")

        return parsed_data  

    def _process_component_parameters(
        self,
        comp_data: ComponentData,
        declared_params: Dict[str, str],
        param_manager: ParameterManager
    ) -> Dict[str, Quantity]:
        """
        Processes raw parameters for a component instance, performs dimensional validation.
        """
        processed: Dict[str, Quantity] = {}
        raw_params = comp_data.parameters
        instance_id = comp_data.instance_id

        # Check for unexpected parameters provided in the netlist
        for raw_param_name in raw_params:
            if raw_param_name not in declared_params:
                # This should now be a warning, as some components might allow optional params?
                # Or stricter: Error if not declared. Let's stick to warning.
                logger.warning(f"Component '{instance_id}' provided parameter '{raw_param_name}' which is not declared by type '{comp_data.component_type}'. Declared: {list(declared_params.keys())}. Ignoring.")

        # Process declared parameters
        for param_name, expected_dim_str in declared_params.items():
            if param_name not in raw_params:
                 # Make required parameters explicit? Assume all declared are required for now.
                raise ParameterError(f"Required parameter '{param_name}' missing for component '{instance_id}'. Declared parameters: {list(declared_params.keys())}")

            raw_value = raw_params[param_name]
            resolved_quantity: Optional[Quantity] = None

            # Try resolving as global parameter first if it's a string
            if isinstance(raw_value, str):
                try:
                    # Future: This is where expression evaluation would go. For now, just lookup.
                    resolved_quantity = param_manager.get_parameter(raw_value)
                    logger.debug(f"Resolved parameter '{param_name}' for '{instance_id}' using global parameter '{raw_value}' -> {resolved_quantity:~P}")
                except KeyError:
                    # Not a global parameter name, treat as a literal value string
                    pass
                except ParameterError as e: # Catch errors from future expression eval
                     raise ParameterError(f"Error resolving global/expression parameter '{raw_value}' for '{param_name}' in '{instance_id}': {e}") from e


            # If not resolved globally, parse as a literal value
            if resolved_quantity is None:
                try:
                    value_str = str(raw_value)
                    resolved_quantity = self._ureg.Quantity(value_str)
                    logger.debug(f"Parsed literal parameter '{param_name}' for '{instance_id}': '{value_str}' -> {resolved_quantity:~P}")
                except Exception as e:
                    raise ParameterError(f"Error parsing literal value for parameter '{param_name}' ('{raw_value}') in component '{instance_id}': {e}") from e

            # Validate dimensionality
            try:
                # Use Pint's dimensionality check directly
                if not resolved_quantity.is_compatible_with(expected_dim_str):
                    # Provide more context in the error
                    raise pint.DimensionalityError(
                        resolved_quantity.units,
                        self._ureg.parse_expression(expected_dim_str).units, # Get target units for better msg
                        resolved_quantity.dimensionality,
                        self._ureg.parse_expression(expected_dim_str).dimensionality,
                        extra_msg=f"Parameter '{param_name}' in component '{instance_id}'"
                    )

                processed[param_name] = resolved_quantity
                logger.debug(f"Validated parameter '{param_name}' for '{instance_id}'. Value: {resolved_quantity:~P}, Expected Dim: '{expected_dim_str}'")

            except pint.DimensionalityError as e:
                # Construct a more informative message
                err_msg = (
                    f"Dimensionality mismatch for {e.extra_msg}. " # Use the msg field we set
                    f"Got value {resolved_quantity:~P} (dimensionality {e.dim1}), "
                    f"but expected dimension compatible with '{expected_dim_str}' ({e.dim2})."
                )
                logger.error(err_msg)
                # Re-raise with the enhanced message if possible, or wrap it
                raise pint.DimensionalityError(e.units1, e.units2, e.dim1, e.dim2, err_msg) from e

        return processed