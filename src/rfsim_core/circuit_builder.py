import logging
from typing import Dict, Any, Optional

from .data_structures import Circuit
from .data_structures import Component as ComponentData
# Import the actual simulation-ready component base/registry
from .components.base import ComponentBase, COMPONENT_REGISTRY, ComponentError
from .parameters import ParameterManager, ParameterError
from .units import ureg, pint, Quantity
import copy

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

    def build_circuit(self, parsed_data: Circuit) -> Circuit:
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
        

        # --- Create a new Circuit object to avoid mutating the input ---
        # Perform a deep copy of essential structural elements.
        # ParameterManager is assumed immutable or safe to share reference.
        # sim_components will be added specifically.
        try:
            sim_circuit = Circuit(
                name=parsed_data.name,
                nets=copy.deepcopy(parsed_data.nets),
                external_ports=copy.deepcopy(parsed_data.external_ports),
                external_port_impedances=copy.deepcopy(parsed_data.external_port_impedances),
                parameter_manager=param_manager, # Share reference
                ground_net_name=parsed_data.ground_net_name,
                # Deep copy components (raw data) if needed downstream, otherwise could be shallow
                components=copy.deepcopy(parsed_data.components)
            )
            # Add dict for components for simulation
            setattr(sim_circuit, 'sim_components', {})
            logger.debug(f"Created new Circuit object '{sim_circuit.name}' for simulation.")
        except Exception as e:
            logger.error(f"Failed to create base simulation circuit object: {e}", exc_info=True)
            raise CircuitBuildError(f"Failed to create base simulation circuit object: {e}") from e

        processed_components: Dict[str, ComponentBase] = {}
        errors = []

        # --- Perform Build-Specific Validations and Instantiation ---
        for instance_id, comp_data in parsed_data.components.items(): # Iterate original data
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

                # Check for declared ports that *must* be connected but aren't**
                missing_ports = declared_ports_set - used_ports_set
                if missing_ports:
                   # Declared ports must be connected.
                   errors.append(f"Component '{instance_id}' (type '{comp_type_str}') is missing required connections for declared ports: {sorted(list(missing_ports))}. Connected ports are: {sorted(list(used_ports_set))}.")

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
            except (ComponentError, ValueError, TypeError) as e: # Added TypeError
                # Catch errors during component's __init__
                errors.append(f"Instantiation failed for component '{instance_id}': {e}")
                continue # Skip this component

        if errors:
            error_msg = f"Circuit build failed for '{parsed_data.name}' with {len(errors)} errors:\n- " + "\n- ".join(errors)
            logger.error(error_msg)
            raise CircuitBuildError(error_msg)
        
        # Add the processed components to the *new* simulation circuit object
        sim_circuit.sim_components = processed_components # Directly assign the dict

        logger.info(f"Successfully built circuit '{sim_circuit.name}'. Created {len(processed_components)} simulation-ready components.")

        return sim_circuit 

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

        # Check for unexpected parameters provided in the netlist (Error not Warning)**
        for raw_param_name in raw_params:
            if raw_param_name not in declared_params:
                 # Raise an error for undeclared parameters to enforce strict interface adherence
                 raise ParameterError(f"Component '{instance_id}' (type '{comp_data.component_type}') was provided parameter '{raw_param_name}' which is not declared by the component type. Declared parameters are: {list(declared_params.keys())}.")

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
                        extra_msg=f"for parameter '{param_name}' in component '{instance_id}'" # Simplified extra_msg
                    )

                processed[param_name] = resolved_quantity
                logger.debug(f"Validated parameter '{param_name}' for '{instance_id}'. Value: {resolved_quantity:~P}, Expected Dim: '{expected_dim_str}'")

            except pint.DimensionalityError as e:
                # Construct a more informative message
                err_msg = (
                    f"Dimensionality mismatch {e.extra_msg}. " # Use the msg field we set
                    f"Got value {resolved_quantity:~P} (units '{e.units1}', dimensionality '{e.dim1}'), "
                    f"but expected dimension compatible with '{expected_dim_str}' (target units '{e.units2}', dimensionality '{e.dim2}')."
                )
                logger.error(err_msg)
                # Re-raise with the enhanced message if possible, or wrap it
                # Re-raising the original exception preserves the traceback
                e.extra_msg = err_msg # Modify the message in place
                raise e # Re-raise the modified exception

        return processed