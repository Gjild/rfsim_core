# --- src/rfsim_core/circuit_builder.py ---
import logging
from typing import Dict, Any, Optional, List, Tuple
import copy

from .data_structures import Circuit
from .data_structures import Component as ComponentData
from .components.base import ComponentBase, COMPONENT_REGISTRY, ComponentError
# ParameterManager is now built here, not just passed
from .parameters import ParameterManager, ParameterError, ParameterDefinition, ParameterDefinitionError, ParameterScopeError, CircularParameterDependencyError
from .units import ureg, pint, Quantity
logger = logging.getLogger(__name__)

class CircuitBuildError(ValueError):
    """Custom exception for errors during circuit building/processing."""
    pass

class CircuitBuilder:
    """
    Takes parsed circuit data, builds the ParameterManager by collecting and
    validating all parameter definitions, performs structural validation,
    and instantiates simulation-ready component objects.
    """
    def __init__(self):
        self._ureg = ureg # Use the shared registry
        logger.info("CircuitBuilder initialized.")

    def build_circuit(self, parsed_data: Circuit) -> Circuit:
        """
        Processes the parsed circuit data, collects and validates all parameter
        definitions, builds the ParameterManager, validates component structure,
        and populates the circuit with simulation-ready component instances.

        Args:
            parsed_data: The Circuit object returned by NetlistParser, containing
                         raw component data and raw global parameter definitions.

        Returns:
            A new Circuit object, populated with a built ParameterManager
            and a dictionary of simulation-ready component instances in the
            `sim_components` attribute.

        Raises:
            CircuitBuildError: If any errors occur during processing.
            ParameterError: If parameter definition or building fails.
        """
        logger.info(f"Building simulation-ready circuit '{parsed_data.name}'...")

        # --- Create a new Circuit object to avoid mutating the input ---
        try:
            sim_circuit = Circuit(
                name=parsed_data.name,
                nets=copy.deepcopy(parsed_data.nets),
                external_ports=copy.deepcopy(parsed_data.external_ports),
                external_port_impedances=copy.deepcopy(parsed_data.external_port_impedances),
                parameter_manager=None, # Will be set below
                ground_net_name=parsed_data.ground_net_name,
                # Store raw components data if needed later (e.g., port mapping in MNA)
                components=copy.deepcopy(parsed_data.components)
            )
            # Add dict for components for simulation
            setattr(sim_circuit, 'sim_components', {})
            # Retrieve raw global params stored by parser
            raw_global_params = getattr(parsed_data, 'raw_global_parameters', {})
            logger.debug(f"Created new Circuit object '{sim_circuit.name}' for simulation.")
        except Exception as e:
            logger.error(f"Failed to create base simulation circuit object: {e}", exc_info=True)
            raise CircuitBuildError(f"Failed to create base simulation circuit object: {e}") from e
        
        # --- Build Parameter Manager ---
        param_manager = ParameterManager()
        all_definitions: List[ParameterDefinition] = []
        param_errors = []
        try:
            all_definitions = self._collect_parameter_definitions(
                parsed_data.components,
                raw_global_params
            )
            param_manager.add_definitions(all_definitions)
            param_manager.build() # This performs context creation, dep checks, etc.
            sim_circuit.parameter_manager = param_manager # Store the built manager
            logger.info(f"ParameterManager built successfully for '{sim_circuit.name}'.")
            logger.debug(f"Parameter Manager State: {len(param_manager.get_all_internal_names())} params defined.")

        except (ParameterError, ParameterDefinitionError, ComponentError, CircularParameterDependencyError) as e:
            # Catch specific errors during definition collection or PM build
            error_msg = f"Circuit build failed for '{parsed_data.name}' due to parameter error: {e}"
            logger.error(error_msg, exc_info=False) # Log without full trace for config errors usually
            # Raise CircuitBuildError FROM the original error to preserve cause
            raise CircuitBuildError(error_msg) from e
        except Exception as e:
            # Catch unexpected errors
            error_msg = f"Unexpected error during parameter manager setup for '{parsed_data.name}': {e}"
            logger.error(error_msg, exc_info=True)
            raise CircuitBuildError(error_msg) from e

        if param_errors:
            error_msg = f"Circuit build failed for '{parsed_data.name}' due to parameter errors:\n- " + "\n- ".join(param_errors)
            logger.error(error_msg)
            raise CircuitBuildError(error_msg)

         # --- Instantiate Simulation Components & Validate Structure ---
        processed_components: Dict[str, ComponentBase] = {}
        structure_errors = []
        # --- Perform Build-Specific Validations and Instantiation ---
        for instance_id, comp_data in parsed_data.components.items(): # Iterate original raw data
            comp_type_str = comp_data.component_type
            logger.debug(f"Building component '{instance_id}' of type '{comp_type_str}'")

            # Component type existence already validated by parser
            ComponentClass = COMPONENT_REGISTRY[comp_type_str]

            # 1. Validate Ports Used vs Declared
            try:
                declared_ports_set = set(ComponentClass.declare_ports())
                used_ports_set = set(comp_data.ports.keys())
                extra_ports = used_ports_set - declared_ports_set
                if extra_ports:
                     structure_errors.append(f"Component '{instance_id}' (type '{comp_type_str}') uses undeclared ports: {sorted(list(extra_ports))}. Declared: {sorted(list(declared_ports_set))}.")
                missing_ports = declared_ports_set - used_ports_set
                if missing_ports:
                   structure_errors.append(f"Component '{instance_id}' (type '{comp_type_str}') is missing required connections for ports: {sorted(list(missing_ports))}.")
            except Exception as e:
                structure_errors.append(f"Error validating ports for component '{instance_id}': {e}")
                continue # Skip this component if port validation fails critically

            # 2. Identify Internal Parameter Names for this instance
            # (Parameter *values* are not resolved here)
            instance_internal_names = [
                f"{instance_id}.{param_name}" for param_name in ComponentClass.declare_parameters().keys()
            ]
            # Verify these names were actually created and are in the manager
            for internal_name in instance_internal_names:
                try:
                    param_manager.get_parameter_definition(internal_name) # Check existence
                except ParameterScopeError:
                     # This indicates a mismatch between declare_parameters and what was found/defined
                     structure_errors.append(f"Internal Error: Expected parameter '{internal_name}' for component '{instance_id}' was not found in the built ParameterManager. Check declaration vs. netlist definition.")

            # 3. Instantiate Simulation-Ready Component
            # Pass ParameterManager reference and the list of internal names
            if not structure_errors: # Only instantiate if structure looks ok so far
                try:
                    sim_component = ComponentClass(
                        instance_id=instance_id,
                        component_type=comp_type_str,
                        parameter_manager=param_manager, # Pass the manager reference
                        parameter_internal_names=instance_internal_names # Pass the names
                    )
                    processed_components[instance_id] = sim_component
                    logger.debug(f"Successfully created simulation component: {sim_component!r}")
                except (ComponentError, ValueError, TypeError) as e:
                    structure_errors.append(f"Instantiation failed for component '{instance_id}': {e}")
                    continue

        if structure_errors:
            error_msg = f"Circuit build failed for '{parsed_data.name}' with {len(structure_errors)} structure errors:\n- " + "\n- ".join(structure_errors)
            logger.error(error_msg)
            raise CircuitBuildError(error_msg)
        
        sim_circuit.sim_components = processed_components

        logger.info(f"Successfully built circuit '{sim_circuit.name}'. Created {len(processed_components)} simulation-ready components.")

        return sim_circuit 
    


    def _collect_parameter_definitions(
        self,
        raw_components: Dict[str, ComponentData],
        raw_global_params: Dict[str, Any]
    ) -> List[ParameterDefinition]:
        """
        Collects ParameterDefinition objects from global and instance parameters.
        Determines and validates the 'declared_dimension_str' for each.
        """
        definitions = []
        logger.debug("Collecting and validating parameter definitions...")

        # --- Process Global Parameters ---
        for name, value_or_dict in raw_global_params.items():
            expr_str: Optional[str] = None
            const_str: Optional[str] = None
            declared_dim_str: Optional[str] = None

            if isinstance(value_or_dict, dict): # Expression case
                expr_str = value_or_dict.get('expression')
                declared_dim_str = value_or_dict.get('dimension')
                if not expr_str:
                    raise ParameterDefinitionError(f"Global parameter '{name}' defined as dict but missing 'expression' key.")
                if not declared_dim_str:
                    raise ParameterDefinitionError(f"Global parameter '{name}' defined as expression ('{expr_str}') but missing mandatory 'dimension' key in YAML.")
                logger.debug(f"Global expr param: {name}, expr='{expr_str}', declared_dim='{declared_dim_str}'")

            else: # Constant case
                const_str = str(value_or_dict)
                # Infer dimension for constants
                try:
                    qty = self._ureg.Quantity(const_str)
                    # *** FIX: Store the string representation of the parsed units ***
                    if qty.dimensionless:
                        declared_dim_str = "dimensionless"
                    else:
                        # Use the units part of the quantity as the string
                        # e.g., for '10 nH', this will be 'nanohenry'
                        # e.g., for '50 ohm', this will be 'ohm'
                        declared_dim_str = str(qty.units)

                    # Ensure the result is a non-empty string (safety check)
                    if not declared_dim_str and declared_dim_str != "dimensionless":
                        raise ValueError("Inferred unit string is empty.")

                    logger.debug(f"Global const param: {name}, value='{const_str}', inferred_unit_str='{declared_dim_str}'") # Log change
                except (pint.UndefinedUnitError, pint.DimensionalityError, ValueError, TypeError) as e:
                    raise ParameterDefinitionError(f"Error parsing global constant parameter '{name}' value '{const_str}' to infer unit string: {e}") from e

            try:
                definitions.append(ParameterDefinition(
                    name=name,
                    scope='global',
                    owner_id=None,
                    expression_str=expr_str,
                    constant_value_str=const_str,
                    declared_dimension_str=declared_dim_str # Use the inferred *unit* string or YAML string
                ))
            except ValueError as e: # Catch validation errors from ParameterDefinition itself
                raise ParameterDefinitionError(f"Invalid definition for global parameter '{name}': {e}")


        # --- Process Instance Parameters ---
        defined_instance_params: Dict[Tuple[str, str], Any] = {} # (instance_id, param_name) -> raw_value_or_dict

        for instance_id, comp_data in raw_components.items():
            comp_type_str = comp_data.component_type
            ComponentClass = COMPONENT_REGISTRY[comp_type_str] # Type already checked by parser
            declared_params_spec = ComponentClass.declare_parameters() # Dict[param_name, expected_dim_str]

            # 1. Check for unexpected parameters provided in the instance definition
            raw_instance_params = comp_data.parameters # Raw YAML data for this instance
            for provided_param_name in raw_instance_params:
                if provided_param_name not in declared_params_spec:
                    raise ParameterDefinitionError(f"Component '{instance_id}' (type '{comp_type_str}') provided parameter '{provided_param_name}' which is not declared by the component type. Declared: {list(declared_params_spec.keys())}.")
                # Store the raw value/dict provided for this instance param
                defined_instance_params[(instance_id, provided_param_name)] = raw_instance_params[provided_param_name]


            # 2. Create definitions for *all* declared parameters for this instance
            for param_name, expected_dim_str in declared_params_spec.items():
                raw_value_or_dict = raw_instance_params.get(param_name)

                if raw_value_or_dict is None:
                    # Parameter declared by component type but not provided in instance definition
                    raise ParameterDefinitionError(f"Required parameter '{param_name}' missing for component instance '{instance_id}' (type '{comp_type_str}').")

                expr_str: Optional[str] = None
                const_str: Optional[str] = None
                # Dimension comes from component declaration, NOT from YAML here
                declared_dim_str = expected_dim_str

                if isinstance(raw_value_or_dict, dict): # Expression specified for instance param
                    expr_str = raw_value_or_dict.get('expression')
                    yaml_dim = raw_value_or_dict.get('dimension') # Check if user mistakenly added dimension
                    if not expr_str:
                         raise ParameterDefinitionError(f"Instance parameter '{instance_id}.{param_name}' defined as dict but missing 'expression' key.")
                    if yaml_dim is not None:
                         logger.warning(f"Instance parameter '{instance_id}.{param_name}' provided a 'dimension' key ('{yaml_dim}') in YAML. This is ignored; dimension '{declared_dim_str}' declared by component type '{comp_type_str}' is used.")
                    logger.debug(f"Instance expr param: {instance_id}.{param_name}, expr='{expr_str}', declared_dim='{declared_dim_str}'")

                else: # Constant or reference string specified for instance param
                    const_str = str(raw_value_or_dict)
                    # We don't infer dimension here, we use the one declared by the component.
                    # We also don't parse the constant yet; ParameterManager handles that.
                    logger.debug(f"Instance const/ref param: {instance_id}.{param_name}, value='{const_str}', declared_dim='{declared_dim_str}'")

                try:
                    definitions.append(ParameterDefinition(
                        name=param_name,
                        scope='instance',
                        owner_id=instance_id,
                        expression_str=expr_str,
                        constant_value_str=const_str,
                        declared_dimension_str=declared_dim_str # Dimension from ComponentClass.declare_parameters()
                    ))
                except ValueError as e: # Catch validation errors from ParameterDefinition itself
                    raise ParameterDefinitionError(f"Invalid definition for instance parameter '{instance_id}.{param_name}': {e}")


        logger.info(f"Collected {len(definitions)} parameter definitions.")
        return definitions