# --- src/rfsim_core/circuit_builder.py ---
import logging
from typing import Dict, Any, Optional, List, Tuple
import copy

from .data_structures import Circuit
from .data_structures import Component as ComponentData
from .components.base import ComponentBase, COMPONENT_REGISTRY, ComponentError
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
    User-facing semantic error reporting for port/parameter mismatches is
    deferred to SemanticValidator.
    """
    def __init__(self):
        self._ureg = ureg 
        logger.info("CircuitBuilder initialized.")

    def build_circuit(self, parsed_data: Circuit) -> Circuit:
        logger.info(f"Building simulation-ready circuit '{parsed_data.name}'...")

        try:
            sim_circuit = Circuit(
                name=parsed_data.name,
                nets=copy.deepcopy(parsed_data.nets),
                external_ports=copy.deepcopy(parsed_data.external_ports),
                external_port_impedances=copy.deepcopy(parsed_data.external_port_impedances),
                parameter_manager=None, 
                ground_net_name=parsed_data.ground_net_name,
                components=copy.deepcopy(parsed_data.components) # Store raw component data from parser
            )
            setattr(sim_circuit, 'sim_components', {}) # Initialize dict for simulation-ready components
            raw_global_params = getattr(parsed_data, 'raw_global_parameters', {})
            logger.debug(f"Created new Circuit object '{sim_circuit.name}' for simulation.")
        except Exception as e:
            logger.error(f"Failed to create base simulation circuit object: {e}", exc_info=True)
            raise CircuitBuildError(f"Failed to create base simulation circuit object: {e}") from e
        
        param_manager = ParameterManager()
        all_definitions: List[ParameterDefinition] = []
        
        try:
            all_definitions = self._collect_parameter_definitions(
                parsed_data.components, # Use raw components from parsed_data
                raw_global_params
            )
            param_manager.add_definitions(all_definitions)
            param_manager.build() 
            sim_circuit.parameter_manager = param_manager 
            logger.info(f"ParameterManager built successfully for '{sim_circuit.name}'.")
            logger.debug(f"Parameter Manager State: {len(param_manager.get_all_internal_names())} params defined.")

        except (ParameterError, ParameterDefinitionError, ComponentError, CircularParameterDependencyError) as e:
            error_msg = f"Circuit build failed for '{parsed_data.name}' due to parameter error: {e}"
            logger.error(error_msg, exc_info=False) 
            raise CircuitBuildError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error during parameter manager setup for '{parsed_data.name}': {e}"
            logger.error(error_msg, exc_info=True)
            raise CircuitBuildError(error_msg) from e

        processed_components: Dict[str, ComponentBase] = {}
        internal_structure_errors: List[str] = [] 

        # Iterate through raw component data stored on sim_circuit (copied from parsed_data)
        for instance_id, comp_data in sim_circuit.components.items(): 
            comp_type_str = comp_data.component_type
            logger.debug(f"Building component '{instance_id}' of type '{comp_type_str}'")

            if comp_type_str not in COMPONENT_REGISTRY:
                logger.warning(f"Component '{instance_id}' uses unknown type '{comp_type_str}'. "
                               f"No simulation object will be created for it. SemanticValidator should report this.")
                # This is a semantic issue, not a CircuitBuildError for internal structure.
                # SemanticValidator will pick this up based on raw comp_data.
                continue # Skip creating a sim_component for this instance_id

            ComponentClass = COMPONENT_REGISTRY[comp_type_str]

            instance_internal_names = []
            try:
                declared_param_keys = ComponentClass.declare_parameters().keys()
            except Exception as e:
                internal_structure_errors.append(f"Critical error fetching parameter declarations for component type '{comp_type_str}' (instance '{instance_id}'): {e}. Cannot proceed with this component.")
                continue 
            
            for param_name in declared_param_keys:
                internal_name = f"{instance_id}.{param_name}"
                try:
                    param_manager.get_parameter_definition(internal_name) 
                    instance_internal_names.append(internal_name)
                except ParameterScopeError:
                    logger.warning(f"Internal Note: Declared parameter '{param_name}' for component '{instance_id}' "
                                   f"does not have a corresponding definition in ParameterManager ('{internal_name}'). "
                                   f"This might be due to it being an undeclared parameter in YAML or a missing required one. "
                                   f"SemanticValidator will report. Skipping this parameter for component instantiation.")
            
            try:
                sim_component = ComponentClass(
                    instance_id=instance_id,
                    component_type=comp_type_str,
                    parameter_manager=param_manager, 
                    parameter_internal_names=instance_internal_names 
                )
                processed_components[instance_id] = sim_component
                logger.debug(f"Successfully created simulation component: {sim_component!r}")
            except (ComponentError, ValueError, TypeError) as e:
                logger.error(f"Instantiation failed for component '{instance_id}': {e}. SemanticValidator may report further details.")
                continue
        
        if internal_structure_errors: 
            error_msg = f"Circuit build failed for '{parsed_data.name}' with {len(internal_structure_errors)} critical internal errors:\n- " + "\n- ".join(internal_structure_errors)
            logger.error(error_msg)
            raise CircuitBuildError(error_msg)
        
        sim_circuit.sim_components = processed_components # Assign the dict of created sim_components
        logger.info(f"Successfully built circuit '{sim_circuit.name}'. Created {len(processed_components)} simulation-ready components.")
        return sim_circuit 

    def _collect_parameter_definitions(
        self,
        raw_components: Dict[str, ComponentData], # Takes raw component data
        raw_global_params: Dict[str, Any]
    ) -> List[ParameterDefinition]:
        definitions = []
        logger.debug("Collecting and validating parameter definitions...")

        for name, value_or_dict in raw_global_params.items():
            expr_str: Optional[str] = None
            const_str: Optional[str] = None
            declared_dim_str: Optional[str] = None

            if isinstance(value_or_dict, dict): 
                expr_str = value_or_dict.get('expression')
                declared_dim_str = value_or_dict.get('dimension')
                if not expr_str:
                    raise ParameterDefinitionError(f"Global parameter '{name}' defined as dict but missing 'expression' key.")
                if not declared_dim_str:
                    raise ParameterDefinitionError(f"Global parameter '{name}' defined as expression ('{expr_str}') but missing mandatory 'dimension' key in YAML.")
                logger.debug(f"Global expr param: {name}, expr='{expr_str}', declared_dim='{declared_dim_str}'")
            else: 
                const_str = str(value_or_dict)
                try:
                    qty = self._ureg.Quantity(const_str)
                    if qty.dimensionless:
                        declared_dim_str = "dimensionless"
                    else:
                        declared_dim_str = str(qty.units) 
                    if not declared_dim_str and declared_dim_str != "dimensionless": # Ensure not empty string if not dimensionless
                        # Attempt to re-parse units if initial str(qty.units) was empty (e.g. for pure numbers becoming dimensionless)
                        # This path might not be strictly necessary if Pint handles it well, but defensive.
                        pu = self._ureg.parse_units(str(qty.units))
                        if pu == self._ureg.dimensionless:
                            declared_dim_str = "dimensionless"
                        else:
                            declared_dim_str = str(pu) # Use Pint's canonical form
                        if not declared_dim_str and declared_dim_str != "dimensionless":
                             raise ValueError("Inferred unit string is empty or invalid after parsing.")

                    logger.debug(f"Global const param: {name}, value='{const_str}', inferred_unit_str='{declared_dim_str}'")
                except (pint.UndefinedUnitError, pint.DimensionalityError, ValueError, TypeError) as e:
                    raise ParameterDefinitionError(f"Error parsing global constant parameter '{name}' value '{const_str}' to infer unit string: {e}") from e
            try:
                definitions.append(ParameterDefinition(
                    name=name, scope='global', owner_id=None,
                    expression_str=expr_str, constant_value_str=const_str,
                    declared_dimension_str=declared_dim_str, is_value_provided=True
                ))
            except ValueError as e: 
                raise ParameterDefinitionError(f"Invalid definition for global parameter '{name}': {e}")

        for instance_id, comp_data in raw_components.items():
            comp_type_str = comp_data.component_type
            
            # If component type is unknown, we can't get its declared parameters.
            # SemanticValidator will report the unknown type. CircuitBuilder will skip sim_comp creation.
            # Here, we should skip collecting param defs for this component to avoid crashing.
            if comp_type_str not in COMPONENT_REGISTRY:
                logger.debug(f"Skipping parameter definition collection for component '{instance_id}' due to unknown type '{comp_type_str}'.")
                continue
                
            ComponentClass = COMPONENT_REGISTRY[comp_type_str] 
            
            try:
                declared_params_spec = ComponentClass.declare_parameters()
            except Exception as e:
                 raise ComponentError(f"Failed to get parameter declarations from component type '{comp_type_str}' (instance '{instance_id}'): {e}") from e

            raw_instance_params = comp_data.parameters 

            for provided_param_name in raw_instance_params:
                if provided_param_name not in declared_params_spec:
                    logger.warning(f"Component '{instance_id}' (type '{comp_type_str}') provided parameter '{provided_param_name}' "
                                   f"which is not declared by the component type. Declared: {list(declared_params_spec.keys())}. "
                                   f"This parameter will be IGNORED by CircuitBuilder and reported by SemanticValidator.")
            
            for param_name, expected_dim_str in declared_params_spec.items():
                raw_value_or_dict = raw_instance_params.get(param_name)
                
                is_value_provided_for_def = True
                expr_str_for_def: Optional[str] = None
                const_str_for_def: Optional[str] = None
                
                if raw_value_or_dict is None:
                    logger.warning(f"Required parameter '{param_name}' for component instance '{instance_id}' (type '{comp_type_str}') "
                                   f"is MISSING from YAML. A ParameterDefinition will be created without a value, "
                                   f"and SemanticValidator will report this as an error.")
                    is_value_provided_for_def = False
                elif isinstance(raw_value_or_dict, dict): 
                    expr_str_for_def = raw_value_or_dict.get('expression')
                    yaml_dim = raw_value_or_dict.get('dimension') 
                    if not expr_str_for_def:
                         raise ParameterDefinitionError(f"Instance parameter '{instance_id}.{param_name}' defined as dict but missing 'expression' key.")
                    if yaml_dim is not None:
                         logger.warning(f"Instance parameter '{instance_id}.{param_name}' provided a 'dimension' key ('{yaml_dim}') in YAML. "
                                        f"This is ignored; dimension '{expected_dim_str}' declared by component type '{comp_type_str}' is used.")
                    logger.debug(f"Instance expr param: {instance_id}.{param_name}, expr='{expr_str_for_def}', declared_dim='{expected_dim_str}'")
                else: 
                    const_str_for_def = str(raw_value_or_dict)
                    logger.debug(f"Instance const/ref param: {instance_id}.{param_name}, value='{const_str_for_def}', declared_dim='{expected_dim_str}'")

                try:
                    definitions.append(ParameterDefinition(
                        name=param_name, scope='instance', owner_id=instance_id,
                        expression_str=expr_str_for_def, 
                        constant_value_str=const_str_for_def,
                        declared_dimension_str=expected_dim_str, 
                        is_value_provided=is_value_provided_for_def 
                    ))
                except ValueError as e: 
                    raise ParameterDefinitionError(f"Invalid definition for instance parameter '{instance_id}.{param_name}': {e}")
        
        logger.info(f"Collected {len(definitions)} parameter definitions.")
        return definitions