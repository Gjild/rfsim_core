# --- src/rfsim_core/validation/semantic_validator.py ---
import logging
from typing import List, Dict, Any, Optional, Tuple 
import numpy as np 

from ..data_structures import Circuit
from ..data_structures import Component as ComponentData 
from ..units import ureg, pint, Quantity 
from .issues import ValidationIssue, ValidationIssueLevel
from .issue_codes import SemanticIssueCode

from ..components import COMPONENT_REGISTRY
from ..components.elements import Resistor, Capacitor, Inductor
from ..components.base import ComponentBase 
from ..parameters import ParameterManager, ParameterError, ParameterScopeError 

logger = logging.getLogger(__name__)

class SemanticValidationError(ValueError):
    """
    Custom exception raised when semantic validation detects one or more errors.
    Contains a list of all error-level ValidationIssue objects.
    """
    def __init__(self, issues: List[ValidationIssue], message: Optional[str] = None):
        self.issues: List[ValidationIssue] = [
            issue for issue in issues if issue.level == ValidationIssueLevel.ERROR
        ]
        # Removed the 'if not self.issues:' block as per review (dead code)

        if message is None:
            error_messages_list = []
            for err in self.issues: # self.issues now guaranteed to contain only errors
                error_messages_list.append(
                    f"  - [{err.code}]: {err.message} (Component: {err.component_id or 'N/A'}, "
                    f"Net: {err.net_name or 'N/A'}, Param: {err.parameter_name or 'N/A'}, Details: {err.details or ''})"
                )
            # Ensure there are actual errors before claiming them in the message
            if not self.issues: # Should not happen if run_sweep calls this correctly
                 summary_message = "SemanticValidationError raised unexpectedly with no error-level issues."
            else:
                 summary_message = (f"Semantic validation failed with {len(self.issues)} error(s):\n" +
                                   "\n".join(error_messages_list))
            super().__init__(summary_message, *[])
        else:
            super().__init__(message, *[])


class SemanticValidator:
    """
    Performs semantic validation on a simulation-ready circuit object.
    This runs after initial parsing (NetlistParser) and circuit construction (CircuitBuilder).
    """

    def __init__(self, circuit: Circuit):
        if not isinstance(circuit, Circuit):
            raise TypeError("SemanticValidator requires a valid Circuit object.")
        if not hasattr(circuit, 'sim_components'):
            raise ValueError("Input Circuit object to SemanticValidator is not simulation-ready (missing 'sim_components' attribute). Ensure CircuitBuilder.build_circuit() was called.")
        if not isinstance(circuit.parameter_manager, ParameterManager):
             raise ValueError("Circuit object provided to SemanticValidator has an uninitialized or invalid ParameterManager.")


        self.circuit: Circuit = circuit
        self.issues: List[ValidationIssue] = []
        self._ureg = ureg

        self._net_connection_counts: Dict[str, int] = self._compute_net_connection_counts()
        logger.debug(f"SemanticValidator initialized for circuit '{self.circuit.name}'. Net connection counts: {self._net_connection_counts}")

    def _compute_net_connection_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for comp_instance_id, comp_data_struct in self.circuit.components.items():
            for port_id, port_obj in comp_data_struct.ports.items():
                if port_obj.net: 
                    net_name = port_obj.net.name
                    counts[net_name] = counts.get(net_name, 0) + 1
                else:
                    logger.debug(f"Port '{port_id}' of component '{comp_instance_id}' has no associated net object during connection count. Will be reported by PORT_UNLINKED_001.")
        return counts

    def validate(self) -> List[ValidationIssue]:
        self.issues = []
        logger.info(f"Running semantic validation for circuit '{self.circuit.name}'...")

        self._check_id_uniqueness()
        self._check_registered_component_types()
        self._check_port_definitions()
        self._check_component_parameter_declarations() 
        self._check_refined_net_connectivity()
        self._check_external_port_validity()
        self._check_ground_net_validity()
        self._check_component_parameter_dimensionality() 
        self._check_ideal_dc_path_identification_preliminary()
        
        if not self.issues:
            logger.info(f"Semantic validation for circuit '{self.circuit.name}' completed with no issues found.")
        else:
            num_errors = sum(1 for issue in self.issues if issue.level == ValidationIssueLevel.ERROR)
            num_warnings = sum(1 for issue in self.issues if issue.level == ValidationIssueLevel.WARNING)
            num_infos = sum(1 for issue in self.issues if issue.level == ValidationIssueLevel.INFO)
            logger.info(
                f"Semantic validation for circuit '{self.circuit.name}' completed. "
                f"Found: {num_errors} errors, {num_warnings} warnings, {num_infos} info messages."
            )
            
        return self.issues

    def _add_issue(self, level: ValidationIssueLevel, code_enum: SemanticIssueCode,
                   component_id: Optional[str] = None, net_name: Optional[str] = None,
                   parameter_name: Optional[str] = None, details: Optional[Dict[str, Any]] = None,
                   **kwargs_for_format):
        """
        Helper to create and add a ValidationIssue.
        Explicit component_id, net_name, parameter_name are for the main fields of ValidationIssue
        and are also made available for message formatting and automatically added to the details dict.
        **kwargs_for_format provides values for message template placeholders and for the details dict.
        The 'details' dict argument can provide additional or overriding values for the details dict.
        """
        
        # --- Prepare arguments for message formatting ---
        msg_fmt_args = dict(kwargs_for_format) 
        # Add explicit main identifiers if they are not already in msg_fmt_args
        # (allowing kwargs_for_format to override if specific formatting needed for the message itself)
        if component_id is not None and 'component_id' not in msg_fmt_args:
            msg_fmt_args['component_id'] = component_id
        if net_name is not None and 'net_name' not in msg_fmt_args:
            msg_fmt_args['net_name'] = net_name
        if parameter_name is not None and 'parameter_name' not in msg_fmt_args:
            msg_fmt_args['parameter_name'] = parameter_name

        message = code_enum.format_message(**msg_fmt_args)

        # --- Prepare details for the ValidationIssue object ---
        # Start with kwargs_for_format
        current_issue_details_wip: Dict[str, Any] = {}
        for k, v_raw in kwargs_for_format.items():
            # Sanitization logic for values from kwargs_for_format
            if isinstance(v_raw, (list, set)):
                try: current_issue_details_wip[k] = sorted([str(i) for i in v_raw])
                except TypeError: current_issue_details_wip[k] = str(v_raw) 
            elif not isinstance(v_raw, (str, int, float, bool, complex, type(None))):
                current_issue_details_wip[k] = str(v_raw) 
            else:
                current_issue_details_wip[k] = v_raw 
        
        # Add explicit main identifiers to details_wip if they are not already present from kwargs_for_format.
        # This ensures they are available in ValidationIssue.details.
        # We add them BEFORE merging the 'details' argument, so 'details' can override if necessary.
        if component_id is not None and 'component_id' not in current_issue_details_wip:
            current_issue_details_wip['component_id'] = component_id
        if net_name is not None and 'net_name' not in current_issue_details_wip:
            current_issue_details_wip['net_name'] = net_name
        if parameter_name is not None and 'parameter_name' not in current_issue_details_wip:
            current_issue_details_wip['parameter_name'] = parameter_name

        # Then merge/override with the explicit 'details' dictionary argument
        if details: # The 'details' argument passed to _add_issue
            for k_detail, v_detail_raw in details.items():
                # Sanitization logic for values from the 'details' dict
                if isinstance(v_detail_raw, (list, set)):
                    try: current_issue_details_wip[k_detail] = sorted([str(i) for i in v_detail_raw])
                    except TypeError: current_issue_details_wip[k_detail] = str(v_detail_raw)
                elif not isinstance(v_detail_raw, (str, int, float, bool, complex, type(None))):
                     current_issue_details_wip[k_detail] = str(v_detail_raw)
                else:
                    current_issue_details_wip[k_detail] = v_detail_raw
        
        # Final details object: filter out any keys that ended up with a None value
        # (e.g., if component_id was None, it shouldn't appear as details['component_id']=None)
        final_details_for_issue_obj = {
            k: v for k, v in current_issue_details_wip.items() if v is not None
        }
        
        issue = ValidationIssue(
            level=level,
            code=code_enum.code,
            message=message,
            component_id=component_id,         
            net_name=net_name,                 
            parameter_name=parameter_name,     
            details=final_details_for_issue_obj if final_details_for_issue_obj else None
        )
        self.issues.append(issue)

    def _check_id_uniqueness(self):
        logger.debug("Executing _check_id_uniqueness...")
        for comp_id_key in self.circuit.sim_components.keys(): # Corrected: Iterate circuit.components for raw data IDs
            if not isinstance(comp_id_key, str) or not comp_id_key:
                self._add_issue(
                    level=ValidationIssueLevel.ERROR,
                    code_enum=SemanticIssueCode.ID_COMP_INVALID_001,
                    component_id=str(comp_id_key)
                    # **{'component_id': str(comp_id_key)} # Removed duplicate
                )
        
        for net_name_key in self.circuit.nets.keys():
            if not isinstance(net_name_key, str) or not net_name_key:
                self._add_issue(
                    level=ValidationIssueLevel.ERROR,
                    code_enum=SemanticIssueCode.ID_NET_INVALID_001,
                    net_name=str(net_name_key)
                    # **{'net_name': str(net_name_key)} # Removed duplicate
                )

    def _check_registered_component_types(self):
        logger.debug("Executing _check_registered_component_types...")
        for comp_id, comp_data in self.circuit.components.items():
            comp_type_str = comp_data.component_type
            if comp_type_str not in COMPONENT_REGISTRY:
                self._add_issue(
                    level=ValidationIssueLevel.ERROR,
                    code_enum=SemanticIssueCode.COMP_TYPE_001,
                    component_id=comp_id,
                    # **kwargs for message formatting and details (component_id removed)
                    unregistered_type=comp_type_str,
                    available_types=list(COMPONENT_REGISTRY.keys())
                )

    def _check_port_definitions(self):
        logger.debug("Executing _check_port_definitions...")
        for comp_id, sim_comp_instance in self.circuit.sim_components.items():
            raw_comp_data = self.circuit.components.get(comp_id)
            if not raw_comp_data:
                logger.error(f"Internal Inconsistency: Sim component '{comp_id}' exists, but raw data missing. Skipping port checks.")
                continue

            ComponentClass = type(sim_comp_instance)
            comp_type_str_for_msg = ComponentClass.component_type_str

            try:
                declared_ports_by_type = set(ComponentClass.declare_ports())
            except Exception as e:
                logger.error(f"Error calling declare_ports() for type '{comp_type_str_for_msg}' (comp '{comp_id}'): {e}")
                self._add_issue(
                    level=ValidationIssueLevel.ERROR, 
                    code_enum=SemanticIssueCode.COMP_PORT_DECL_FAIL_001, 
                    component_id=comp_id,
                    # **kwargs (component_id removed)
                    component_type=comp_type_str_for_msg,
                    error_details=str(e)
                )
                continue

            used_ports_in_instance = set(raw_comp_data.ports.keys())

            extra_ports = used_ports_in_instance - declared_ports_by_type
            if extra_ports:
                self._add_issue(
                    level=ValidationIssueLevel.ERROR, 
                    code_enum=SemanticIssueCode.PORT_DEF_001,
                    component_id=comp_id,
                    # **kwargs (component_id removed)
                    component_type=comp_type_str_for_msg,
                    extra_ports=list(extra_ports),
                    declared_ports=list(declared_ports_by_type)
                )

            missing_ports = declared_ports_by_type - used_ports_in_instance
            if missing_ports:
                self._add_issue(
                    level=ValidationIssueLevel.ERROR, 
                    code_enum=SemanticIssueCode.PORT_DEF_002,
                    component_id=comp_id,
                    # **kwargs (component_id removed)
                    component_type=comp_type_str_for_msg,
                    missing_ports=list(missing_ports),
                    declared_ports=list(declared_ports_by_type)
                )

            for port_yaml_id, port_ds_obj in raw_comp_data.ports.items(): 
                if port_ds_obj.net is None:
                    original_name = port_ds_obj.original_yaml_net_name or "<Original YAML net name not available>"
                    self._add_issue(
                        level=ValidationIssueLevel.ERROR, 
                        code_enum=SemanticIssueCode.PORT_UNLINKED_001,
                        component_id=comp_id,
                        # **kwargs (component_id removed)
                        port_id=str(port_yaml_id),
                        original_net_name_from_yaml=original_name
                    )
                elif port_ds_obj.net.name not in self.circuit.nets:
                     self._add_issue(
                        level=ValidationIssueLevel.ERROR, 
                        code_enum=SemanticIssueCode.PORT_CONN_002,
                        component_id=comp_id,
                        net_name=port_ds_obj.net.name,
                        # **kwargs (component_id, net_name removed)
                        port_id=str(port_yaml_id)
                    )

    def _check_component_parameter_declarations(self):
        logger.debug("Executing _check_component_parameter_declarations...")
        pm = self.circuit.parameter_manager

        for comp_id, raw_comp_data in self.circuit.components.items():
            sim_comp_instance = self.circuit.sim_components.get(comp_id)
            if not sim_comp_instance:
                logger.debug(f"Sim component instance for '{comp_id}' not found (likely unknown type), skipping param declaration checks.")
                continue

            ComponentClass = type(sim_comp_instance)
            comp_type_str_for_msg = ComponentClass.component_type_str
            
            try:
                declared_param_names_by_type = set(ComponentClass.declare_parameters().keys())
            except Exception as e:
                logger.error(f"Error calling declare_parameters() for type '{comp_type_str_for_msg}' (comp '{comp_id}'): {e}")
                continue

            provided_params_in_yaml = set(raw_comp_data.parameters.keys())

            undeclared_yaml_params = provided_params_in_yaml - declared_param_names_by_type
            for undeclared_param_name in undeclared_yaml_params:
                self._add_issue(
                    level=ValidationIssueLevel.ERROR, 
                    code_enum=SemanticIssueCode.PARAM_INST_UNDCL_001,
                    component_id=comp_id,
                    parameter_name=undeclared_param_name,
                    # **kwargs (component_id, parameter_name removed)
                    component_type=comp_type_str_for_msg,
                    declared_params=list(declared_param_names_by_type)
                )

            for declared_base_name in declared_param_names_by_type:
                internal_name = f"{comp_id}.{declared_base_name}"
                try:
                    param_def = pm.get_parameter_definition(internal_name)
                    if not param_def.is_value_provided: 
                        self._add_issue(
                            level=ValidationIssueLevel.ERROR, 
                            code_enum=SemanticIssueCode.PARAM_INST_MISSING_001,
                            component_id=comp_id,
                            parameter_name=declared_base_name, 
                            # **kwargs (component_id, parameter_name removed)
                            component_type=comp_type_str_for_msg,
                            declared_params=list(declared_param_names_by_type)
                        )
                except ParameterScopeError:
                    logger.error(f"Internal Inconsistency: Declared parameter '{declared_base_name}' (internal: '{internal_name}') for component '{comp_id}' "
                                 f"has no ParameterDefinition in ParameterManager. CircuitBuilder should have created one, possibly marked as 'value not provided'.")
                    self._add_issue(
                        level=ValidationIssueLevel.ERROR, 
                        code_enum=SemanticIssueCode.PARAM_INST_MISSING_001,
                        component_id=comp_id,
                        parameter_name=declared_base_name,
                        details={"internal_consistency_note": "ParameterDefinition missing in ParameterManager."},
                        # **kwargs (component_id, parameter_name removed)
                        component_type=comp_type_str_for_msg,
                        declared_params=list(declared_param_names_by_type)
                    )

    def _check_refined_net_connectivity(self):
        logger.debug("Executing _check_refined_net_connectivity...")
        all_net_names = set(self.circuit.nets.keys())
        external_net_names = set(self.circuit.external_ports.keys())
        ground_net_name = self.circuit.ground_net_name
        
        internal_net_names = all_net_names - external_net_names - {ground_net_name}

        for net_name in internal_net_names:
            count = self._net_connection_counts.get(net_name, 0)
            if count == 0:
                self._add_issue(
                    level=ValidationIssueLevel.WARNING, 
                    code_enum=SemanticIssueCode.NET_CONN_001,
                    net_name=net_name
                    # **{'net_name': net_name} # Removed duplicate
                )
            elif count == 1:
                found_comp_id_str = "unknown_component"
                found_port_id_str = "unknown_port"
                component_found = False
                for comp_data_instance in self.circuit.components.values():
                    for port_yaml_id, port_ds_obj in comp_data_instance.ports.items():
                        if port_ds_obj.net and port_ds_obj.net.name == net_name:
                            found_comp_id_str = comp_data_instance.instance_id
                            found_port_id_str = str(port_yaml_id)
                            component_found = True
                            break
                    if component_found:
                        break
                
                self._add_issue(
                    level=ValidationIssueLevel.WARNING, 
                    code_enum=SemanticIssueCode.NET_CONN_002,
                    net_name=net_name, # Explicit primary subject
                    component_id=found_comp_id_str if component_found else None, # Explicit component this net connects to
                    # **kwargs for formatting message and for details
                    # 'net_name' will be picked up by _add_issue for formatting from explicit arg.
                    # 'component_id' (for {component_id} if used by template) also picked up.
                    # Template uses {connected_to_component} and {connected_to_port}.
                    connected_to_component=found_comp_id_str,
                    connected_to_port=found_port_id_str
                )

    def _check_external_port_validity(self):
        logger.debug("Executing _check_external_port_validity...")
        pm = self.circuit.parameter_manager

        for port_name in self.circuit.external_ports.keys(): 
            if port_name == self.circuit.ground_net_name:
                self._add_issue(
                    level=ValidationIssueLevel.ERROR, 
                    code_enum=SemanticIssueCode.EXT_PORT_001,
                    net_name=port_name
                    # **{'net_name': port_name} # Removed duplicate
                )
            if self._net_connection_counts.get(port_name, 0) == 0:
                self._add_issue(
                    level=ValidationIssueLevel.ERROR, 
                    code_enum=SemanticIssueCode.EXT_PORT_002,
                    net_name=port_name
                    # **{'net_name': port_name} # Removed duplicate
                )

        for port_name, z0_str_val in self.circuit.external_port_impedances.items():
            if not z0_str_val: 
                self._add_issue(
                    level=ValidationIssueLevel.ERROR, 
                    code_enum=SemanticIssueCode.EXT_PORT_003,
                    net_name=port_name
                    # **{'net_name': port_name} # Removed duplicate
                )
                continue 

            try:
                qty = self._ureg.Quantity(z0_str_val)
                if not qty.is_compatible_with("ohm"):
                    self._add_issue(
                        level=ValidationIssueLevel.ERROR, 
                        code_enum=SemanticIssueCode.EXT_PORT_DIM_001,
                        net_name=port_name,
                        # **kwargs (net_name removed)
                        value=z0_str_val,
                        parsed_dimensionality=str(qty.dimensionality)
                    )
            except (pint.UndefinedUnitError, pint.DimensionalityError, TypeError, ValueError):
                param_ref_name_stripped = z0_str_val.strip()
                global_param_internal_name = f"{pm.GLOBAL_SCOPE_PREFIX}.{param_ref_name_stripped}"
                
                try:
                    param_def = pm.get_parameter_definition(global_param_internal_name)
                    declared_dim_of_global_param = param_def.declared_dimension_str
                    
                    if not self._ureg.is_compatible_with(declared_dim_of_global_param, "ohm"):
                        self._add_issue(
                            level=ValidationIssueLevel.ERROR, 
                            code_enum=SemanticIssueCode.EXT_PORT_REF_DIM_001,
                            net_name=port_name,
                            parameter_name=param_ref_name_stripped,
                            details={'referenced_param_internal_name': global_param_internal_name},
                            # **kwargs (net_name, parameter_name removed)
                            declared_dimension_of_ref=declared_dim_of_global_param
                        )
                except ParameterScopeError: 
                    self._add_issue(
                        level=ValidationIssueLevel.ERROR, 
                        code_enum=SemanticIssueCode.EXT_PORT_REF_001,
                        net_name=port_name,
                        parameter_name=param_ref_name_stripped
                        # **kwargs (net_name, parameter_name removed)
                    )
                except Exception as e: 
                    logger.error(f"Unexpected error validating Z0 reference '{z0_str_val}' for port '{port_name}': {e}")
                    self._add_issue(
                        level=ValidationIssueLevel.ERROR, 
                        code_enum=SemanticIssueCode.EXT_PORT_REF_VALIDATE_FAIL_001, 
                        net_name=port_name,
                        parameter_name=param_ref_name_stripped,
                        # **kwargs (net_name, parameter_name removed)
                        error_details=str(e)
                    )

    def _check_ground_net_validity(self):
        logger.debug("Executing _check_ground_net_validity...")
        ground_name = self.circuit.ground_net_name
        if self.circuit.components: 
            if self._net_connection_counts.get(ground_name, 0) == 0:
                self._add_issue(
                    level=ValidationIssueLevel.WARNING, 
                    code_enum=SemanticIssueCode.GND_CONN_001,
                    net_name=ground_name
                    # **{'net_name': ground_name} # Removed duplicate
                )

    def _check_component_parameter_dimensionality(self):
        logger.debug("Executing _check_component_parameter_dimensionality...")
        pm = self.circuit.parameter_manager

        for comp_id, sim_comp_instance in self.circuit.sim_components.items():
            ComponentClass = type(sim_comp_instance)
            comp_type_str_for_msg = ComponentClass.component_type_str
            
            try:
                declared_param_specs_by_type = ComponentClass.declare_parameters()
            except Exception as e:
                logger.error(f"Error calling declare_parameters() for type '{comp_type_str_for_msg}' (comp '{comp_id}'): {e}")
                continue 

            for base_param_name, expected_dim_str_by_type in declared_param_specs_by_type.items():
                internal_name = f"{comp_id}.{base_param_name}"
                
                try:
                    param_def_from_pm = pm.get_parameter_definition(internal_name)
                except ParameterScopeError:
                    logger.debug(f"Parameter '{internal_name}' not found in PM for component '{comp_id}'. "
                                 f"Likely missing from YAML, _check_component_parameter_declarations should report.")
                    continue 
                
                actual_declared_dim_str_in_pm = param_def_from_pm.declared_dimension_str
                if not self._ureg.is_compatible_with(actual_declared_dim_str_in_pm, expected_dim_str_by_type):
                    self._add_issue(
                        level=ValidationIssueLevel.ERROR, 
                        code_enum=SemanticIssueCode.PARAM_DIM_001,
                        component_id=comp_id,
                        parameter_name=internal_name,
                        # **kwargs (component_id, parameter_name removed)
                        declared_in_pm=actual_declared_dim_str_in_pm,
                        expected_by_comp_type=expected_dim_str_by_type,
                        component_type=comp_type_str_for_msg
                    )

                if param_def_from_pm.is_value_provided and pm.is_constant(internal_name):
                    try:
                        const_val_qty = pm.get_constant_value(internal_name)
                        if not const_val_qty.is_compatible_with(expected_dim_str_by_type):
                            self._add_issue(
                                level=ValidationIssueLevel.ERROR, 
                                code_enum=SemanticIssueCode.PARAM_VAL_DIM_001,
                                component_id=comp_id,
                                parameter_name=internal_name,
                                # **kwargs (component_id, parameter_name removed)
                                resolved_value_str=f"{const_val_qty:~P}",
                                expected_dim_str=expected_dim_str_by_type,
                                resolved_value_dim=str(const_val_qty.dimensionality)
                            )
                    except ParameterError as e: 
                         self._add_issue(
                            level=ValidationIssueLevel.ERROR, 
                            code_enum=SemanticIssueCode.PARAM_CONST_VALIDATE_FAIL_001,
                            component_id=comp_id, 
                            parameter_name=internal_name,
                            # **kwargs (component_id, parameter_name removed)
                            expected_dim_str=expected_dim_str_by_type,
                            error_details=str(e) 
                        )
                    except Exception as e: 
                         logger.error(f"Unexpected error getting/checking constant value for '{internal_name}': {e}", exc_info=True)
                         self._add_issue(
                            level=ValidationIssueLevel.ERROR,
                            code_enum=SemanticIssueCode.PARAM_CONST_VALIDATE_FAIL_001, 
                            component_id=comp_id,
                            parameter_name=internal_name,
                            # **kwargs (component_id, parameter_name removed)
                            error_details=str(e),
                            expected_dim_str=expected_dim_str_by_type
                        )

    def _check_ideal_dc_path_identification_preliminary(self):
        logger.debug("Executing _check_ideal_dc_path_identification_preliminary...")
        pm = self.circuit.parameter_manager

        for comp_id, sim_comp_instance in self.circuit.sim_components.items():
            ComponentClass = type(sim_comp_instance)
            comp_type_str_for_msg = ComponentClass.component_type_str

            def get_const_param_val(param_base_name: str) -> Optional[Quantity]:
                internal_param_name = f"{comp_id}.{param_base_name}"
                try:
                    param_def = pm.get_parameter_definition(internal_param_name)
                    if param_def.is_value_provided and pm.is_constant(internal_param_name):
                        return pm.get_constant_value(internal_param_name)
                except (ParameterError, ParameterScopeError):
                    return None
                return None

            if ComponentClass == Resistor:
                val_qty = get_const_param_val("resistance")
                if val_qty is not None and val_qty.magnitude == 0:
                    self._add_issue(
                        level=ValidationIssueLevel.INFO, 
                        code_enum=SemanticIssueCode.DC_SHORT_R0_001,
                        component_id=comp_id, 
                        # **kwargs (component_id removed)
                        value_str=f"{val_qty:~P}"
                    )
            
            elif ComponentClass == Inductor:
                val_qty = get_const_param_val("inductance")
                if val_qty is not None:
                    if val_qty.magnitude == 0:
                        self._add_issue(
                            level=ValidationIssueLevel.INFO, 
                            code_enum=SemanticIssueCode.DC_SHORT_L0_001,
                            component_id=comp_id, 
                            value_str=f"{val_qty:~P}"
                        )
                    elif np.isinf(val_qty.magnitude):
                        self._add_issue(
                            level=ValidationIssueLevel.INFO, 
                            code_enum=SemanticIssueCode.DC_OPEN_LINF_001,
                            component_id=comp_id, 
                            value_str=f"{val_qty:~P}"
                        )
            
            elif ComponentClass == Capacitor:
                val_qty = get_const_param_val("capacitance")
                if val_qty is not None:
                    if np.isinf(val_qty.magnitude):
                        self._add_issue(
                            level=ValidationIssueLevel.INFO, 
                            code_enum=SemanticIssueCode.DC_SHORT_CINF_001,
                            component_id=comp_id, 
                            value_str=f"{val_qty:~P}"
                        )
                    elif val_qty.magnitude == 0:
                         self._add_issue(
                            level=ValidationIssueLevel.INFO, 
                            code_enum=SemanticIssueCode.DC_OPEN_C0_001,
                            component_id=comp_id, 
                            value_str=f"{val_qty:~P}"
                        )
            
            elif hasattr(sim_comp_instance, 'declare_dc_behavior') and \
                 isinstance(sim_comp_instance, ComponentBase) and \
                 callable(getattr(sim_comp_instance, 'declare_dc_behavior')):
                
                all_params_constant = True
                resolved_constant_params_for_dc: Dict[str, Quantity] = {}
                try:
                    declared_param_specs = ComponentClass.declare_parameters()
                except Exception: continue 

                for base_param_name in declared_param_specs.keys():
                    val = get_const_param_val(base_param_name)
                    if val is not None:
                        resolved_constant_params_for_dc[base_param_name] = val
                    else: 
                        all_params_constant = False; break
                
                if all_params_constant:
                    try:
                        # Type hint for clarity, assuming declare_dc_behavior matches this
                        dc_behavior_reports: Optional[List[Tuple[str, str, Dict[str, Any]]]]
                        dc_behavior_reports = sim_comp_instance.declare_dc_behavior(resolved_constant_params_for_dc) # type: ignore
                        
                        if dc_behavior_reports:
                            for report_item in dc_behavior_reports:
                                if isinstance(report_item, tuple) and len(report_item) == 3:
                                    issue_code_str, msg_content, details_dict_from_comp = report_item
                                    
                                    target_issue_code_enum: Optional[SemanticIssueCode] = None
                                    for code_enum_member in SemanticIssueCode:
                                        if code_enum_member.code == issue_code_str:
                                            target_issue_code_enum = code_enum_member
                                            break

                                    # Prepare combined kwargs for message formatting and for ValidationIssue.details
                                    # Start with details from component, then add/override with standard keys for message template
                                    combined_kwargs = dict(details_dict_from_comp or {})
                                    
                                    final_custom_message_str = msg_content
                                    # If msg_content itself is a template, format it using component's own details + comp_id/type
                                    if isinstance(msg_content, str) and '{' in msg_content and '}' in msg_content:
                                        comp_msg_fmt_args = dict(details_dict_from_comp or {})
                                        comp_msg_fmt_args['component_id'] = comp_id
                                        comp_msg_fmt_args['component_type'] = comp_type_str_for_msg
                                        try:
                                            final_custom_message_str = msg_content.format(**comp_msg_fmt_args)
                                        except KeyError as e_fmt:
                                            logger.warning(f"Component '{comp_id}' provided custom DC behavior message template "
                                                           f"'{msg_content}' but formatting failed (missing key {e_fmt}). Using raw message.")
                                    
                                    # These are for the standard DC_CUSTOM_INFO_001 template or a custom one if issue_code_str matched
                                    combined_kwargs['component_type'] = comp_type_str_for_msg
                                    combined_kwargs['custom_message'] = final_custom_message_str
                                    # component_id for message formatting is handled by explicit arg to _add_issue

                                    self._add_issue(
                                        level=ValidationIssueLevel.INFO,
                                        code_enum=target_issue_code_enum or SemanticIssueCode.DC_CUSTOM_INFO_001, 
                                        component_id=comp_id, # Explicit, for ValidationIssue.component_id & {component_id} in template
                                        details=None, # All details are passed via **combined_kwargs
                                        **combined_kwargs
                                    )
                                else:
                                    logger.warning(f"Component '{comp_id}' (type {comp_type_str_for_msg}) "
                                                   f"declare_dc_behavior returned an item not matching "
                                                   f"Tuple[str, str, Dict]: {report_item}")
                    except Exception as e:
                        logger.warning(f"Error calling 'declare_dc_behavior' or processing results for comp '{comp_id}': {e}", exc_info=True)