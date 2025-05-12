# --- src/rfsim_core/validation/issue_codes.py ---
import logging
from enum import Enum
from typing import Tuple

logger = logging.getLogger(__name__)

class SemanticIssueCode(Enum):
    """
    Registry of semantic validation issue codes and their message templates.
    Each enum member's value is a tuple: (code_str, message_template_str).
    """

    # --- Net Connectivity Issues (NET_CONN_...) ---
    NET_CONN_001 = ("NET_CONN_001", "Internal net '{net_name}' is defined but has no component connections (completely floating).")
    NET_CONN_002 = ("NET_CONN_002", "Internal net '{net_name}' has only a single component connection (stub connection). Connected to component '{connected_to_component}' port '{connected_to_port}'.")

    # --- Component Type Issues (COMP_TYPE_...) ---
    COMP_TYPE_001 = ("COMP_TYPE_001", "Component '{component_id}' specifies an unregistered type '{unregistered_type}'. Available types: {available_types}.")

    # --- Port Definition Issues (PORT_DEF_...) ---
    PORT_DEF_001 = ("PORT_DEF_001", "Component '{component_id}' (type '{component_type}') uses undeclared port(s): {extra_ports}. Declared ports are: {declared_ports}.")
    PORT_DEF_002 = ("PORT_DEF_002", "Component '{component_id}' (type '{component_type}') is missing required connections for port(s): {missing_ports}. All declared ports are: {declared_ports}.")
    COMP_PORT_DECL_FAIL_001 = ("COMP_PORT_DECL_FAIL_001", "Failed to retrieve port declarations for component type '{component_type}' (instance '{component_id}'): {error_details}.") # NEW

    # --- Port Connection Issues (PORT_CONN_...) ---
    PORT_UNLINKED_001 = ("PORT_UNLINKED_001", "Port '{port_id}' on component '{component_id}' is not linked to any net object. The YAML specified net name '{original_net_name_from_yaml}' for this port, which might be undefined or an internal error occurred during net assignment.")
    PORT_CONN_002 = ("PORT_CONN_002", "Port '{port_id}' on component '{component_id}' connects to net '{net_name}', but this net object is not registered in the circuit's net dictionary. This indicates a severe internal inconsistency.")

    # --- External Port Issues (EXT_PORT_...) ---
    EXT_PORT_001 = ("EXT_PORT_001", "External port '{net_name}' cannot be the ground net.")
    EXT_PORT_002 = ("EXT_PORT_002", "External port net '{net_name}' has no component connections.")
    EXT_PORT_003 = ("EXT_PORT_003", "External port '{net_name}' reference impedance (Z0) string is missing or empty.")
    EXT_PORT_DIM_001 = ("EXT_PORT_DIM_001", "External port '{net_name}' Z0 literal '{value}' (parsed with dimensionality '{parsed_dimensionality}') is not dimensionally compatible with ohms.")
    EXT_PORT_REF_001 = ("EXT_PORT_REF_001", "External port '{net_name}' Z0 reference '{parameter_name}' does not match any defined global parameter.")
    EXT_PORT_REF_DIM_001 = ("EXT_PORT_REF_DIM_001", "External port '{net_name}' Z0 references global parameter '{parameter_name}' whose declared dimension ('{declared_dimension_of_ref}') is not compatible with ohms.")
    EXT_PORT_REF_VALIDATE_FAIL_001 = ("EXT_PORT_REF_VALIDATE_FAIL_001", "External port '{net_name}' Z0 reference '{parameter_name}' could not be validated due to an unexpected internal error: {error_details}.") # NEW

    # --- Ground Net Issues (GND_CONN_...) ---
    GND_CONN_001 = ("GND_CONN_001", "Ground net '{net_name}' has no component connections, although components exist in the circuit.")

    # --- Component Parameter Dimensionality Issues (PARAM_DIM_...) ---
    PARAM_DIM_001 = ("PARAM_DIM_001", "Parameter '{parameter_name}' for component '{component_id}' has a declared dimension ('{declared_in_pm}') in the ParameterManager that is inconsistent with the dimension ('{expected_by_comp_type}') expected by its component type '{component_type}'.")
    PARAM_VAL_DIM_001 = ("PARAM_VAL_DIM_001", "Constant value for parameter '{parameter_name}' of component '{component_id}' (resolved to {resolved_value_str}) is not dimensionally compatible with the expected dimension '{expected_dim_str}'. Its resolved dimensionality is '{resolved_value_dim}'.")
    PARAM_CONST_VALIDATE_FAIL_001 = ("PARAM_CONST_VALIDATE_FAIL_001", "Failed to validate constant value for parameter '{parameter_name}' of component '{component_id}' due to an unexpected internal error: {error_details}. Expected dimension: '{expected_dim_str}'.") # NEW

    # --- Ideal DC Path Identification Info (DC_...) ---
    DC_SHORT_R0_001 = ("DC_SHORT_R0_001", "Component '{component_id}' (Resistor) has resistance R=0 ({value_str}). Will be treated as an ideal DC short in DC analysis.")
    DC_SHORT_L0_001 = ("DC_SHORT_L0_001", "Component '{component_id}' (Inductor) has inductance L=0 ({value_str}). Will be treated as an ideal DC short in DC analysis.")
    DC_OPEN_LINF_001 = ("DC_OPEN_LINF_001", "Component '{component_id}' (Inductor) has inductance L=inf ({value_str}). Will be treated as an ideal DC open in DC analysis.")
    DC_SHORT_CINF_001 = ("DC_SHORT_CINF_001", "Component '{component_id}' (Capacitor) has capacitance C=inf ({value_str}). Will be treated as an ideal DC short in DC analysis.")
    DC_OPEN_C0_001 = ("DC_OPEN_C0_001", "Component '{component_id}' (Capacitor) has capacitance C=0 ({value_str}). Will be treated as an ideal DC open in DC analysis.")
    DC_CUSTOM_INFO_001 = ("DC_CUSTOM_INFO_001", "Component '{component_id}' (type '{component_type}') reports custom DC behavior: {custom_message}.") # Message is fully formed by component

    # --- ID Uniqueness/Validity (ID_...) ---
    ID_COMP_INVALID_001 = ("ID_COMP_INVALID_001", "Invalid component ID '{component_id}' (found to be None or empty string in circuit.sim_components). This indicates a severe internal inconsistency during circuit building.")
    ID_NET_INVALID_001 = ("ID_NET_INVALID_001", "Invalid net name '{net_name}' (found to be None or empty string in circuit.nets). This indicates an internal inconsistency or a problem during net creation.")

    # --- Undeclared/Missing Instance Parameters (PARAM_INST_...) ---
    PARAM_INST_UNDCL_001 = ("PARAM_INST_UNDCL_001", "Component '{component_id}' (type '{component_type}') defines parameter '{parameter_name}' which is not declared by the component type. Declared parameters are: {declared_params}.")
    PARAM_INST_MISSING_001 = ("PARAM_INST_MISSING_001", "Component '{component_id}' (type '{component_type}') is missing required parameter '{parameter_name}'. All declared parameters are: {declared_params}.")


    @property
    def code(self) -> str:
        return self.value[0]

    @property
    def template(self) -> str:
        return self.value[1]

    def format_message(self, **kwargs) -> str:
        """Formats the message template with provided keyword arguments."""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing key {e} for formatting message template of {self.name} (code: {self.code}): '{self.template}'. Provided args: {kwargs}")
            return f"Error formatting message for {self.code}: Missing key {e}. Template: '{self.template}' Args: {kwargs}"