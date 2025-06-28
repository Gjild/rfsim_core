# src/rfsim_core/validation/issue_codes.py
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
    NET_CONN_002 = ("NET_CONN_002", "Internal net '{net_name}' has only a single connection to component '{connected_to_component}' port '{connected_to_port}'.")

    # --- Component Type & Port Issues (COMP_...) ---
    COMP_TYPE_001 = ("COMP_TYPE_001", "Component '{component_fqn}' specifies an unregistered type '{component_type}'. Available types: {available_types}.")
    COMP_LEAF_PORT_DEF_UNDECLARED = ("COMP_LEAF_PORT_DEF_UNDECLARED", "Component '{component_fqn}' uses undeclared port(s): {extra_ports}. Declared ports are: {declared_ports}.")
    COMP_LEAF_PORT_DEF_MISSING = ("COMP_LEAF_PORT_DEF_MISSING", "Component '{component_fqn}' is missing required connections for port(s): {missing_ports}. All declared ports are: {declared_ports}.")

    # --- External Port Issues (EXT_PORT_...) ---
    EXT_PORT_001 = ("EXT_PORT_001", "External port '{net_name}' cannot be the ground net.")
    EXT_PORT_002 = ("EXT_PORT_002", "External port net '{net_name}' has no component connections.")
    EXT_PORT_Z0_MISSING = ("EXT_PORT_Z0_MISSING", "External port '{net_name}' reference impedance (Z0) string is missing or empty.")
    EXT_PORT_Z0_DIM_MISMATCH = ("EXT_PORT_Z0_DIM_MISMATCH", "External port '{net_name}' Z0 literal '{value}' has dimensionality '{parsed_dimensionality}', which is not compatible with ohms.")

    # --- Ground Net Issues (GND_...) ---
    GND_CONN_001 = ("GND_CONN_001", "Ground net '{net_name}' has no component connections, although components exist in the circuit.")

    # --- Leaf Component Parameter Issues (PARAM_...) ---
    PARAM_LEAF_UNDCL = ("PARAM_LEAF_UNDCL", "Component '{component_fqn}' defines parameter '{parameter_name}' which is not declared by its type. Declared parameters are: {declared_params}.")
    PARAM_LEAF_MISSING = ("PARAM_LEAF_MISSING", "Component '{component_fqn}' is missing required parameter '{parameter_name}'. All declared parameters are: {declared_params}.")
    PARAM_LEAF_DIM_MISMATCH = ("PARAM_LEAF_DIM_MISMATCH", "Constant value for parameter '{parameter_name}' of component '{component_fqn}' (resolved to '{resolved_value_str}') is not dimensionally compatible with the expected dimension '{expected_dim_str}'.")

    # --- Subcircuit Instantiation Issues (SUB_INST_...) ---
    SUB_INST_PORT_MAP_UNDECLARED = ("SUB_INST_PORT_MAP_UNDECLARED", "Subcircuit instance '{instance_fqn}' maps port '{undeclared_sub_port_name}', which is not an external port of its definition '{sub_def_name}'. Available ports: {available_sub_ports}.")
    SUB_INST_PORT_MAP_MISSING = ("SUB_INST_PORT_MAP_MISSING", "Subcircuit instance '{instance_fqn}' fails to map required external port(s) from its definition '{sub_def_name}': {missing_sub_ports}.")
    SUB_INST_PORT_MAP_REQUIRED = ("SUB_INST_PORT_MAP_REQUIRED", "Subcircuit instance '{instance_fqn}' must include a 'ports' block because its definition '{sub_def_name}' declares external ports.")
    SUB_INST_PARAM_OVERRIDE_UNDECLARED = ("SUB_INST_PARAM_OVERRIDE_UNDECLARED", "Subcircuit instance '{instance_fqn}' attempts to override parameter '{override_target_in_sub}', which does not exist in the subcircuit definition.")
    SUB_INST_PARAM_OVERRIDE_DIM_MISMATCH = ("SUB_INST_PARAM_OVERRIDE_DIM_MISMATCH", "Subcircuit instance '{instance_fqn}' provides override value '{override_value_str}' for parameter '{override_target_in_sub}'. The value's dimension ('{provided_dim_str}') is incompatible with the parameter's declared dimension ('{expected_dim_str}').")

    # --- Ideal DC Path Identification Info (DC_INFO_...) ---
    DC_INFO_SHORT_R0 = ("DC_INFO_SHORT_R0", "Component '{component_fqn}' (Resistor) with value {value_str} will be treated as an ideal DC short.")
    DC_INFO_SHORT_L0 = ("DC_INFO_SHORT_L0", "Component '{component_fqn}' (Inductor) with value {value_str} will be treated as an ideal DC short.")
    DC_INFO_OPEN_LINF = ("DC_INFO_OPEN_LINF", "Component '{component_fqn}' (Inductor) with value {value_str} will be treated as an ideal DC open.")
    DC_INFO_SHORT_CINF = ("DC_INFO_SHORT_CINF", "Component '{component_fqn}' (Capacitor) with value {value_str} will be treated as an ideal DC short.")
    DC_INFO_OPEN_C0 = ("DC_INFO_OPEN_C0", "Component '{component_fqn}' (Capacitor) with value {value_str} will be treated as an ideal DC open.")

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