# src/rfsim_core/parser/parser.py
import logging
import re
import string
from pathlib import Path
from typing import Any, Dict, List, Set, Union

import cerberus
import yaml

from .raw_data import (
    ParsedCircuitNode,
    ParsedComponentData,
    ParsedLeafComponentData,
    ParsedSubcircuitData,
)
from .exceptions import ParsingError, SchemaValidationError

logger = logging.getLogger(__name__)

# --- CORRECTED REGEX DEFINITIONS ---

# Define the core identifier pattern without anchors. This is the fragment used for composition.
ID_REGEX_FRAGMENT = r"[a-zA-Z_][a-zA-Z0-9_]*"

# This regex enforces the "Simplicity Through Constraint" architectural mandate.
# It defines a valid identifier for a single token, explicitly forbidding '.' and '-'.
# It is built from the fragment and anchored to ensure a full string match.
ID_REGEX = f"^{ID_REGEX_FRAGMENT}$"
ALLOWED_ID_CHARS = set(string.ascii_letters + string.digits + "_")

# This more permissive regex allows dot-separated chains of valid identifiers.
# It is used exclusively for the keys in a subcircuit's 'parameters' override block.
# It is now correctly built from the un-anchored fragment.
PARAM_OVERRIDE_KEY_REGEX = f"^{ID_REGEX_FRAGMENT}(\\.{ID_REGEX_FRAGMENT})*$"


class EnhancedValidator(cerberus.Validator):
    """Custom Cerberus validator to enforce the project's strict naming conventions."""
    def __init__(self, *args, **kwargs):
        super(EnhancedValidator, self).__init__(*args, **kwargs)
        self.rules['id_regex'] = {'schema': {'type': 'boolean'}}
        self.rules['param_override_key_regex'] = {'schema': {'type': 'boolean'}}
        self.rules['unique_elements_by_key'] = {'schema': {'type': 'string'}}

    def _validate_id_regex(self, constraint: bool, field: str, value: Any):
        if not constraint: return
        if not isinstance(value, str):
            self._error(field, "must be a string to be validated by id_regex.")
            return

        if not re.match(ID_REGEX, value):
            invalid_chars = sorted(list(set(value) - ALLOWED_ID_CHARS))
            message = (
                f"Identifier '{value}' is invalid. Identifiers must start with a letter or underscore, "
                "and can only contain letters, numbers, and underscores. The dot '.' and hyphen '-' characters are forbidden. "
                f"This identifier contains the following forbidden character(s): {invalid_chars}"
            )
            self._error(field, message)

    def _validate_param_override_key_regex(self, constraint: bool, field: str, value: str):
        if constraint and isinstance(value, str) and not re.match(PARAM_OVERRIDE_KEY_REGEX, value):
            self._error(
                field,
                f"Parameter override key '{value}' is invalid. Must be a valid identifier or a "
                f"dot-separated chain of valid identifiers (e.g., 'gain' or 'amp1.R_load.value').",
            )

    def _validate_unique_elements_by_key(self, key_for_uniqueness: str, field: str, value: List[Dict]):
        """
        Validates that all dictionaries in a list have a unique value for a given key.
        The rule's arguments are validated against this schema:
        {'type': 'string'}
        """
        if not isinstance(value, list):
            return # Let the 'type: list' rule handle this.

        seen_keys = set()
        duplicates = []
        for item in value:
            if not isinstance(item, dict):
                continue # Let sub-schema validation handle this.
            
            item_key = item.get(key_for_uniqueness)
            if item_key is not None:
                if item_key in seen_keys:
                    duplicates.append(item_key)
                else:
                    seen_keys.add(item_key)
        
        if duplicates:
            unique_duplicates = sorted(list(set(duplicates)))
            self._error(field, f"Duplicate values found for key '{key_for_uniqueness}': {unique_duplicates}")

class NetlistParser:
    """
    Recursively parses and validates a hierarchy of netlist YAML files.
    Its sole responsibility is to produce a tree of Intermediate Representation (IR) objects.
    """
    _id_rule = {"type": "string", "required": True, "empty": False, "id_regex": True}
    _net_name_rule = {"type": "string", "required": True, "empty": False, "id_regex": True}
    _param_key_rule = {"type": "string", "required": True, "empty": False, "id_regex": True}
    _param_override_key_rule = {"type": "string", "required": True, "empty": False, "param_override_key_regex": True}

    _param_value_schema = {
        "oneof": [
            {"type": ["string", "number"]},
            {"type": "dict", "schema": {"expression": {"type": "string", "required": True}, "dimension": {"type": "string", "required": True}}},
        ]
    }

    _standard_component_schema = {
        "type": {"type": "string", "required": True, "id_regex": True, "forbidden": ["Subcircuit"]},
        "id": _id_rule,
        "ports": {"type": "dict", "required": True, "minlength": 1, "keysrules": {"oneof": [{"type": "string", "id_regex": True}, {"type": "integer"}]}, "valuesrules": _net_name_rule},
        "parameters": {"type": "dict", "required": False, "keysrules": _param_key_rule, "valuesrules": _param_value_schema},
    }

    _subcircuit_component_schema = {
        "type": {"type": "string", "required": True, "allowed": ["Subcircuit"]},
        "id": _id_rule,
        "definition_file": {"type": "string", "required": True, "empty": False},
        "ports": {"type": "dict", "required": False, "minlength": 1, "keysrules": _param_key_rule, "valuesrules": _net_name_rule},
        "parameters": {"type": "dict", "required": False, "keysrules": _param_override_key_rule, "valuesrules": _param_value_schema},
    }

    _schema = {
        "circuit_name": {"type": "string", "required": False, "id_regex": True},
        "ground_net": {"type": "string", "required": False, "id_regex": True, "default": "gnd"},
        "parameters": {"type": "dict", "required": False, "keysrules": _param_key_rule, "valuesrules": _param_value_schema},
        "components": {"type": "list", "required": True, "minlength": 1, "unique_elements_by_key": "id", "schema": {"type": "dict", "oneof_schema": [_standard_component_schema, _subcircuit_component_schema]}},
        "ports": {"type": "list", "required": False, "unique_elements_by_key": "id", "schema": {"type": "dict", "schema": {"id": _id_rule, "reference_impedance": {"type": "string", "required": True}}}},
        "sweep": {
            "type": "dict", "required": False, "schema": {
                "type": {"type": "string", "required": True, "allowed": ["linear", "log", "list"]},
                "start": {"type": "string", "required": True, "dependencies": {"type": ["linear", "log"]}},
                "stop": {"type": "string", "required": True, "dependencies": {"type": ["linear", "log"]}},
                "num_points": {"type": "integer", "required": True, "min": 1, "dependencies": {"type": ["linear", "log"]}},
                "points": {"type": "list", "required": True, "minlength": 1, "schema": {"type": ["string", "number"]}, "dependencies": {"type": ["list"]}},
            },
        },
    }

    def __init__(self):
        self._validator = EnhancedValidator(self._schema)
        self._validator.allow_unknown = False
        logger.info("NetlistParser initialized with strict structural validation rules.")

    def parse_to_circuit_tree(self, top_level_yaml_path: Union[str, Path]) -> ParsedCircuitNode:
        """Parses the top-level YAML and all subcircuits, returning the complete IR tree."""
        top_path = Path(top_level_yaml_path).resolve()
        logger.info(f"Starting hierarchical parsing from top-level file: {top_path}")
        return self._parse_recursive(top_path, visited_paths=set())

    def _parse_recursive(self, yaml_path: Path, visited_paths: Set[Path]) -> ParsedCircuitNode:
        """Internal recursive function that parses one file and returns its IR node."""
        resolved_path = yaml_path.resolve()
        if resolved_path in visited_paths:
            raise ParsingError(f"Circular subcircuit dependency detected involving: {resolved_path}", file_path=resolved_path)
        
        visited_paths.add(resolved_path)
        logger.debug(f"Parsing definition file: {resolved_path}")

        yaml_content = self._load_yaml(resolved_path)
        if not self._validator.validate(yaml_content):
            raise SchemaValidationError(self._validator.errors, resolved_path)
        
        validated_data = self._validator.document

        parsed_components: List[ParsedComponentData] = []
        for comp_data_raw in validated_data.get("components", []):
            comp_id = comp_data_raw["id"]
            if comp_data_raw["type"] == "Subcircuit":
                sub_def_file_path = (resolved_path.parent / comp_data_raw["definition_file"]).resolve()
                sub_node = self._parse_recursive(sub_def_file_path, visited_paths.copy())
                parsed_components.append(
                    ParsedSubcircuitData(
                        instance_id=comp_id,
                        component_type="Subcircuit",
                        definition_file_path=sub_def_file_path,
                        sub_circuit_definition_node=sub_node,
                        raw_port_mapping=comp_data_raw.get("ports", {}),
                        raw_parameter_overrides=comp_data_raw.get("parameters", {}),
                        source_yaml_path=resolved_path,
                    )
                )
            else:
                parsed_components.append(
                    ParsedLeafComponentData(
                        instance_id=comp_id,
                        component_type=comp_data_raw["type"],
                        raw_ports_dict=comp_data_raw["ports"],
                        raw_parameters_dict=comp_data_raw.get("parameters", {}),
                        source_yaml_path=resolved_path,
                    )
                )

        return ParsedCircuitNode(
            circuit_name=validated_data.get("circuit_name", resolved_path.stem),
            ground_net_name=validated_data["ground_net"],
            source_yaml_path=resolved_path,
            components=parsed_components,
            raw_parameters_dict=validated_data.get("parameters", {}),
            raw_external_ports_list=validated_data.get("ports", []),
            raw_sweep_config=validated_data.get("sweep"),
        )

    def _load_yaml(self, source: Path) -> Dict[str, Any]:
        """Loads and performs basic sanity checks on a YAML file."""
        if not source.is_file():
            raise ParsingError(details=f"Netlist file not found at path: {source}", file_path=source)
        try:
            with source.open("r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
            if content is None:
                raise ParsingError(details=f"The YAML file is empty or contains no valid content.", file_path=source)
            if not isinstance(content, dict):
                raise ParsingError(details=f"The root of the YAML file must be a dictionary (mapping).", file_path=source)
            return content
        except PermissionError as e:
            raise ParsingError(details=f"Permission denied when trying to read file: {e}", file_path=source) from e
        except yaml.YAMLError as e:
            raise ParsingError(details=f"Invalid YAML syntax: {e}", file_path=source) from e