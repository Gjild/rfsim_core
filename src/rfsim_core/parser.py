# src/rfsim_core/parser.py
import logging
import yaml
import cerberus
from pathlib import Path
from typing import Dict, Any, Union, TextIO

from .data_structures import Circuit, Component, Net, Port
from .parameters import ParameterManager, ParameterError
from .units import ureg # For potential future use here, though mainly in ParameterManager now

logger = logging.getLogger(__name__)

class SchemaValidationError(ValueError):
    """Custom exception for YAML schema validation errors."""
    def __init__(self, errors, *args):
        self.errors = errors
        message = f"YAML schema validation failed: {errors}"
        super().__init__(message, *args)

class ParsingError(ValueError):
    """Custom exception for general netlist parsing errors."""
    pass


class NetlistParser:
    """
    Parses and validates a circuit netlist from a YAML source.
    Builds an internal Circuit representation.
    """

    # Basic schema for Phase 1
    _schema = {
        'components': {
            'type': 'list',
            'required': True,
            'empty': False, # Usually want at least one component? Or allow empty? Let's allow empty for now.
            'schema': {
                'type': 'dict',
                'schema': {
                    'type': {'type': 'string', 'required': True, 'empty': False},
                    'id': {'type': 'string', 'required': True, 'empty': False},
                    'ports': {
                        'type': 'dict',
                        'required': True,
                        'minlength': 1, # Require at least one port connection
                        'keysrules': {'type': 'string', 'empty': False}, # Port IDs/names are strings
                        'valuesrules': {'type': 'string', 'empty': False} # Net names are strings
                    },
                    'parameters': { # Component parameters (raw values for now)
                        'type': 'dict',
                        'required': False,
                        'keysrules': {'type': 'string', 'empty': False},
                        # Values can be strings, numbers, etc. No strict check yet.
                    }
                }
            }
        },
        'ports': { # External ports definition
            'type': 'list',
            'required': False, # Optional, can have circuits with no external ports
            'schema': {
                'type': 'dict',
                'schema': {
                    'id': {'type': 'string', 'required': True, 'empty': False},
                    'reference_impedance': {
                        'type': 'string', # Strictly string for Phase 1
                        'required': True,
                        'empty': False
                    }
                }
            }
        },
        'parameters': { # Global parameters
            'type': 'dict',
            'required': False,
            'keysrules': {'type': 'string', 'empty': False},
            # Values can be strings or numbers, ParameterManager will parse
            'valuesrules': {'type': ['string', 'number']}
        },
         'circuit_name': { # Optional circuit name
            'type': 'string',
            'required': False,
            'empty': False
        },
        'ground_net': { # Optional override for ground net name
             'type': 'string',
             'required': False,
             'empty': False
        }
    }

    def __init__(self):
        self._validator = cerberus.Validator(self._schema)
        logger.info("NetlistParser initialized.")

    def parse(self, source: Union[str, Path, TextIO]) -> Circuit:
        """
        Parses the YAML netlist source and returns a Circuit object.

        Args:
            source: YAML source, can be a file path (str or Path),
                    a YAML string, or an open file-like object.

        Returns:
            A populated Circuit object.

        Raises:
            FileNotFoundError: If source is a path and the file doesn't exist.
            yaml.YAMLError: If the YAML syntax is invalid.
            SchemaValidationError: If the YAML structure doesn't match the schema.
            ParsingError: For other logical errors during parsing (e.g., duplicates).
            ParameterError: For errors parsing global parameters.
        """
        logger.info(f"Starting parsing of netlist source: {type(source)}")
        yaml_content = self._load_yaml(source)
        if not self._validator.validate(yaml_content):
            logger.error(f"Schema validation failed. Errors: {self._validator.errors}")
            raise SchemaValidationError(self._validator.errors)

        logger.info("YAML schema validation successful.")
        validated_data = self._validator.document # Use the normalized/validated data

        # --- Build Circuit Object ---
        circuit_name = validated_data.get('circuit_name', 'UnnamedCircuit')
        ground_name = validated_data.get('ground_net', 'gnd')
        circuit = Circuit(name=circuit_name, ground_net_name=ground_name)

        # 1. Process Global Parameters
        global_params_data = validated_data.get('parameters', {})
        try:
            circuit.parameter_manager = ParameterManager(global_params_data)
            logger.debug(f"Parameter Manager state:\n{circuit.parameter_manager}")
        except ParameterError as e:
             logger.error(f"Failed to initialize ParameterManager: {e}")
             raise # Re-raise the specific error

        # 2. Process Components and Connections
        component_errors = []
        for comp_data in validated_data.get('components', []):
            comp_id = comp_data['id']
            comp_type = comp_data['type']
            comp_params = comp_data.get('parameters', {}) # Get raw component parameters

            if comp_id in circuit.components:
                component_errors.append(f"Duplicate component ID '{comp_id}' found.")
                continue # Skip adding this duplicate component

            component = Component(
                instance_id=comp_id,
                component_type=comp_type,
                parameters=comp_params # Store raw parameters
            )

            # Create ports and connect them to nets
            for port_id, net_name in comp_data['ports'].items():
                try:
                    # Ensure port is created on the component
                    if port_id not in component.ports:
                        component.add_port(port_id)
                    port = component.ports[port_id]

                    # Get or create the net
                    net = circuit.get_or_create_net(net_name)

                    # Establish connections
                    if port.net is not None and port.net != net:
                         # This condition is tricky, maybe redundant if IDs are unique
                         component_errors.append(f"Port '{port_id}' on component '{comp_id}' cannot be connected to multiple nets ('{port.net.name}' and '{net_name}').")
                    elif port.net is None:
                         port.net = net
                         if component not in net.connected_components:
                             net.connected_components.append(component)
                         logger.debug(f"Connected port '{port_id}' of '{comp_id}' to net '{net_name}'.")

                except Exception as e:
                    component_errors.append(f"Error processing port '{port_id}' for component '{comp_id}': {e}")

            # Add component to circuit only if no errors during its processing
            if not any(comp_id in err for err in component_errors[-len(comp_data['ports']):]): # Basic check
                 circuit.add_component(component)

        if component_errors:
            raise ParsingError("Errors occurred during component processing:\n- " + "\n- ".join(component_errors))


        # 3. Process External Ports
        port_errors = []
        for port_data in validated_data.get('ports', []):
            port_id = port_data['id'] # This is the net name used as the external port
            impedance_str = port_data['reference_impedance']
            try:
                circuit.set_external_port(port_id, impedance_str)
            except Exception as e:
                port_errors.append(f"Error processing external port '{port_id}': {e}")

        if port_errors:
            raise ParsingError("Errors occurred during external port processing:\n- " + "\n- ".join(port_errors))

        # --- Final Checks (Example - could be expanded in Semantic Validation Phase) ---
        if circuit.ground_net_name not in circuit.nets:
             logger.warning(f"Designated ground net '{circuit.ground_net_name}' was not explicitly used by any component or port. Creating it.")
             circuit.get_or_create_net(circuit.ground_net_name, is_ground=True) # Ensure it exists

        logger.info(f"Successfully parsed netlist and built circuit '{circuit.name}'.")
        logger.info(f"Found {len(circuit.components)} components, {len(circuit.nets)} nets ({len(circuit.external_ports)} external).")
        return circuit


    def _load_yaml(self, source: Union[str, Path, TextIO]) -> Dict[str, Any]:
        """Loads YAML from various source types."""
        try:
            if isinstance(source, Path):
                logger.debug(f"Loading YAML from path: {source}")
                # Check if the path exists and is a file before trying to open
                if not source.exists() or not source.is_file():
                     raise FileNotFoundError(f"YAML file not found or is not a regular file at path: {source}")
                with source.open('r') as f:
                    return yaml.safe_load(f)
            elif isinstance(source, str):
                path = Path(source)
                # 1) If it's an existing file (any extension), load it.
                if path.exists() and path.is_file():
                    logger.debug(f"Loading YAML from path string: {source}")
                    try:
                        with path.open('r') as f:
                            return yaml.safe_load(f)
                    except Exception as e:
                        logger.error(f"Failed to read file at path: {source}: {e}")
                        raise ParsingError(f"Failed to read file at path: {source}: {e}") from e

                # 2) If it looks like a YAML filename but doesn't actually exist, error out.
                if path.suffix.lower() in (".yaml", ".yml"):
                    logger.error(f"YAML file not found or is not a regular file at path: {source}")
                    raise FileNotFoundError(f"YAML file not found or is not a regular file at path: {source}")

                # 3) Otherwise, treat the string itself as YAML content.
                logger.debug("Loading YAML from string content.")
                return yaml.safe_load(source)

            elif hasattr(source, 'read'): # File-like object
                 logger.debug("Loading YAML from file-like object.")
                 return yaml.safe_load(source)
            else:
                raise TypeError(f"Unsupported source type for YAML loading: {type(source)}")
        except FileNotFoundError:
            # This block now correctly catches FileNotFoundError raised above
            # or potentially from path.open if used without the exists() check (less likely now)
            logger.error(f"YAML file not found at path: {source}")
            raise # Re-raise the FileNotFoundError
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML syntax: {e}")
            raise ParsingError(f"Invalid YAML syntax: {e}") from e
        except Exception as e:
            # This broader catch is for other unexpected errors during loading/parsing
            logger.error(f"Failed to load YAML source: {e}", exc_info=True)
            raise ParsingError(f"Failed to load YAML source: {e}") from e