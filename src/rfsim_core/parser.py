# --- src/rfsim_core/parser.py ---
import logging
import yaml
import cerberus
import pint
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union, TextIO, Tuple, Set

from .data_structures import Circuit, Component, Net, Port
from .parameters import ParameterManager, ParameterError
from .components import ComponentError
from .units import ureg

from .components import COMPONENT_REGISTRY

logger = logging.getLogger(__name__)

class SchemaValidationError(ValueError):
    def __init__(self, errors, *args):
        self.errors = errors
        message = f"YAML schema validation failed: {errors}"
        super().__init__(message, *args)

class ParsingError(ValueError):
    pass


class NetlistParser:
    """
    Parses and validates a circuit netlist from a YAML source.
    Builds an internal Circuit representation and extracts sweep frequencies.
    """
    # Schema updated slightly for clarity and sweep validation
    _schema = {
        'circuit_name': {'type': 'string', 'required': False, 'empty': False},
        'ground_net': {'type': 'string', 'required': False, 'empty': False},
        'parameters': {
            'type': 'dict', 'required': False, 'keysrules': {'type': 'string', 'empty': False},
            'valuesrules': {'type': ['string', 'number']}
        },
        'components': {
            'type': 'list', 'required': True, 'minlength': 0, # Allow circuits with only ports?
            'schema': {
                'type': 'dict',
                'schema': {
                    'type': {'type': 'string', 'required': True, 'empty': False},
                    'id': {'type': 'string', 'required': True, 'empty': False},
                    'ports': {
                        'type': 'dict', 'required': True, 'minlength': 1,
                        'keysrules': {'type': ['string', 'integer'], 'empty': False}, # Allow int port IDs
                        'valuesrules': {'type': 'string', 'empty': False}
                    },
                    'parameters': {
                        'type': 'dict', 'required': False, 'keysrules': {'type': 'string', 'empty': False},
                    }
                }
            }
        },
        'ports': {
            'type': 'list', 'required': False, 'minlength': 0,
            'schema': {
                'type': 'dict',
                'schema': {
                    'id': {'type': 'string', 'required': True, 'empty': False},
                    'reference_impedance': {
                        'type': 'string', 'required': True, 'empty': False
                    }
                }
            }
        },
        'sweep': {
            'type': 'dict', 'required': True,
            'schema': {
                'type': {'type': 'string', 'required': True, 'allowed': ['linear', 'log', 'list']},
                'start': {'type': 'string', 'required': False, 'dependencies': {'type': ['linear', 'log']}},
                'stop': {'type': 'string', 'required': False, 'dependencies': {'type': ['linear', 'log']}},
                'num_points': {'type': 'integer', 'required': False, 'min': 2, 'dependencies': {'type': ['linear', 'log']}},
                'points': {'type': 'list', 'required': False, 'minlength': 1, 'schema': {'type': 'string'}, 'dependencies': {'type': ['list']}},
            }
        }
    }

    def __init__(self):
        self._validator = cerberus.Validator(self._schema)
        self._validator.allow_unknown = False
        logger.info("NetlistParser initialized.")

    def parse(self, source: Union[str, Path, TextIO]) -> Tuple[Circuit, np.ndarray]:
        """
        Parses the YAML netlist source.

        Args:
            source: YAML source (path, string, or file-like object).

        Returns:
            A tuple containing:
                - circuit: The populated Circuit object.
                - freq_array_hz: NumPy array of frequency points (in Hz, guaranteed > 0).

        Raises:
            FileNotFoundError, yaml.YAMLError, SchemaValidationError, ParsingError, ParameterError.
        """
        logger.info(f"Starting parsing of netlist source: {type(source)}")
        yaml_content = self._load_yaml(source)
        if not self._validator.validate(yaml_content):
            logger.error(f"Schema validation failed. Errors: {self._validator.errors}")
            raise SchemaValidationError(self._validator.errors)

        logger.info("YAML schema validation successful.")
        validated_data = self._validator.document

        # --- Build Circuit Object (Initial Structure) ---
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
             raise

        # 2. Process Components and Connections
        component_errors = []
        unique_comp_ids = set()
        used_net_names: Set[str] = {ground_name} # Track all nets mentioned
        net_connection_counts: Dict[str, int] = {} # net_name -> count of component ports connected

        for comp_data_raw in validated_data.get('components', []):
            comp_id = comp_data_raw['id']
            comp_type = comp_data_raw['type']
            comp_params = comp_data_raw.get('parameters', {})

            # Basic Semantic Check: Duplicate component ID
            if comp_id in unique_comp_ids:
                component_errors.append(f"Duplicate component ID '{comp_id}' found.")
                continue
            unique_comp_ids.add(comp_id)

            # Basic Semantic Check: Known component type
            if comp_type not in COMPONENT_REGISTRY:
                 error_msg = f"Unknown component type '{comp_type}' for instance '{comp_id}'. Registered types: {list(COMPONENT_REGISTRY.keys())}"
                 component_errors.append(error_msg)
                 raise ComponentError(err_msg)

            component = Component(instance_id=comp_id, component_type=comp_type, parameters=comp_params)
            has_error_for_this_comp = False

            # Create ports and connect them to nets, count connections
            for port_id, net_name in comp_data_raw['ports'].items():
                used_net_names.add(net_name) # Keep track of all used net names
                try:
                    # Port ID uniqueness *within the component instance* is checked here
                    port = component.add_port(port_id)
                    net = circuit.get_or_create_net(net_name)

                    # Establish connections
                    if port.net is not None: # Should not happen due to add_port check, but safety first
                         raise ValueError(f"Internal error: Port '{port_id}' on '{comp_id}' already connected.")
                    port.net = net
                    if component not in net.connected_components:
                        net.connected_components.append(component)

                    # Increment connection count for the net
                    net_connection_counts[net_name] = net_connection_counts.get(net_name, 0) + 1
                    logger.debug(f"Connected port '{port_id}' of '{comp_id}' to net '{net_name}'. Net count: {net_connection_counts[net_name]}")

                # Establish connections (port -> net, net -> component)
                except Exception as e:
                    err_msg = f"Error processing connection port='{port_id}', net='{net_name}' for component '{comp_id}': {e}"
                    component_errors.append(err_msg)
                    logger.error(err_msg)
                    has_error_for_this_comp = True

            # Add component to circuit only if no errors *during its processing*
            if not has_error_for_this_comp:
                 circuit.add_component(component)

        # Raise accumulated errors from component structure processing
        if component_errors:
            raise ParsingError("Errors occurred during component structure processing:\n- " + "\n- ".join(component_errors))

        # 3. Process External Ports
        port_errors = []
        external_port_names = set()
        external_port_nets: Set[str] = set() # Track nets designated as external ports
        for port_data in validated_data.get('ports', []):
            port_id = port_data['id'] # This is the net name used as the external port
            impedance_str = port_data['reference_impedance']

            if port_id in external_port_names:
                    port_errors.append(f"Duplicate external port definition for net '{port_id}'.")
                    continue
            external_port_names.add(port_id)
            external_port_nets.add(port_id)

            try:
                # Add this line back:
                circuit.set_external_port(port_id, impedance_str)
                logger.debug(f"Registered external port '{port_id}' via circuit.set_external_port.") # Optional debug log

            except Exception as e:
                # Log the specific error during port setting
                err_msg = f"Error setting external port '{port_id}' in circuit object: {e}"
                logger.error(err_msg)
                port_errors.append(err_msg) # Append the specific error

        if port_errors:
            # Ensure the detailed errors are reported
            raise ParsingError("Errors occurred during external port processing:\n- " + "\n- ".join(port_errors))

        # 4. Process Sweep Configuration (No Change Structurally)
        sweep_data = validated_data.get('sweep') # Schema ensures it exists
        try:
             freq_array_hz = self._parse_sweep(sweep_data)
             logger.info(f"Parsed frequency sweep: {len(freq_array_hz)} points from {freq_array_hz[0]:.3e} Hz to {freq_array_hz[-1]:.3e} Hz.")
        except (ValueError, pint.errors.UndefinedUnitError, pint.errors.DimensionalityError) as e:
             raise ParsingError(f"Failed to parse sweep configuration: {e}") from e

        # --- Post-Parse Semantic Validation ---
        try:
            self._validate_parsed_structure(circuit, used_net_names, net_connection_counts, external_port_nets)
        except ParsingError as e:
            logger.error(f"Semantic validation failed: {e}")
            raise

        logger.info(f"Successfully parsed and validated netlist structure '{circuit.name}'.")
        logger.info(f"Found {len(circuit.components)} components, {len(circuit.nets)} nets ({len(circuit.external_ports)} external).")

        return circuit, freq_array_hz
    
    def _validate_parsed_structure(
        self,
        circuit: Circuit,
        used_net_names: Set[str],
        net_connection_counts: Dict[str, int],
        external_port_nets: Set[str]
    ):
        """Performs semantic checks on the parsed circuit structure."""
        logger.info("Performing semantic validation on parsed structure...")
        errors = []
        warnings = []
        ground_name = circuit.ground_net_name
        ground_net_object = circuit.get_ground_net() # Ensures it exists

        # Check 1: Ground Net Connectivity
        if ground_name not in net_connection_counts:
            # This means ground was defined but literally nothing connected to it.
            warnings.append(f"Ground net '{ground_name}' is defined but no component ports are connected to it.")
        elif net_connection_counts[ground_name] == 0:
             # This case should ideally not happen if the first check passes, but defensively:
             warnings.append(f"Ground net '{ground_name}' has zero connections recorded (internal inconsistency?).")

        # Check 2: External Port Connectivity
        for port_name in external_port_nets:
            if port_name not in used_net_names:
                # This check is slightly redundant if parsing logic worked, but good backup.
                errors.append(f"External port '{port_name}' is defined but the net name was never used by any component.")
            elif port_name not in net_connection_counts or net_connection_counts[port_name] == 0:
                 errors.append(f"External port '{port_name}' is defined but no component ports are connected to its net.")
            # Future: Add check if external port net *only* connects to other external ports (useless loop)

        # Check 3: Floating Internal Nets
        internal_nets = set(circuit.nets.keys()) - {ground_name} - external_port_nets
        for net_name in internal_nets:
            count = net_connection_counts.get(net_name, 0)
            if count < 2:
                if count == 1:
                    # Find the single component/port connected
                    connected_details = "unknown connection"
                    for comp in circuit.components.values():
                        for port_id, port_obj in comp.ports.items():
                            if port_obj.net and port_obj.net.name == net_name:
                                connected_details = f"component '{comp.instance_id}' port '{port_id}'"
                                break
                        else: continue
                        break
                    warnings.append(f"Internal net '{net_name}' is only connected to one component port ({connected_details}). This net is floating.")
                elif count == 0:
                    # This implies a net was created (e.g., get_or_create_net) but never used in a connection.
                    # Should be less common if used_net_names check works, but possible.
                    warnings.append(f"Internal net '{net_name}' exists but has zero component connections. This net is floating.")

        # Report errors/warnings
        if errors:
            raise ParsingError("Semantic validation errors found:\n- " + "\n- ".join(errors))
        if warnings:
            logger.warning("Semantic validation warnings found:\n- " + "\n- ".join(warnings))
        else:
            logger.info("Semantic validation successful (no errors or warnings).")

    def _parse_sweep(self, sweep_data: Dict[str, Any]) -> np.ndarray:
        """Parses the validated sweep dictionary and returns frequency array in Hz (> 0)."""
        sweep_type = sweep_data['type']
        logger.info(f"Parsing '{sweep_type}' sweep.")
        freq_values_hz = np.array([], dtype=float) # Initialize as empty array

        try:
            if sweep_type == 'linear' or sweep_type == 'log':
                required = ['start', 'stop', 'num_points']
                if not all(k in sweep_data for k in required):
                     raise ValueError(f"Missing required fields {required} for '{sweep_type}' sweep.")

                start_qty = ureg.Quantity(sweep_data['start'])
                stop_qty = ureg.Quantity(sweep_data['stop'])
                num_points = int(sweep_data['num_points']) # Ensure integer

                if not start_qty.is_compatible_with("Hz"): raise pint.DimensionalityError(start_qty.units, ureg.Hz, msg="'start' frequency")
                if not stop_qty.is_compatible_with("Hz"): raise pint.DimensionalityError(stop_qty.units, ureg.Hz, msg="'stop' frequency")

                start_hz = start_qty.to(ureg.Hz).magnitude
                stop_hz = stop_qty.to(ureg.Hz).magnitude

                if start_hz <= 0: raise ValueError(f"Start frequency must be > 0 Hz for AC analysis. Got {start_hz} Hz.")
                if stop_hz <= 0: raise ValueError(f"Stop frequency must be > 0 Hz for AC analysis. Got {stop_hz} Hz.")
                if start_hz >= stop_hz: raise ValueError(f"Start frequency ({start_hz} Hz) must be less than stop frequency ({stop_hz} Hz).")
                if num_points < 2: raise ValueError(f"Number of points must be >= 2. Got {num_points}.")


                if sweep_type == 'linear':
                    freq_values_hz = np.linspace(start_hz, stop_hz, num_points, dtype=float)
                else: # log
                    freq_values_hz = np.geomspace(start_hz, stop_hz, num_points, dtype=float)

            elif sweep_type == 'list':
                 if 'points' not in sweep_data: raise ValueError("Missing required field 'points' for 'list' sweep.")
                 points_str = sweep_data['points']
                 freq_list_hz = []
                 for p_str in points_str:
                     qty = ureg.Quantity(p_str)
                     if not qty.is_compatible_with("Hz"): raise pint.DimensionalityError(qty.units, ureg.Hz, msg=f"frequency point '{p_str}'")
                     freq_hz = qty.to(ureg.Hz).magnitude
                     if freq_hz <= 0: raise ValueError(f"Frequency point '{p_str}' ({freq_hz} Hz) must be > 0 Hz for AC analysis.")
                     freq_list_hz.append(freq_hz)

                 if not freq_list_hz: raise ValueError("'points' list cannot be empty.")
                 # Sort and remove duplicates
                 freq_values_hz = np.array(sorted(list(set(freq_list_hz))), dtype=float)

        except pint.DimensionalityError as e:
            dim_str = getattr(e, 'dim1', '<unknown dim>')
            unit_str = getattr(e, 'units1', '<unknown unit>')
            field_name = getattr(e, 'msg', '<unknown field>')
            raise ValueError(f"Invalid frequency unit for {field_name} (units: {unit_str}, dim: {dim_str}): Expected Hertz compatible.") from e
        except ValueError as e: # Catch specific value errors raised above
            raise ValueError(f"Invalid sweep parameter: {e}") from e
        except Exception as e: # Catch other parsing errors
             raise ValueError(f"Error processing {sweep_type} sweep parameters: {e}") from e

        # Final check (redundant given checks above, but safe)
        if np.any(freq_values_hz <= 0):
             # This should ideally not be reachable due to checks above
             raise ValueError("Internal error: Sweep contains non-positive frequencies after parsing.")
        if len(freq_values_hz) == 0:
            raise ValueError("Sweep resulted in zero valid frequency points.")

        return freq_values_hz

    def _load_yaml(self, source: Union[str, Path, TextIO]) -> Dict[str, Any]:
        """Loads YAML from various source types."""
        try:
            if isinstance(source, Path):
                logger.debug(f"Loading YAML from path: {source}")
                if not source.is_file(): raise FileNotFoundError(f"YAML file not found or is not a file: {source}")
                with source.open('r') as f: return yaml.safe_load(f)
            elif isinstance(source, str):
                path = Path(source)
                if path.is_file(): # Check if it's an existing file path
                    logger.debug(f"Loading YAML from path string: {source}")
                    with path.open('r') as f: return yaml.safe_load(f)
                elif path.suffix.lower() in (".yaml", ".yml") and not path.exists():
                     raise FileNotFoundError(f"YAML file path specified but not found: {source}")
                else: # Treat as YAML content string
                    logger.debug("Loading YAML from string content.")
                    return yaml.safe_load(source)
            elif hasattr(source, 'read'): # File-like object
                 logger.debug("Loading YAML from file-like object.")
                 return yaml.safe_load(source)
            else:
                raise TypeError(f"Unsupported source type for YAML loading: {type(source)}")
        except FileNotFoundError: raise # Re-raise specific error
        except yaml.YAMLError as e: raise ParsingError(f"Invalid YAML syntax: {e}") from e
        except Exception as e: raise ParsingError(f"Failed to load or parse YAML source: {e}") from e
