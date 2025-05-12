# --- src/rfsim_core/parser.py ---
import logging
import yaml
import cerberus
import pint
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union, TextIO, Tuple, Set, List 

from .data_structures import Circuit, Component, Net, Port
from .parameters import ParameterManager, ParameterError, ParameterDefinitionError 
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
    Builds an internal Circuit representation containing raw definitions
    and extracts sweep frequencies. It does *not* build the ParameterManager.
    Minimal semantic checks are performed here; most are deferred to SemanticValidator.
    """
    _param_value_schema = {
        'oneof': [
            {'type': ['string', 'number']},
            {
                'type': 'dict',
                'schema': {
                    'expression': {'type': 'string', 'required': True, 'empty': False},
                    'dimension': {'type': 'string', 'required': True, 'empty': False},
                },
                'required': True 
            }
        ]
    }
    _schema = {
        'circuit_name': {'type': 'string', 'required': False, 'empty': False},
        'ground_net': {'type': 'string', 'required': False, 'empty': False},
        'parameters': {
            'type': 'dict', 'required': False,
            'keysrules': {'type': 'string', 'empty': False},
            'valuesrules': _param_value_schema
        },
        'components': {
            'type': 'list', 'required': True, 'minlength': 0,
            'schema': {
                'type': 'dict',
                'schema': {
                    'type': {'type': 'string', 'required': True, 'empty': False},
                    'id': {'type': 'string', 'required': True, 'empty': False},
                    'ports': {
                        'type': 'dict', 'required': True, 'minlength': 1,
                        'keysrules': {'type': ['string', 'integer'], 'empty': False}, 
                        'valuesrules': {'type': 'string', 'empty': False}
                    },
                    'parameters': {
                        'type': 'dict', 'required': False,
                        'keysrules': {'type': 'string', 'empty': False},
                        'valuesrules': _param_value_schema 
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
        logger.info(f"Starting parsing of netlist source: {type(source)}")
        yaml_content = self._load_yaml(source)
        if not self._validator.validate(yaml_content):
            logger.error(f"Schema validation failed. Errors: {self._validator.errors}")
            raise SchemaValidationError(self._validator.errors)

        logger.info("YAML schema validation successful.")
        validated_data = self._validator.document

        circuit_name = validated_data.get('circuit_name', 'UnnamedCircuit')
        ground_name = validated_data.get('ground_net', 'gnd')
        circuit = Circuit(name=circuit_name, ground_net_name=ground_name)

        raw_global_params_data = validated_data.get('parameters', {})
        setattr(circuit, 'raw_global_parameters', raw_global_params_data)
        logger.debug(f"Stored raw global parameters data: {raw_global_params_data}")

        # Accumulates errors that are critical for parsing structure (e.g., duplicate IDs)
        # Unknown component types are no longer added here to cause a ParsingError.
        structural_component_errors: List[str] = []
        unique_comp_ids: Set[str] = set()
        
        for comp_data_raw in validated_data.get('components', []):
            comp_id = comp_data_raw['id']
            comp_type = comp_data_raw['type']
            comp_params_raw = comp_data_raw.get('parameters', {})

            if comp_id in unique_comp_ids:
                structural_component_errors.append(f"Duplicate component ID '{comp_id}' found.")
                # This component instance is problematic; skip further processing for it.
                # The error will be raised later if structural_component_errors is non-empty.
                continue 
            unique_comp_ids.add(comp_id)

            if comp_type not in COMPONENT_REGISTRY:
                # Log for debugging, but do not add to structural_component_errors.
                # SemanticValidator will report this as COMP_TYPE_001.
                # The raw Component object will still be created and added to the circuit.
                logger.debug(f"NetlistParser: Encountered unknown component type '{comp_type}' for instance '{comp_id}'. Raw component data will be created.")
            
            # Create the raw Component data structure
            component = Component(instance_id=comp_id, component_type=comp_type, parameters=comp_params_raw)
            has_port_structure_error_for_this_comp = False

            for port_id, net_name_str_from_yaml in comp_data_raw['ports'].items():
                try:
                    # This can raise ValueError for duplicate port_id on this specific component
                    port = component.add_port(port_id) 
                    port.original_yaml_net_name = net_name_str_from_yaml 

                    net = circuit.get_or_create_net(net_name_str_from_yaml)

                    if port.net is not None: 
                         # This should be caught by component.add_port if port_id is duplicate,
                         # or indicates an internal logic error if port.net was somehow pre-assigned.
                         raise ValueError(f"Internal error: Port '{port_id}' on '{comp_id}' already connected or port object reused.")
                    port.net = net
                    if component not in net.connected_components:
                        net.connected_components.append(component)
                    
                    logger.debug(f"Connected port '{port_id}' of '{comp_id}' to net '{net_name_str_from_yaml}'.")

                except Exception as e: # Catch errors from component.add_port or other unexpected issues here
                    err_msg = f"Error processing port structure (port='{port_id}', net='{net_name_str_from_yaml}') for component '{comp_id}': {e}"
                    structural_component_errors.append(err_msg)
                    logger.error(err_msg)
                    has_port_structure_error_for_this_comp = True
            
            # Add component to circuit only if no structural errors occurred *for this component's ports*
            if not has_port_structure_error_for_this_comp:
                 circuit.add_component(component)
            # If has_port_structure_error_for_this_comp is true, this component instance is not added,
            # and the error is in structural_component_errors.

        # Raise ParsingError if any critical structural errors were found
        if structural_component_errors:
            raise ParsingError("Errors occurred during component structure processing:\n- " + "\n- ".join(structural_component_errors))

        port_errors = []
        external_port_names = set()
        for port_data in validated_data.get('ports', []):
            port_id = port_data['id'] 
            impedance_str = port_data['reference_impedance']

            if port_id in external_port_names:
                    port_errors.append(f"Duplicate external port definition for net '{port_id}'.")
                    continue
            external_port_names.add(port_id)
            
            try:
                circuit.set_external_port(port_id, impedance_str)
                logger.debug(f"Registered external port '{port_id}' with Z0='{impedance_str}'.")

            except Exception as e:
                err_msg = f"Error setting external port '{port_id}' in circuit object: {e}"
                logger.error(err_msg)
                port_errors.append(err_msg)

        if port_errors:
            raise ParsingError("Errors occurred during external port processing:\n- " + "\n- ".join(port_errors))

        sweep_data = validated_data.get('sweep')
        try:
             freq_array_hz = self._parse_sweep(sweep_data)
             logger.info(f"Parsed frequency sweep: {len(freq_array_hz)} points from {freq_array_hz[0]:.3e} Hz to {freq_array_hz[-1]:.3e} Hz.")
        except (ValueError, pint.errors.UndefinedUnitError, pint.errors.DimensionalityError) as e:
             raise ParsingError(f"Failed to parse sweep configuration: {e}") from e
        
        logger.info(f"Successfully parsed netlist structure '{circuit.name}'.")
        logger.info(f"Found {len(circuit.components)} components, {len(circuit.nets)} nets ({len(circuit.external_ports)} external).")

        return circuit, freq_array_hz
    
    def _parse_sweep(self, sweep_data: Dict[str, Any]) -> np.ndarray:
        """Parses the validated sweep dictionary and returns frequency array in Hz (> 0)."""
        sweep_type = sweep_data['type']
        logger.info(f"Parsing '{sweep_type}' sweep.")
        freq_values_hz = np.array([], dtype=float) 

        try:
            if sweep_type == 'linear' or sweep_type == 'log':
                required = ['start', 'stop', 'num_points']
                if not all(k in sweep_data for k in required):
                     raise ValueError(f"Missing required fields {required} for '{sweep_type}' sweep.")

                start_qty = ureg.Quantity(sweep_data['start'])
                stop_qty = ureg.Quantity(sweep_data['stop'])
                num_points = int(sweep_data['num_points']) 

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
                 freq_values_hz = np.array(sorted(list(set(freq_list_hz))), dtype=float)

        except pint.DimensionalityError as e:
            dim_str = getattr(e, 'dim1', '<unknown dim>')
            unit_str = getattr(e, 'units1', '<unknown unit>')
            field_name = getattr(e, 'msg', '<unknown field>')
            raise ValueError(f"Invalid frequency unit for {field_name} (units: {unit_str}, dim: {dim_str}): Expected Hertz compatible.") from e
        except ValueError as e: 
            raise ValueError(f"Invalid sweep parameter: {e}") from e
        except Exception as e: 
             raise ValueError(f"Error processing {sweep_type} sweep parameters: {e}") from e

        if np.any(freq_values_hz <= 0):
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
                if path.is_file(): 
                    logger.debug(f"Loading YAML from path string: {source}")
                    with path.open('r') as f: return yaml.safe_load(f)
                elif path.suffix.lower() in (".yaml", ".yml") and not path.exists():
                     raise FileNotFoundError(f"YAML file path specified but not found: {source}")
                else: 
                    logger.debug("Loading YAML from string content.")
                    return yaml.safe_load(source)
            elif hasattr(source, 'read'): 
                 logger.debug("Loading YAML from file-like object.")
                 return yaml.safe_load(source)
            else:
                raise TypeError(f"Unsupported source type for YAML loading: {type(source)}")
        except FileNotFoundError: raise 
        except yaml.YAMLError as e: raise ParsingError(f"Invalid YAML syntax: {e}") from e
        except Exception as e: raise ParsingError(f"Failed to load or parse YAML source: {e}") from e