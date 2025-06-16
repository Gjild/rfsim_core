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

    # Schema for R, L, C, etc. (standard components)
    _standard_component_schema = {
        'type': {'type': 'string', 'required': True, 
                 'allowed': ['Resistor', 'Capacitor', 'Inductor']}, # This list can be expanded later
        'id': {'type': 'string', 'required': True, 'empty': False},
        'ports': { # How the component instance connects to nets in the current circuit
            'type': 'dict', 
            'required': True, # For R,L,C, ports dict is required. Min port count checked by SemanticValidator.
            'minlength': 1, # At least one port connection must be specified if 'ports' key exists.
            'keysrules': {'type': ['string', 'integer'], 'empty': False}, # Port ID on the component itself (e.g., 0, 1 or "P1")
            'valuesrules': {'type': 'string', 'empty': False} # Net name in the current circuit
        },
        'parameters': { # Instance parameters for the component
            'type': 'dict', 
            'required': False, # Parameters dict itself is optional (component might use defaults)
            'keysrules': {'type': 'string', 'empty': False}, # Parameter name (e.g., "resistance")
            'valuesrules': _param_value_schema # Reference the shared _param_value_schema
        }
    }

    # Schema for Subcircuit component instances
    _subcircuit_component_schema = {
        'type': {'type': 'string', 'required': True, 'allowed': ['Subcircuit']},
        'id': {'type': 'string', 'required': True, 'empty': False},
        'definition_file': {'type': 'string', 'required': True, 'empty': False},
        'ports': { # Port mapping: { sub_ext_port_name_in_def: parent_circuit_net_name }
            'type': 'dict',
            'required': False, # The 'ports' key for mapping is optional for a subcircuit instance
                               # (e.g., if the subcircuit definition has no external ports).
            'minlength': 1,    # If 'ports' key IS present, it must map at least one port.
            'keysrules': {'type': 'string', 'empty': False}, # Subcircuit's external port name (from its definition)
            'valuesrules': {'type': 'string', 'empty': False} # Parent circuit's net name to connect to
        },
        'parameters': { # Parameter overrides for the subcircuit
            'type': 'dict',
            'required': False, # The 'parameters' key for overrides is optional.
            'keysrules': {'type': 'string', 'empty': False}, # Interface param name or relative FQN within subcircuit
            'valuesrules': _param_value_schema # Reference the shared _param_value_schema
        }
    }

    _schema = {
        'circuit_name': {'type': 'string', 'required': False, 'empty': False},
        'ground_net': {'type': 'string', 'required': False, 'empty': False},
        'parameters': {
            'type': 'dict', 'required': False,
            'keysrules': {'type': 'string', 'empty': False},
            'valuesrules': _param_value_schema # Global parameters
        },
        'components': {
            'type': 'list', 'required': True, 'minlength': 0, # 'components' key is required, can be empty list
            'schema': { # Schema for EACH item in the 'components' list
                'type': 'dict', # Each component instance definition must be a dictionary
                'oneof_schema': [ # The component dict must conform to ONE of these schemas
                    _standard_component_schema,   # Schema for standard R,L,C components
                    _subcircuit_component_schema  # Schema for Subcircuit instances
                ]
                # Cerberus will try to validate against each schema in 'oneof_schema'.
                # The strict 'allowed' constraint on the 'type' field within each sub-schema
                # (e.g., ['Resistor', 'Capacitor', 'Inductor'] vs ['Subcircuit'])
                # ensures that only one sub-schema can match a given component dictionary.
            }
        },
        'ports': { # This is for the top-level 'ports' section (external ports of the current circuit)
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
        self._validator = cerberus.Validator(self._schema) # Use the comprehensive class schema
        self._validator.allow_unknown = False # Disallow fields not defined in the schema
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

        structural_component_errors: List[str] = []
        unique_comp_ids: Set[str] = set()
        
        for comp_data_raw in validated_data.get('components', []):
            comp_id = comp_data_raw['id']
            comp_type = comp_data_raw['type']
            # For Subcircuits, comp_data_raw will also contain 'definition_file'
            # and its 'ports' field will be the mapping, 'parameters' the overrides.
            # For RLC, 'ports' field is connections, 'parameters' instance values.
            # The schema validation ensures comp_data_raw has the correct structure based on 'type'.
            
            # All components have 'parameters' (optional) and 'ports' (optional for subcircuit, required for RLC)
            # in their schema, but the meaning/structure of 'ports' differs.
            # The 'parameters' field is structurally the same (dict of string to param_value_schema).
            comp_params_raw = comp_data_raw.get('parameters', {}) 

            if comp_id in unique_comp_ids:
                structural_component_errors.append(f"Duplicate component ID '{comp_id}' found.")
                continue 
            unique_comp_ids.add(comp_id)

            # COMPONENT_REGISTRY check is now more of a semantic check,
            # but NetlistParser can still log if a type is completely unknown
            # to any schema variant (though 'allowed' in schemas should catch this).
            # If 'type' is 'Subcircuit', it's fine. If R,L,C, also fine.
            # If something else, it would fail the 'allowed' in both schemas.
            # So, no explicit COMPONENT_REGISTRY check is needed here for schema validation pass/fail.
            # CircuitBuilder will use COMPONENT_REGISTRY to instantiate.
            
            component = Component(instance_id=comp_id, component_type=comp_type, parameters=comp_params_raw)
            
            # Special handling for Subcircuit's 'definition_file'
            if comp_type == "Subcircuit":
                definition_file = comp_data_raw.get('definition_file') # Already validated as required by schema
                # The NetlistParser's primary role for subcircuits in this phase is schema validation. 
                # Subcircuit-specific details (like definition_file and port mappings from the instance's YAML ports block) 
                # are explicitly not stored within the Component dataclass's parameters or ports attributes. Instead, 
                # the raw dictionary representation (comp_data_raw) of the subcircuit instance is passed directly to the CircuitBuilder. 
                # The CircuitBuilder (in Phase 8 Task 3) will then directly extract and interpret these details from comp_data_raw, 
                # ensuring the Component dataclass's parameters attribute remains dedicated solely to simulation parameters (i.e., parameter overrides).
                pass # No specific action for Subcircuit definition_file in parser beyond schema validation.

            has_port_structure_error_for_this_comp = False
            
            # The 'ports' field in comp_data_raw has different meanings for RLC vs Subcircuit.
            # For RLC, it's {comp_port_id: net_name}.
            # For Subcircuit, it's {sub_ext_port_name: parent_net_name}.
            # The parser should primarily create Port objects for RLC style connections.
            # Subcircuit port mapping is handled by CircuitBuilder.
            if comp_type != "Subcircuit":
                # This is for R, L, C components
                # The schema ensures `comp_data_raw['ports']` exists and matches RLC port style if type is R,L,C.
                yaml_ports_for_rlc = comp_data_raw.get('ports', {}) # Should be present due to schema if RLC
                for port_id, net_name_str_from_yaml in yaml_ports_for_rlc.items():
                    try:
                        port = component.add_port(port_id) 
                        port.original_yaml_net_name = net_name_str_from_yaml 
                        net = circuit.get_or_create_net(net_name_str_from_yaml)
                        if port.net is not None: 
                                raise ValueError(f"Internal error: Port '{port_id}' on '{comp_id}' already connected or port object reused.")
                        port.net = net
                        if component not in net.connected_components:
                            net.connected_components.append(component)
                        logger.debug(f"Connected port '{port_id}' of '{comp_id}' (type {comp_type}) to net '{net_name_str_from_yaml}'.")
                    except Exception as e:
                        err_msg = f"Error processing RLC-style port structure (port='{port_id}', net='{net_name_str_from_yaml}') for component '{comp_id}': {e}"
                        structural_component_errors.append(err_msg)
                        logger.error(err_msg)
                        has_port_structure_error_for_this_comp = True
            # else if comp_type == "Subcircuit":
                # The `comp_data_raw['ports']` is the mapping dict.
                # The `Component` object for a Subcircuit instance does not have `Port` objects in its `ports` dict
                # in the same way an RLC component does. This is handled by CircuitBuilder.
                # The `Component` dataclass's `ports` field will be empty for a SubcircuitInstance.
                # SemanticValidator for SubcircuitInstance will check its `port_mapping_yaml` from `comp_data_raw`.

            if not has_port_structure_error_for_this_comp:
                    circuit.add_component(component)
        
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
                if len(freq_array_hz) > 0: # Add check for empty array before accessing elements
                    logger.info(f"Parsed frequency sweep: {len(freq_array_hz)} points from {freq_array_hz[0]:.3e} Hz to {freq_array_hz[-1]:.3e} Hz.")
                else: # Handle case of empty frequency array from sweep parsing
                    logger.info("Parsed frequency sweep: 0 points.")
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
                if start_hz > stop_hz: raise ValueError(f"Start frequency ({start_hz} Hz) must not be greater than stop frequency ({stop_hz} Hz).")
                if num_points < 1 : raise ValueError(f"Number of points must be >= 1. Got {num_points}.")
                if num_points == 1 and start_hz != stop_hz: raise ValueError(f"For num_points=1, start frequency ({start_hz} Hz) must be equal to stop frequency ({stop_hz} Hz).")
                if num_points > 1 and start_hz == stop_hz : logger.warning(f"Sweep from {start_hz} to {stop_hz} with {num_points} points will result in identical frequency points.")


                if sweep_type == 'linear':
                    freq_values_hz = np.linspace(start_hz, stop_hz, num_points, dtype=float)
                else: # log
                    if start_hz == stop_hz : # geomspace needs distinct start/stop unless num_points=1
                        if num_points == 1:
                             freq_values_hz = np.array([start_hz], dtype=float)
                        else: # Revert to linspace behavior for safety if start=stop but num_points > 1 for log
                             logger.warning(f"Log sweep with start=stop={start_hz} and num_points={num_points}>1. Using linspace behavior (all points will be {start_hz}).")
                             freq_values_hz = np.full(num_points, start_hz, dtype=float)
                    else:
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
                    # Remove duplicates and sort for canonical order, though MNA should handle any order.
                    freq_values_hz = np.array(sorted(list(set(freq_list_hz))), dtype=float)

        except pint.DimensionalityError as e:
            dim_str = getattr(e, 'dim1', '<unknown dim>')
            unit_str = getattr(e, 'units1', '<unknown unit>')
            field_name = getattr(e, 'extra_msg', '<unknown field>') # Corrected from 'msg' based on Pint source for extra_msg
            raise ValueError(f"Invalid frequency unit for {field_name} (units: {unit_str}, dim: {dim_str}): Expected Hertz compatible.") from e
        except ValueError as e: 
            raise ValueError(f"Invalid sweep parameter: {e}") from e
        except Exception as e: 
                raise ValueError(f"Error processing {sweep_type} sweep parameters: {e}") from e

        if freq_values_hz.size > 0 and np.any(freq_values_hz <= 0): # Check only if array is not empty
                raise ValueError("Internal error: Sweep contains non-positive frequencies after parsing.")
        if len(freq_values_hz) == 0 and sweep_data['type'] != 'list' and sweep_data.get('num_points',0) > 0 : # List can be empty if input is empty. Start/stop/num should produce points.
            raise ValueError("Sweep resulted in zero valid frequency points, despite num_points > 0.")
        elif len(freq_values_hz) == 0 and sweep_data['type'] == 'list' and not sweep_data.get('points'): # Empty list explicitly provided
            pass # Empty sweep is fine if explicitly an empty list of points
        elif len(freq_values_hz) == 0: # General catch-all for unexpected empty result
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