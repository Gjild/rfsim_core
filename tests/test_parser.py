# tests/test_parser.py
import pytest
import yaml
from pathlib import Path
from io import StringIO

# Make sure Pint Quantity comparison works well in tests
# You might need to adjust precision settings if necessary
from pint import UnitRegistry
# from rfsim_core.units import ureg # Use the same registry as the main code
# For isolated testing, maybe create a fresh one
ureg = UnitRegistry()
Quantity = ureg.Quantity

from rfsim_core import NetlistParser, SchemaValidationError, ParsingError
from rfsim_core import ParameterError
from rfsim_core import Circuit, Component, Net




# --- Test Cases ---

def test_valid_yaml_string_parsing(parser, valid_yaml_string):
    """Test parsing a valid YAML string."""
    circuit = parser.parse(valid_yaml_string)

    assert isinstance(circuit, Circuit)
    assert circuit.name == "Test RLC Circuit"
    assert circuit.ground_net_name == "gnd"

    # Check global parameters
    assert circuit.parameter_manager is not None
    assert circuit.parameter_manager.get_parameter("supply_voltage") == Quantity("5 V")
    assert circuit.parameter_manager.get_parameter("default_cap") == Quantity("10 pF")
    with pytest.raises(KeyError):
        circuit.parameter_manager.get_parameter("non_existent")

    # Check components
    assert len(circuit.components) == 3
    assert "R1" in circuit.components
    assert "C1" in circuit.components
    assert "L1" in circuit.components

    r1 = circuit.components["R1"]
    assert r1.component_type == "Resistor"
    assert r1.parameters == {"resistance": "1kohm"} # Check raw param storage
    assert list(r1.ports.keys()) == ["p1", "p2"]
    assert r1.ports["p1"].net.name == "net1"
    assert r1.ports["p2"].net.name == "gnd"
    assert r1.ports["p2"].net.is_ground

    l1 = circuit.components["L1"]
    assert l1.ports["p2"].net.name == "out"

    # Check nets
    assert len(circuit.nets) == 3 # net1, gnd, out
    assert "net1" in circuit.nets
    assert "gnd" in circuit.nets
    assert "out" in circuit.nets
    assert circuit.nets["gnd"].is_ground
    assert circuit.nets["out"].is_external

    # Check external ports
    assert len(circuit.external_ports) == 1
    assert "out" in circuit.external_ports
    assert circuit.external_ports["out"].name == "out"
    assert circuit.external_port_impedances["out"] == "50 ohm"

    # Check connections reflected in nets
    assert r1 in circuit.nets["net1"].connected_components
    assert circuit.components["C1"] in circuit.nets["net1"].connected_components
    assert l1 in circuit.nets["net1"].connected_components
    assert l1 in circuit.nets["out"].connected_components


def test_valid_yaml_file_parsing(parser, valid_yaml_string, tmp_path):
    """Test parsing a valid YAML file."""
    p = tmp_path / "test_circuit.yaml"
    p.write_text(valid_yaml_string)
    circuit = parser.parse(p) # Pass Path object
    assert isinstance(circuit, Circuit)
    assert circuit.name == "Test RLC Circuit"
    assert "R1" in circuit.components

    circuit_from_str_path = parser.parse(str(p)) # Pass path as string
    assert isinstance(circuit_from_str_path, Circuit)
    assert circuit_from_str_path.name == "Test RLC Circuit"


def test_valid_yaml_stream_parsing(parser, valid_yaml_string):
    """Test parsing a valid YAML stream."""
    stream = StringIO(valid_yaml_string)
    circuit = parser.parse(stream)
    assert isinstance(circuit, Circuit)
    assert circuit.name == "Test RLC Circuit"


def test_missing_components_section(parser):
     # Schema requires 'components', let's make it optional for robustness test
     # Temporarily modify schema for test or adjust schema if empty components allowed
     original_schema = parser._validator.schema.copy()
     parser._validator.schema['components']['required'] = False
     try:
         circuit = parser.parse("ports: [{id: p1, reference_impedance: '50ohm'}]")
         assert len(circuit.components) == 0
         assert "p1" in circuit.external_ports
     finally:
          parser._validator.schema = original_schema # Restore original schema


def test_invalid_schema_missing_comp_id(parser, yaml_bad_schema_comp):
    """Test YAML with missing required component ID."""
    with pytest.raises(SchemaValidationError) as excinfo:
        parser.parse(yaml_bad_schema_comp)
    assert "'id': ['required field']" in str(excinfo.value.errors)


def test_invalid_schema_missing_port_impedance(parser, yaml_bad_schema_port):
    """Test YAML with missing required port impedance."""
    with pytest.raises(SchemaValidationError) as excinfo:
        parser.parse(yaml_bad_schema_port)
    assert "'reference_impedance': ['required field']" in str(excinfo.value.errors)


def test_invalid_global_parameter_unit(parser, yaml_bad_global_param_unit):
    """Test YAML with an invalid unit string for a global parameter."""
    with pytest.raises(ParameterError) as excinfo:
        parser.parse(yaml_bad_global_param_unit)
    assert "Error parsing global parameter 'bad_param'" in str(excinfo.value)
    assert "'foobars'" in str(excinfo.value) # Check unit name included


def test_duplicate_component_id(parser, yaml_duplicate_comp_id):
    """Test YAML with duplicate component IDs."""
    with pytest.raises(ParsingError) as excinfo:
        parser.parse(yaml_duplicate_comp_id)
    assert "Duplicate component ID 'R1'" in str(excinfo.value)


def test_malformed_yaml(parser, yaml_malformed):
    """Test parsing syntactically incorrect YAML."""
    with pytest.raises(ParsingError) as excinfo: # Wraps yaml.YAMLError
        parser.parse(yaml_malformed)
    assert "Invalid YAML syntax" in str(excinfo.value)


def test_non_existent_file(parser):
    """Test parsing a non-existent file path."""
    with pytest.raises(FileNotFoundError):
        parser.parse("non_existent_file.yaml")