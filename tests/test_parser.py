import pytest
import numpy as np
import yaml
from pathlib import Path
import logging # Import logging for caplog level setting

from rfsim_core.parser import NetlistParser, SchemaValidationError, ParsingError
from rfsim_core.data_structures import Circuit, Component, Net
from rfsim_core.units import ureg

# --- Fixtures for YAML Content ---

@pytest.fixture
def basic_rlc_yaml_constants():
    """Valid YAML with only constant parameters."""
    # FIX: Use strings for ground net and net names '0'
    return """
circuit_name: Basic RLC
ground_net: "0" # Use string "0"
parameters:
  global_R: 1e3 ohm
  global_L: 1 uH
components:
  - type: Resistor
    id: R1
    ports: {0: N1, 1: "0"} # Use string "0" for net name
    parameters:
      resistance: 50 ohm
  - type: Capacitor
    id: C1
    ports: {0: N1, 1: N2}
    parameters:
      capacitance: 10 pF
  - type: Inductor
    id: L1
    ports: {0: N2, 1: "0"} # Use string "0" for net name
    parameters:
      inductance: global_L # Reference global constant
ports:
  - id: N1
    reference_impedance: "50 ohm" # Ensure string
sweep:
  type: linear
  start: 1 GHz
  stop: 10 GHz
  num_points: 11
"""

@pytest.fixture
def yaml_with_global_expr():
    """Valid YAML including a global expression parameter."""
    return """
circuit_name: Global Expr Circuit
ground_net: gnd
parameters:
  R_base: 100 ohm
  L_val: 10 nH
  gain:
    expression: 'sqrt(freq / 1e9)' # dimensionless gain vs GHz
    dimension: 'dimensionless'     # MUST be provided
components:
  - type: Resistor
    id: R1
    ports: {p1: N1, p2: gnd}
    parameters:
      resistance: R_base
ports:
  - id: N1
    reference_impedance: "50 ohm" # Ensure string
sweep:
  type: list
  points: ['1 GHz', '5 GHz']
"""

@pytest.fixture
def yaml_with_instance_expr():
    """Valid YAML including an instance parameter defined as an expression."""
    return """
circuit_name: Instance Expr Circuit
ground_net: gnd
parameters:
  R_load: 50 ohm
components:
  - type: Resistor
    id: R1
    ports: {0: N1, 1: gnd}
    parameters:
      # Instance parameter defined by expression
      resistance:
        expression: 'R_load * 2'
        dimension: 'ohm' # Present but ignored by builder
  - type: Resistor
    id: R2
    ports: {0: N1, 1: gnd}
    parameters:
      resistance: R_load # Instance param referencing global
ports:
  - id: N1
    reference_impedance: "50 ohm" # Ensure string
sweep:
  type: log
  start: 1 MHz
  stop: 1 GHz
  num_points: 3
"""

@pytest.fixture
def yaml_invalid_schema_missing_dimension():
    """Invalid YAML - global expression missing 'dimension'."""
    return """
parameters:
  bad_gain:
    expression: 'freq / 1e9'
    # dimension: ... MISSING
components: [] # Minimal components to pass basic checks
sweep: { type: list, points: ['1 Hz'] }
"""

@pytest.fixture
def yaml_invalid_connectivity_floating():
    """YAML that should produce a floating net warning."""
    return """
components:
  - type: Resistor
    id: R1
    ports: {0: N1, 1: N_FLOATING} # N_FLOATING only connected once
    parameters: {resistance: 1k}
  - type: Resistor
    id: R2
    ports: {0: N1, 1: gnd}
    parameters: {resistance: 1k}
ports:
  - id: N1
    reference_impedance: "50 ohm" # Ensure string
sweep: { type: list, points: ['1 Hz'] }
"""

# --- Test Class ---

class TestNetlistParser:

    def test_parse_valid_basic_constants(self, basic_rlc_yaml_constants):
        """Test parsing a valid netlist with constant parameters."""
        parser = NetlistParser()
        circuit, freq_array = parser.parse(basic_rlc_yaml_constants)

        assert isinstance(circuit, Circuit)
        assert circuit.name == "Basic RLC"
        assert circuit.ground_net_name == "0" # Check string ground name
        assert "R1" in circuit.components
        assert "C1" in circuit.components
        assert "L1" in circuit.components
        assert "N1" in circuit.external_ports
        assert circuit.external_port_impedances["N1"] == "50 ohm"

        raw_globals = getattr(circuit, 'raw_global_parameters', None)
        assert raw_globals is not None
        assert raw_globals == {"global_R": "1e3 ohm", "global_L": "1 uH"}

        assert circuit.components["R1"].parameters == {"resistance": "50 ohm"}
        assert circuit.components["C1"].parameters == {"capacitance": "10 pF"}
        assert circuit.components["L1"].parameters == {"inductance": "global_L"}

        assert isinstance(freq_array, np.ndarray)
        assert len(freq_array) == 11
        assert np.isclose(freq_array[0], 1e9)
        assert np.isclose(freq_array[-1], 10e9)
        assert np.all(freq_array > 0)

    def test_parse_valid_global_expression(self, yaml_with_global_expr):
        """Test parsing with a global expression parameter."""
        parser = NetlistParser()
        circuit, freq_array = parser.parse(yaml_with_global_expr)

        assert isinstance(circuit, Circuit)
        assert circuit.name == "Global Expr Circuit"
        assert "R1" in circuit.components

        raw_globals = getattr(circuit, 'raw_global_parameters')
        assert raw_globals == {
            "R_base": "100 ohm",
            "L_val": "10 nH",
            "gain": {
                "expression": 'sqrt(freq / 1e9)',
                "dimension": 'dimensionless'
            }
        }
        assert circuit.components["R1"].parameters == {"resistance": "R_base"}

        assert len(freq_array) == 2
        assert np.allclose(freq_array, [1e9, 5e9])

    def test_parse_valid_instance_expression(self, yaml_with_instance_expr):
        """Test parsing with an instance parameter defined as an expression."""
        parser = NetlistParser()
        circuit, freq_array = parser.parse(yaml_with_instance_expr)

        assert isinstance(circuit, Circuit)
        assert "R1" in circuit.components
        assert "R2" in circuit.components

        assert circuit.components["R1"].parameters == {
            "resistance": {
                "expression": 'R_load * 2',
                "dimension": 'ohm'
            }
        }
        assert circuit.components["R2"].parameters == {"resistance": "R_load"}

        raw_globals = getattr(circuit, 'raw_global_parameters')
        assert raw_globals == {"R_load": "50 ohm"}

        assert len(freq_array) == 3
        assert np.all(freq_array > 0)

    def test_parse_invalid_schema_missing_dimension(self, yaml_invalid_schema_missing_dimension):
        """Test schema validation fails for global expression without dimension."""
        parser = NetlistParser()
        with pytest.raises(SchemaValidationError) as excinfo:
            parser.parse(yaml_invalid_schema_missing_dimension)
        # FIX: Simplified check on the error string representation
        error_string = str(excinfo.value.errors)
        assert "'parameters'" in error_string
        assert "'bad_gain'" in error_string
        # Check for the specific error detail Cerberus provides
        assert "'dimension': ['required field']" in error_string

    def test_parse_invalid_schema_bad_param_value_structure(self):
        """Test schema validation fails for invalid parameter value structure."""
        parser = NetlistParser()
        bad_yaml = """
parameters:
  bad_param: [1, 2, 3] # Not allowed structure
components: []
sweep: { type: list, points: ['1 Hz'] }
"""
        with pytest.raises(SchemaValidationError) as excinfo:
            parser.parse(bad_yaml)
        assert "'parameters'" in str(excinfo.value.errors)
        assert "bad_param" in str(excinfo.value.errors['parameters'][0])

    # FIX: Add caplog fixture to arguments
    def test_parse_invalid_connectivity_floating(self, yaml_invalid_connectivity_floating, caplog):
        """Test semantic validation catches floating internal nets (as warning)."""
        parser = NetlistParser()
        with caplog.at_level(logging.WARNING): # Ensure logging level is captured
            circuit, _ = parser.parse(yaml_invalid_connectivity_floating)
        assert "Internal net 'N_FLOATING' is only connected to one component port" in caplog.text
        assert "warnings found" in caplog.text

    def test_parse_invalid_connectivity_unconnected_port(self):
        """Test semantic validation catches unconnected external ports."""
        parser = NetlistParser()
        bad_yaml = """
components:
  - type: Resistor
    id: R1
    ports: {0: N1, 1: gnd}
    parameters: {resistance: 1k}
ports:
  - id: N1 # N1 is connected
    reference_impedance: "50 ohm" # String
  - id: N2 # N2 is defined as external but not connected to anything
    reference_impedance: "50 ohm" # String
sweep: { type: list, points: ['1 Hz'] }
"""
        with pytest.raises(ParsingError) as excinfo:
            parser.parse(bad_yaml)

        # FIX: Update assertion to match the actual error message
        expected_error_fragment = "External port 'N2' is defined but the net name was never used by any component"
        assert expected_error_fragment in str(excinfo.value)

    def test_parse_from_file(self, tmp_path, basic_rlc_yaml_constants):
        """Test parsing from a temporary file."""
        p = tmp_path / "test_netlist.yaml"
        p.write_text(basic_rlc_yaml_constants)
        parser = NetlistParser()
        circuit, freq_array = parser.parse(p)
        assert isinstance(circuit, Circuit)
        assert circuit.name == "Basic RLC"
        assert len(freq_array) == 11

    def test_parse_from_invalid_path(self):
        """Test parsing from a non-existent file path."""
        parser = NetlistParser()
        with pytest.raises(FileNotFoundError):
            parser.parse("non_existent_netlist.yaml")