# tests/test_parser.py
import pytest
import yaml
import pint
import numpy as np
from pathlib import Path
from io import StringIO

# Use the application's unit registry if defined globally, otherwise create one
try:
    from rfsim_core import ureg, Quantity
except ImportError:
    ureg = pint.UnitRegistry()
    Quantity = ureg.Quantity

# Import necessary classes and exceptions from your project
from rfsim_core import (
    NetlistParser,
    SchemaValidationError,
    ParsingError,
    ParameterError,
    Circuit,
    Component,
    Net,
)

# --- Helper Functions ---
def assert_quantity_close(q1, q2, rtol=1e-5, atol=1e-8, **kwargs):
    """
    Asserts that two Pint quantities are close in magnitude (using numpy.allclose)
    and have the same dimensionality. Handles array quantities.
    """
    assert isinstance(q1, Quantity), f"q1 is not a Quantity (type: {type(q1)})"
    assert isinstance(q2, Quantity), f"q2 is not a Quantity (type: {type(q2)})"
    assert q1.dimensionality == q2.dimensionality, \
        f"Dimensionality mismatch: {q1.dimensionality} vs {q2.dimensionality}"

    q2_converted = q2.to(q1.units)
    np.testing.assert_allclose(q1.magnitude, q2_converted.magnitude, rtol=rtol, atol=atol, **kwargs)

# --- Fixtures ---
@pytest.fixture
def parser():
    """Provides a NetlistParser instance."""
    return NetlistParser()

# Updated: Added a minimal valid sweep section
@pytest.fixture
def valid_yaml_string():
    """Provides a basic valid YAML string including a sweep."""
    return """
circuit_name: Test RLC Circuit
sweep:
  type: list
  points: ['1 GHz'] # Minimal sweep
parameters:
  supply_voltage: '5 V'
  default_cap: '10 pF'
components:
  - type: Resistor
    id: R1
    ports: { p1: net1, p2: gnd }
    parameters:
      resistance: '1kohm' # Component params are strings for now
  - type: Capacitor
    id: C1
    ports: { p1: net1, p2: gnd }
    parameters:
      capacitance: default_cap # Raw string, resolution later
  - type: Inductor
    id: L1
    ports: { p1: net1, p2: 'out' } # Use quotes for net name 'out'
    parameters:
      inductance: '1 uH'
ports:
  - id: out
    reference_impedance: '50 ohm'
"""

# Keep fixtures for specific error conditions, add sweep if needed to pass schema
@pytest.fixture
def yaml_missing_components():
    # Needs sweep to be valid schema-wise
    return """
sweep: { type: list, points: ['100 MHz'] }
ports:
  - id: out
    reference_impedance: '50 ohm'
# components: section is missing (will test optional components later)
"""

@pytest.fixture
def yaml_bad_schema_comp():
    # Needs sweep
    return """
sweep: { type: list, points: ['100 MHz'] }
components:
  - type: Resistor
    # id is missing
    ports: { p1: n1, p2: gnd }
"""

@pytest.fixture
def yaml_bad_schema_port():
    # Needs sweep
    return """
sweep: { type: list, points: ['100 MHz'] }
components:
  - type: Resistor
    id: R1
    ports: { p1: n1, p2: gnd }
ports:
  - id: p1
    # reference_impedance is missing
"""

@pytest.fixture
def yaml_bad_global_param_unit():
    # Needs sweep
     return """
sweep: { type: list, points: ['100 MHz'] }
parameters:
  bad_param: '10 foobars'
components:
  - type: Resistor
    id: R1
    ports: { p1: n1, p2: gnd }
"""

@pytest.fixture
def yaml_duplicate_comp_id():
    # Needs sweep
    return """
sweep: { type: list, points: ['100 MHz'] }
components:
  - type: Resistor
    id: R1
    ports: { p1: n1, p2: gnd }
  - type: Capacitor
    id: R1 # Duplicate ID
    ports: { p1: n1, p2: gnd }
"""

@pytest.fixture
def yaml_malformed():
    # Syntax error occurs before schema check
    return "sweep: {type: list, points: ['1GHz']}\ncomponents: \n- type: R\n id: R1\n ports: {p1: n1}" # Indentation error

@pytest.fixture
def valid_yaml_with_linear_sweep():
    return """
circuit_name: Swept Circuit Linear
sweep:
  type: linear
  start: '1 MHz'
  stop: '100 MHz'
  num_points: 10
components:
  - type: Resistor
    id: R1
    ports: {p1: P1, p2: gnd}
    parameters: {resistance: '50 ohm'}
ports:
  - {id: P1, reference_impedance: '50 ohm'}
"""

@pytest.fixture
def yaml_log_sweep():
     return """
sweep: { type: log, start: '1kHz', stop: '1 GHz', num_points: 3 }
components:
  - type: Resistor
    id: R1
    ports: # Use standard mapping
      p1: P1
      p2: gnd
    parameters: {resistance: '1k'}
ports: [{id: P1, reference_impedance: '50'}]
"""

@pytest.fixture
def yaml_list_sweep():
     return """
sweep: { type: list, points: ['10 MHz', '20 MHz', '10 MHz'] }
components:
  - type: Resistor
    id: R1
    ports: # Use standard mapping
      p1: P1
      p2: gnd
    parameters: {resistance: '1k'}
ports: [{id: P1, reference_impedance: '50'}]
"""

@pytest.fixture
def yaml_sweep_missing_stop():
     return """
sweep: { type: linear, start: '1MHz', num_points: 10 } # Missing stop
components:
  - type: Resistor
    id: R1
    ports: # Use standard mapping
      p1: P1
      p2: gnd
    parameters: {resistance: '1k'}
ports: [{id: P1, reference_impedance: '50'}]
"""

@pytest.fixture
def yaml_sweep_bad_unit():
     return """
sweep: { type: linear, start: '1 MV', stop: '10 MV', num_points: 10 }
components:
  - type: Resistor
    id: R1
    ports: # Use standard mapping
      p1: P1
      p2: gnd
    parameters: {resistance: '1k'}
ports: [{id: P1, reference_impedance: '50'}]
"""

@pytest.fixture
def yaml_sweep_log_zero_start():
     return """
sweep: { type: log, start: '0 Hz', stop: '1 GHz', num_points: 3 }
components:
  - type: Resistor
    id: R1
    ports: # Use standard mapping
      p1: P1
      p2: gnd
    parameters: {resistance: '1k'}
ports: [{id: P1, reference_impedance: '50'}]
"""

@pytest.fixture
def yaml_sweep_invalid_range():
     return """
sweep: { type: linear, start: '100 MHz', stop: '1 MHz', num_points: 3 } # Start > Stop
components:
  - type: Resistor
    id: R1
    ports: # Use standard mapping
      p1: P1
      p2: gnd
    parameters: {resistance: '1k'}
ports: [{id: P1, reference_impedance: '50'}]
"""

# --- Core Parsing Tests (Adapted) ---

def test_valid_yaml_string_parsing(parser, valid_yaml_string):
    """Test parsing a valid YAML string (now includes sweep)."""
    circuit = parser.parse(valid_yaml_string)

    # Basic circuit checks
    assert isinstance(circuit, Circuit)
    assert circuit.name == "Test RLC Circuit"
    assert circuit.ground_net_name == "gnd"
    assert len(circuit.components) == 3
    assert len(circuit.nets) == 3
    assert len(circuit.external_ports) == 1
    assert circuit.parameter_manager is not None

    # Check sweep parsing
    assert hasattr(circuit, 'frequency_sweep_hz')
    freqs = circuit.frequency_sweep_hz
    assert isinstance(freqs, np.ndarray)
    assert len(freqs) == 1
    np.testing.assert_allclose(freqs[0], 1e9) # Check value in Hz


def test_valid_yaml_file_parsing(parser, valid_yaml_string, tmp_path):
    """Test parsing a valid YAML file."""
    p = tmp_path / "test_circuit.yaml"
    p.write_text(valid_yaml_string)
    circuit = parser.parse(p) # Pass Path object
    assert isinstance(circuit, Circuit)
    assert circuit.name == "Test RLC Circuit"
    assert "R1" in circuit.components
    assert hasattr(circuit, 'frequency_sweep_hz') # Check sweep loaded

    circuit_from_str_path = parser.parse(str(p)) # Pass path as string
    assert isinstance(circuit_from_str_path, Circuit)
    assert circuit_from_str_path.name == "Test RLC Circuit"
    assert hasattr(circuit_from_str_path, 'frequency_sweep_hz')


def test_valid_yaml_stream_parsing(parser, valid_yaml_string):
    """Test parsing a valid YAML stream."""
    stream = StringIO(valid_yaml_string)
    circuit = parser.parse(stream)
    assert isinstance(circuit, Circuit)
    assert circuit.name == "Test RLC Circuit"
    assert hasattr(circuit, 'frequency_sweep_hz')


def test_missing_components_section(parser, yaml_missing_components):
     # Schema requires 'components', let's see if Cerberus catches it
     # (Depends on how schema is defined - previously made optional for test)
     # Assuming 'components' is required by the latest schema in parser.py
     with pytest.raises(SchemaValidationError) as excinfo:
         parser.parse(yaml_missing_components)
     assert "'components': ['required field']" in str(excinfo.value.errors).lower()


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
    assert "'foobars'" in str(excinfo.value)


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


# --- Sweep Parsing Tests ---

def test_sweep_parsing_linear(parser, valid_yaml_with_linear_sweep):
    """Test parsing a valid linear sweep."""
    circuit = parser.parse(valid_yaml_with_linear_sweep)
    assert hasattr(circuit, 'frequency_sweep_hz')
    freqs = circuit.frequency_sweep_hz
    assert isinstance(freqs, np.ndarray)
    assert len(freqs) == 10
    # Use assert_quantity_close for comparing quantities
    assert_quantity_close(Quantity(freqs[0], 'Hz'), Quantity('1 MHz'))
    assert_quantity_close(Quantity(freqs[-1], 'Hz'), Quantity('100 MHz'))
    # Check if linear using numpy tools
    assert np.allclose(np.diff(freqs, 2), 0, atol=1e-9), "Sweep points are not linear"

def test_sweep_parsing_log(parser, yaml_log_sweep):
    """Test parsing a valid log sweep."""
    circuit = parser.parse(yaml_log_sweep)
    freqs = circuit.frequency_sweep_hz
    assert len(freqs) == 3
    assert_quantity_close(Quantity(freqs[0], 'Hz'), Quantity('1 kHz'))
    assert_quantity_close(Quantity(freqs[-1], 'Hz'), Quantity('1 GHz'))
    # Check if log spacing (ratio of consecutive points is constant)
    ratios = freqs[1:] / freqs[:-1]
    assert np.allclose(ratios, ratios[0]), "Sweep points are not logarithmic"

def test_sweep_parsing_list(parser, yaml_list_sweep):
    """Test parsing a valid list sweep (handles duplicates and sorting)."""
    circuit = parser.parse(yaml_list_sweep)
    freqs = circuit.frequency_sweep_hz
    assert len(freqs) == 2 # Duplicate removed and sorted
    assert_quantity_close(Quantity(freqs[0], 'Hz'), Quantity('10 MHz'))
    assert_quantity_close(Quantity(freqs[1], 'Hz'), Quantity('20 MHz'))
    assert np.all(np.diff(freqs) > 0) # Check if sorted


def test_sweep_parsing_bad_unit(parser, yaml_sweep_bad_unit):
     """Test parsing sweep with invalid frequency units."""
     with pytest.raises(ParsingError, match="Failed to parse sweep configuration"):
          parser.parse(yaml_sweep_bad_unit)


def test_sweep_parsing_missing_param_schema(parser, yaml_sweep_missing_stop):
     """Test schema validation catches missing sweep params based on dependencies."""
     with pytest.raises(ParsingError) as excinfo:
          parser.parse(yaml_sweep_missing_stop)

def test_sweep_parsing_log_zero_start(parser, yaml_sweep_log_zero_start):
     """Test log sweep with start <= 0 raises ValueError."""
     with pytest.raises(ParsingError, match="Log sweep start frequency must be > 0"):
          parser.parse(yaml_sweep_log_zero_start)

def test_sweep_parsing_invalid_range(parser, yaml_sweep_invalid_range):
     """Test sweep with start >= stop raises ValueError."""
     with pytest.raises(ParsingError, match="Start frequency .* must be less than stop frequency"):
          parser.parse(yaml_sweep_invalid_range)

def test_sweep_schema_requires_sweep(parser):
    """Verify that the sweep section itself is required by schema."""
    yaml_no_sweep = """
circuit_name: No Sweep Circuit
components: [{type: Resistor, id: R1, ports: {p1:P1, p2:gnd}, parameters: {resistance: '1k'}}]
ports: [{id: P1, reference_impedance: '50'}]
"""
    with pytest.raises(SchemaValidationError) as excinfo:
        parser.parse(yaml_no_sweep)
    assert "'sweep': ['required field']" in str(excinfo.value.errors)


# --- Final Basic Check ---
def test_parser_instantiation(parser):
    """Test that the parser object can be created."""
    assert parser is not None
    assert hasattr(parser, 'parse')
    assert hasattr(parser, '_validator')