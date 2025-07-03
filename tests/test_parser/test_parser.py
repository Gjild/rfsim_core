# tests/test_parser/test_parser.py

import pytest
import yaml
from pathlib import Path

# --- Import core RFSim classes to be tested ---
from rfsim_core.parser import (
    NetlistParser,
    ParsingError,
    SchemaValidationError,
    ParsedLeafComponentData,
    ParsedSubcircuitData
)
from rfsim_core.circuit_builder import CircuitBuilder
from rfsim_core.errors import CircuitBuildError

# --- Helper function to create the netlist files for tests ---

def create_netlist_files(tmp_path: Path) -> Path:
    """
    Creates a standard set of YAML files in a temporary directory for testing.
    This follows the structure laid out in the "Definitive Test Architecture".
    """
    netlists_path = tmp_path / "netlists"

    # --- Valid Netlists ---
    valid_path = netlists_path / "valid"
    valid_path.mkdir(parents=True, exist_ok=True)
    (valid_path / "simple_rlc.yaml").write_text("""
circuit_name: SimpleRLC
ground_net: gnd
components:
  - id: R1
    type: Resistor
    ports: {0: net_in, 1: net_out}
    parameters: {resistance: '50 ohm'}
  - id: L1
    type: Inductor
    ports: {0: net_out, 1: gnd}
    parameters: {inductance: '10 nH'}
ports:
  - {id: net_in, reference_impedance: '50 ohm'}
  - {id: net_out, reference_impedance: '50 ohm'}
    """)

    (valid_path / "hierarchical_attenuator.yaml").write_text("""
circuit_name: HierarchicalAttenuator
ground_net: gnd
components:
  - id: X1
    type: Subcircuit
    definition_file: ./attenuator_section.yaml
    ports:
      IN: net_in
      OUT: net_mid
    parameters:
      R1_val: '100 ohm'
      R2_val: '50 ohm'
  - id: X2
    type: Subcircuit
    definition_file: ./attenuator_section.yaml
    ports:
      IN: net_mid
      OUT: net_out
ports:
  - {id: net_in, reference_impedance: '50 ohm'}
  - {id: net_out, reference_impedance: '50 ohm'}
    """)

    (valid_path / "attenuator_section.yaml").write_text("""
circuit_name: AttenuatorSection
ground_net: gnd
parameters:
  R1_val: '100 ohm'
  R2_val: '50 ohm'
components:
  - id: R1
    type: Resistor
    ports: {0: IN, 1: OUT}
    parameters: {resistance: R1_val}
  - id: R2
    type: Resistor
    ports: {0: OUT, 1: gnd}
    parameters: {resistance: R2_val}
ports:
  - {id: IN, reference_impedance: '50 ohm'}
  - {id: OUT, reference_impedance: '50 ohm'}
    """)

    (valid_path / "no_gnd_key.yaml").write_text("""
circuit_name: NoGndKey
components:
  - id: R1
    type: Resistor
    ports: {0: in, 1: gnd}
    parameters: {resistance: '50 ohm'}
    """)

    # --- Invalid Schema Netlists ---
    invalid_schema_path = netlists_path / "invalid_schema"
    invalid_schema_path.mkdir(parents=True, exist_ok=True)
    (invalid_schema_path / "invalid_identifier.yaml").write_text("""
circuit_name: InvalidID-circuit # Invalid due to '-'
components:
  - id: R-1 # Invalid ID due to '-'
    type: Resistor
    ports:
      0: net.in # Invalid net name due to '.'
      1: gnd
    parameters:
      resistance: '50 ohm'
    """)
    (invalid_schema_path / "duplicate_component_id.yaml").write_text("""
circuit_name: DupeID
components:
  - id: R1
    type: Resistor
    ports: {0: in, 1: mid}
    parameters: {resistance: '50 ohm'}
  - id: R1 # Duplicate ID
    type: Resistor
    ports: {0: mid, 1: gnd}
    parameters: {resistance: '100 ohm'}
    """)
    (invalid_schema_path / "malformed.yaml").write_text("""
circuit_name: Malformed
components:
- id: R1
  type: Resistor
   ports: {0: in, 1: out} # Bad indentation
  parameters: {resistance: '50 ohm'}
    """)
    (invalid_schema_path / "circular_top.yaml").write_text("""
circuit_name: CircularTop
components:
  - id: X1
    type: Subcircuit
    definition_file: ./circular_sub.yaml
    ports: {IN: net_in, OUT: net_out}
    """)
    (invalid_schema_path / "circular_sub.yaml").write_text("""
circuit_name: CircularSub
components:
  - id: X1
    type: Subcircuit
    definition_file: ./circular_top.yaml
    ports: {IN: IN, OUT: OUT}
    """)
    (invalid_schema_path / "python_keyword_id.yaml").write_text("""
circuit_name: KeywordID
parameters:
  my_param: 'for * 2' # This expression will cause a syntax error
components:
  - id: R1
    type: Resistor
    ports: {0: in, 1: out}
    parameters: {resistance: '50 ohm'}
  - id: for # Using a Python keyword as an ID. Parsing is ok, build should fail.
    type: Resistor
    ports: {0: out, 1: gnd}
    parameters: {resistance: '100 ohm'}
ports:
  - {id: in, reference_impedance: '50 ohm'}
    """)

    return netlists_path


@pytest.fixture(scope="module")
def netlists_dir(tmp_path_factory):
    """A module-scoped pytest fixture to create the test netlist files once per module."""
    tmp_path = tmp_path_factory.mktemp("netlists_root")
    return create_netlist_files(tmp_path)


class TestNetlistParser:
    """
    Tests the NetlistParser's ability to correctly parse valid netlists into
    the IR and robustly reject malformed files with actionable errors, as
    defined in the project's definitive test architecture.
    """

    # === Test Case Group 1: Valid Netlist Parsing ===

    def test_parse_simple_rlc(self, netlists_dir):
        """Verifies parsing a basic RLC circuit into a ParsedCircuitNode IR."""
        parser = NetlistParser()
        netlist_path = netlists_dir / "valid" / "simple_rlc.yaml"

        ir_root = parser.parse_to_circuit_tree(netlist_path)

        assert ir_root.circuit_name == "SimpleRLC"
        assert ir_root.ground_net_name == "gnd"
        assert len(ir_root.components) == 2

        r1_ir = ir_root.components[0]
        assert isinstance(r1_ir, ParsedLeafComponentData)
        assert r1_ir.instance_id == "R1"
        assert r1_ir.component_type == "Resistor"
        assert r1_ir.raw_ports_dict == {0: "net_in", 1: "net_out"}
        assert r1_ir.raw_parameters_dict == {"resistance": "50 ohm"}

        l1_ir = ir_root.components[1]
        assert isinstance(l1_ir, ParsedLeafComponentData)
        assert l1_ir.instance_id == "L1"
        assert l1_ir.component_type == "Inductor"
        assert l1_ir.raw_ports_dict == {0: "net_out", 1: "gnd"}

        assert len(ir_root.raw_external_ports_list) == 2
        assert ir_root.raw_external_ports_list[0]['id'] == 'net_in'

    def test_parse_hierarchical(self, netlists_dir):
        """Verifies correct recursive parsing of a hierarchical netlist."""
        parser = NetlistParser()
        netlist_path = netlists_dir / "valid" / "hierarchical_attenuator.yaml"

        ir_root = parser.parse_to_circuit_tree(netlist_path)

        assert ir_root.circuit_name == "HierarchicalAttenuator"
        assert len(ir_root.components) == 2

        x1_ir = ir_root.components[0]
        assert isinstance(x1_ir, ParsedSubcircuitData)
        assert x1_ir.instance_id == "X1"
        assert x1_ir.component_type == "Subcircuit"
        assert x1_ir.raw_port_mapping == {"IN": "net_in", "OUT": "net_mid"}
        assert x1_ir.raw_parameter_overrides == {"R1_val": "100 ohm", "R2_val": "50 ohm"}

        # Check the nested IR node
        sub_ir = x1_ir.sub_circuit_definition_node
        assert sub_ir is not None
        assert sub_ir.circuit_name == "AttenuatorSection"
        assert len(sub_ir.components) == 2
        assert len(sub_ir.raw_parameters_dict) == 2
        assert sub_ir.raw_parameters_dict['R1_val'] == '100 ohm'

    def test_default_ground_net_is_applied(self, netlists_dir):
        """Verifies the default ground net name is used when the key is omitted."""
        parser = NetlistParser()
        netlist_path = netlists_dir / "valid" / "no_gnd_key.yaml"

        ir_root = parser.parse_to_circuit_tree(netlist_path)

        assert ir_root.ground_net_name == "gnd"

    # === Test Case Group 2: Schema and File Error Rejection (Negative Testing) ===

    def test_invalid_identifier_raises_schema_error(self, netlists_dir):
        """Verifies the custom id_regex validator rejects forbidden characters."""
        parser = NetlistParser()
        netlist_path = netlists_dir / "invalid_schema" / "invalid_identifier.yaml"

        with pytest.raises(SchemaValidationError) as excinfo:
            parser.parse_to_circuit_tree(netlist_path)

        report = excinfo.value.get_diagnostic_report()
        assert "Schema Validation Error" in report
        assert "R-1" in report and "forbidden character(s): ['-']" in report
        assert "net.in" in report and "forbidden character(s): ['.']" in report
        assert "InvalidID-circuit" in report and "forbidden character(s): ['-']" in report

    def test_duplicate_component_id_raises_schema_error(self, netlists_dir):
        """Verifies Cerberus's 'unique' rule catches duplicate component IDs."""
        parser = NetlistParser()
        netlist_path = netlists_dir / "invalid_schema" / "duplicate_component_id.yaml"

        with pytest.raises(SchemaValidationError) as excinfo:
            parser.parse_to_circuit_tree(netlist_path)

        report = excinfo.value.get_diagnostic_report()
        assert "Schema Validation Error" in report
        # Use the correct STRING key to access the error message
        assert "Duplicate values found for key 'id'" in excinfo.value.errors['components'][0]
        assert "duplicate values" in report.lower()

    def test_file_not_found_raises_file_error(self, tmp_path):
        """Verifies a clear error is raised for a non-existent file."""
        parser = NetlistParser()
        non_existent_path = tmp_path / "does_not_exist.yaml"

        with pytest.raises(ParsingError):
            parser.parse_to_circuit_tree(non_existent_path)

    def test_circular_dependency_raises_parsing_error(self, netlists_dir):
        """Verifies that circular subcircuit includes are detected and raise an error."""
        parser = NetlistParser()
        netlist_path = netlists_dir / "invalid_schema" / "circular_top.yaml"

        with pytest.raises(ParsingError) as excinfo:
            parser.parse_to_circuit_tree(netlist_path)

        assert "Circular subcircuit dependency detected" in str(excinfo.value)
        report = excinfo.value.get_diagnostic_report()
        assert "Circular subcircuit dependency detected" in report
        assert "circular_top.yaml" in report

    def test_malformed_yaml_raises_parsing_error(self, netlists_dir):
        """Verifies that invalid YAML syntax is caught and reported."""
        parser = NetlistParser()
        netlist_path = netlists_dir / "invalid_schema" / "malformed.yaml"

        with pytest.raises(ParsingError) as excinfo:
            parser.parse_to_circuit_tree(netlist_path)

        assert "Invalid YAML syntax" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, yaml.YAMLError)
        report = excinfo.value.get_diagnostic_report()
        assert "Invalid YAML syntax" in report

    def test_identifier_as_python_keyword_is_handled_diagnosably(self, netlists_dir):
        """
        High-Value Hardening Test:
        Verifies that using a Python keyword as an identifier, while passing parsing,
        is caught with a diagnosable error during the circuit build process.
        """
        parser = NetlistParser()
        builder = CircuitBuilder()
        netlist_path = netlists_dir / "invalid_schema" / "python_keyword_id.yaml"

        # The parsing itself should succeed because the ID_REGEX allows it.
        parsed_tree = parser.parse_to_circuit_tree(netlist_path)
        assert parsed_tree is not None
        assert any(comp.instance_id == 'for' for comp in parsed_tree.components)

        # The build process should fail when the ExpressionPreprocessor encounters
        # the keyword 'for' in an expression.
        with pytest.raises(CircuitBuildError) as excinfo:
            builder.build_simulation_model(parsed_tree)

        # Assert on the final, user-facing diagnostic report.
        report = str(excinfo.value)
        assert "Actionable Diagnostic Report" in report
        assert "Error Type:     Invalid Expression Syntax" in report
        assert "FQN:            top.my_param" in report
        assert "User Input:     'for * 2'" in report
        # The details should indicate a syntax error because 'for' is an invalid
        # token in the SymPy/Python expression parsing context.
        assert "invalid syntax" in report.lower() or "invalid token" in report.lower()