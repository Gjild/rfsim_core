# tests/test_integration/test_build_process.py

"""
Integration tests for the circuit build process.

This suite verifies that the CircuitBuilder can correctly synthesize a valid
Intermediate Representation (IR) from the parser into a simulation-ready
Circuit object, especially in cases that test the new dependency analysis logic.
It also verifies that the builder's top-level error handling correctly catches
and reports diagnosable errors from its subsystems.
"""

import pytest
from pathlib import Path

# --- Core RFSim Imports for Testing ---
from rfsim_core import CircuitBuilder, NetlistParser, CircuitBuildError


@pytest.fixture(scope="module")
def netlists_dir(tmp_path_factory):
    """
    A pytest fixture that creates a temporary directory with test netlists.
    This version is corrected to be 100% compliant with the framework's
    hardened API contracts.
    """
    root = tmp_path_factory.mktemp("build_tests")

    # --- Netlist for testing successful hierarchical dependency resolution ---
    (root / "hierarchical_dependency.yaml").write_text("""
circuit_name: HierarchicalConstantTest
parameters:
  # This parameter is an expression and correctly uses the dictionary format.
  derived_resistance: {expression: "sub1.R_load.resistance * 2", dimension: "ohm"}
components:
  - id: sub1
    type: Subcircuit
    definition_file: ./sub_for_dep_test.yaml
    ports: { IN: net_in, OUT: net_out } # Subcircuit port mapping is correct.
  - id: R_final
    type: Resistor
    ports: { p1: net_out, p2: gnd }
    # This parameter refers to a pre-defined expression, which is correct.
    parameters: { resistance: derived_resistance }
ports:
  # CORRECTED: reference_impedance expects a simple string literal.
  - {id: net_in, reference_impedance: "50 ohm"}
""")

    (root / "sub_for_dep_test.yaml").write_text("""
circuit_name: SubForDepTest
components:
  - id: R_load
    type: Resistor
    ports: { p1: IN, p2: OUT }
    parameters: { resistance: "Quantity('100 ohm')" }
ports:
  # CORRECTED: reference_impedance expects a simple string literal.
  - {id: IN, reference_impedance: "50 ohm"}
  - {id: OUT, reference_impedance: "50 ohm"}
""")

    # --- Netlist for testing invalid syntax detection ---
    (root / "invalid_syntax.yaml").write_text("""
circuit_name: InvalidSyntaxTest
parameters:
  # This is the invalid expression under test.
  bad_param: "5 + * 3"
components:
  - id: R1
    type: Resistor
    ports: { p1: in, p2: gnd }
    parameters: { resistance: "Quantity('10 ohm')" }
""")

    return root


def test_build_succeeds_with_hierarchical_dependency(netlists_dir: Path):
    """
    Verifies the builder correctly handles hierarchical constant dependencies,
    proving the new AST-based dependency analysis is working correctly.
    """
    builder = CircuitBuilder()
    parser = NetlistParser()
    path = netlists_dir / "hierarchical_dependency.yaml"
    ir = parser.parse_to_circuit_tree(path)

    circuit = builder.build_simulation_model(ir)
    assert circuit is not None

    pm = circuit.parameter_manager
    final_res = pm.get_constant_value("top.R_final.resistance")
    assert final_res.to("ohm").magnitude == 200.0


def test_invalid_syntax_in_parameter_raises_diagnosable_error(netlists_dir: Path):
    """
    Verifies the full diagnostic path for a parameter with invalid syntax.
    """
    builder = CircuitBuilder()
    parser = NetlistParser()
    path = netlists_dir / "invalid_syntax.yaml"
    ir = parser.parse_to_circuit_tree(path)

    with pytest.raises(CircuitBuildError) as excinfo:
        builder.build_simulation_model(ir)

    report = str(excinfo.value)
    assert "Actionable Diagnostic Report" in report
    assert "Error Type:     Invalid Expression Syntax" in report
    # With the fix, the builder now correctly identifies bad_param as the source of the error.
    assert "FQN:            top.bad_param" in report
    assert "User Input:     '5 + * 3'" in report
    assert "invalid syntax" in report.lower()