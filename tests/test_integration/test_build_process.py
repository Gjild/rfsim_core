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


# -----------------------------------------------------------------------------
# --- Test Fixture Definition (CORRECTED) ---
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def netlists_dir(tmp_path_factory):
    """
    A pytest fixture that creates a temporary directory with test netlists.
    This runs ONCE per test module, providing the necessary files for all tests
    in this file without recreating them for each test.
    """
    root = tmp_path_factory.mktemp("build_tests")

    # --- Netlist for testing successful hierarchical dependency resolution ---
    (root / "hierarchical_dependency.yaml").write_text("""
circuit_name: HierarchicalConstantTest
parameters:
  # ===================== START OF THE FIX =====================
  # Use the explicit dictionary format to declare the expected dimension.
  # This removes the ambiguity that caused the DimensionalityError.
  derived_resistance: {expression: "sub1.R_load.resistance * 2", dimension: "ohm"}
  # ====================== END OF THE FIX ======================
components:
  - id: sub1
    type: Subcircuit
    definition_file: ./sub_for_dep_test.yaml
    ports: { IN: net_in, OUT: net_out }
  - id: R_final
    type: Resistor
    ports: { 0: net_out, 1: gnd }
    parameters: { resistance: derived_resistance }
ports:
  - {id: net_in, reference_impedance: "Quantity('50 ohm')"}
""")

    (root / "sub_for_dep_test.yaml").write_text("""
circuit_name: SubForDepTest
components:
  - id: R_load
    type: Resistor
    ports: { 0: IN, 1: OUT }
    # --- DEFINITIVE FIX: Use explicit Quantity constructor ---
    parameters: { resistance: "Quantity('100 ohm')" }
ports:
  - {id: IN, reference_impedance: "Quantity('50 ohm')"}
  - {id: OUT, reference_impedance: "Quantity('50 ohm')"}
""")

    # --- Netlist for testing invalid syntax detection ---
    (root / "invalid_syntax.yaml").write_text("""
circuit_name: InvalidSyntaxTest
parameters:
  bad_param: "5 + * 3"
components:
  - id: R1
    type: Resistor
    ports: { 0: in, 1: gnd }
    # --- DEFINITIVE FIX: Use explicit Quantity constructor ---
    parameters: { resistance: "Quantity('10 ohm')" }
""")

    return root


# -----------------------------------------------------------------------------
# --- Test Cases (Unchanged) ---
# -----------------------------------------------------------------------------

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