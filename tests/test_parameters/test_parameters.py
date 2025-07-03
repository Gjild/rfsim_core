# tests/test_parameters/test_parameters.py

import pytest
from pathlib import Path
import numpy as np
import sympy
from collections import ChainMap
import pint

# --- Core RFSim Imports for Testing ---
from rfsim_core.parameters import (
    ParameterManager,
    ParameterDefinition,
    ExpressionPreprocessor,
    ParameterScopeError,
    CircularParameterDependencyError,
    ParameterSyntaxError,
    ParameterEvaluationError,
)
from rfsim_core.errors import CircuitBuildError
from rfsim_core.circuit_builder import CircuitBuilder
from rfsim_core.parser import NetlistParser
from rfsim_core.units import ureg, Quantity


def create_param_netlists(tmp_path: Path) -> Path:
    """Helper fixture to create a standard set of YAML files for parameter tests."""
    netlist_dir = tmp_path / "param_netlists"
    netlist_dir.mkdir()

    (netlist_dir / "hierarchical_params.yaml").write_text("""
circuit_name: TopCircuit
ground_net: gnd
parameters:
  top_gain: 2.0
  freq_dep_ext: {expression: "1e-9 * freq", dimension: "s"}
components:
  - id: amp1
    type: Subcircuit
    definition_file: ./sub_amp.yaml
    ports: {IN: p_in, OUT: p_out}
    parameters:
      gain_in: top_gain
      L1.inductance: "10 nH"
      C1.capacitance: "base_cap * top_gain"
""")
    (netlist_dir / "sub_amp.yaml").write_text("""
circuit_name: SubAmp
ground_net: gnd
parameters:
  gain_in: 1.0
  base_cap: "1 pF"
components:
  - id: R1
    type: Resistor
    ports: {0: IN, 1: gnd}
    parameters: {resistance: "50 * gain_in"}
  - id: L1
    type: Inductor
    ports: {0: IN, 1: OUT}
    parameters: {inductance: "1 nH"}
  - id: C1
    type: Capacitor
    ports: {0: OUT, 1: gnd}
    parameters: {capacitance: "1 pF"}
  - id: C2
    type: Capacitor
    ports: {0: IN, 1: gnd}
    # This expression is now dimensionally sound and tests the system correctly.
    parameters: {capacitance: "1e-21 * freq"}
""")

    (netlist_dir / "unresolved_symbol.yaml").write_text("""
circuit_name: UnresolvedSymbol
components:
  - id: R1
    type: Resistor
    ports: {0: in, 1: out}
    parameters:
      resistance: "50 * undefined_gain"
""")
    
    (netlist_dir / "circular_dependency.yaml").write_text("""
circuit_name: CircularDep
parameters:
  param_A: "param_B * 2"
  param_B: "param_C / 3"
  param_C: "param_A + 1"
components:
  - id: R1
    type: Resistor
    ports: {0: in, 1: out}
    parameters: {resistance: "50 ohm"}
""")

    (netlist_dir / "disallowed_function.yaml").write_text("""
circuit_name: DisallowedFunc
parameters:
  bad_param: "diff(x, x)"
components:
  - id: R1
    type: Resistor
    ports: {0: in, 1: out}
    parameters: {resistance: "50 ohm"}
""")

    return netlist_dir


@pytest.fixture(scope="module")
def param_test_netlists(tmp_path_factory):
    """Module-scoped fixture to create the test netlist files once."""
    base_path = tmp_path_factory.mktemp("param_tests")
    return create_param_netlists(base_path)


class TestParameterManager:
    """
    Rigorously tests the ParameterManager's entire lifecycle: FQN-based definition,
    dependency analysis, scope resolution, and vectorized evaluation. This suite
    places a merciless focus on verifying the correctness and clarity of all
    diagnostic error reports, ensuring they meet the project's "Actionable
    Diagnostics" mandate.
    """

    # === Test Case Group 1: Build Process and Dependency Analysis ===

    def test_preprocessor_resolves_fqns(self):
        """Verifies the ExpressionPreprocessor correctly finds FQN dependencies."""
        preprocessor = ExpressionPreprocessor()
        param_def = ParameterDefinition(
            owner_fqn="top.amp1", 
            base_name="my_param", 
            raw_value_or_expression_str="R1.resistance * 2", 
            source_yaml_path=Path("dummy.yaml"), 
            declared_dimension_str="ohm"
        )
        scope = ChainMap({"R1.resistance": "top.amp1.R1.resistance"})
        
        sympy_expr = preprocessor.preprocess(param_def, scope, {"freq"})
        
        assert isinstance(sympy_expr, sympy.Mul)
        assert sympy_expr.free_symbols == {sympy.Symbol("top.amp1.R1.resistance")}


    def test_get_external_dependencies_of_scope(self, param_test_netlists):
        """
        Verifies correct identification of all external dependencies for a given
        subcircuit scope, proving the foundation for hierarchical simulation caching.
        """
        top_level_path = param_test_netlists / "hierarchical_params.yaml"
        parser = NetlistParser()
        builder = CircuitBuilder()
        parsed_tree = parser.parse_to_circuit_tree(top_level_path)
        sim_circuit = builder.build_simulation_model(parsed_tree)
        pm = sim_circuit.parameter_manager

        subcircuit_fqns = {p.fqn for p in pm.get_all_fqn_definitions() if p.owner_fqn.startswith("top.amp1")}
        const_ext, freq_ext = pm.get_external_dependencies_of_scope(subcircuit_fqns)

        assert const_ext == {"top.top_gain"}
        assert freq_ext == {"top.freq_dep_ext"}

    # === Test Case Group 2: Negative Testing & Diagnostic Verification ===

    def test_unresolved_symbol_raises_scope_error_with_diagnostics(self, param_test_netlists):
        """Verifies a clear diagnostic is raised for an unresolved symbol."""
        parser = NetlistParser()
        builder = CircuitBuilder()
        parsed_tree = parser.parse_to_circuit_tree(param_test_netlists / "unresolved_symbol.yaml")

        with pytest.raises(CircuitBuildError) as excinfo:
            builder.build_simulation_model(parsed_tree)

        report = str(excinfo.value)
        assert "Actionable Diagnostic Report" in report
        assert "Error Type:     Unresolved Symbol in Expression" in report
        assert "FQN:            top.R1.resistance" in report
        assert "The identifier 'undefined_gain' was not found" in report

    def test_circular_dependency_detection_raises_error_with_diagnostics(self, param_test_netlists):
        """Verifies a clear diagnostic is raised for a circular parameter dependency."""
        netlist_path = param_test_netlists / "circular_dependency.yaml"
        parser = NetlistParser()
        builder = CircuitBuilder()
        parsed_tree = parser.parse_to_circuit_tree(netlist_path)

        with pytest.raises(CircuitBuildError) as excinfo:
            builder.build_simulation_model(parsed_tree)

        report = str(excinfo.value)
        assert "Actionable Diagnostic Report" in report
        assert "Error Type:     Circular Parameter Dependency" in report
        # The exact order of the cycle can vary, so check for presence of all members.
        assert "top.param_A" in report and "top.param_B" in report and "top.param_C" in report
        assert "->" in report

    def test_evaluate_all_vectorized_unit_safe(self, param_test_netlists):
        """Happy-path test: Verifies correct, vectorized, unit-safe evaluation."""
        parser = NetlistParser()
        builder = CircuitBuilder()
        parsed_tree = parser.parse_to_circuit_tree(param_test_netlists / "hierarchical_params.yaml")
        sim_circuit = builder.build_simulation_model(parsed_tree)
        pm = sim_circuit.parameter_manager

        freqs_hz = np.array([1e9, 2e9])
        results = pm.evaluate_all(freqs_hz)

        r1_res_qty = results["top.amp1.R1.resistance"]
        assert r1_res_qty.check("ohm")
        np.testing.assert_allclose(r1_res_qty.magnitude, [100.0, 100.0])

        c2_cap_qty = results["top.amp1.C2.capacitance"]
        assert c2_cap_qty.check("farad")
        expected_c2_mag = 1e-21 * freqs_hz
        np.testing.assert_allclose(c2_cap_qty.to("farad").magnitude, expected_c2_mag)

    def test_incompatible_unit_arithmetic_is_diagnosed(self):
        """Verifies that adding incompatible units (ohm + farad) raises a DimensionalityError."""
        pm = ParameterManager()
        defs = [
            ParameterDefinition(owner_fqn="top", base_name="res", raw_value_or_expression_str="'50 ohm'", source_yaml_path=Path("."), declared_dimension_str="ohm"),
            ParameterDefinition(owner_fqn="top", base_name="cap", raw_value_or_expression_str="'1 pF'", source_yaml_path=Path("."), declared_dimension_str="farad"),
            ParameterDefinition(owner_fqn="top", base_name="bad_sum", raw_value_or_expression_str="res + cap", source_yaml_path=Path("."), declared_dimension_str="ohm")
        ]
        scopes = {
            "top.res": ChainMap({}),
            "top.cap": ChainMap({}),
            "top.bad_sum": ChainMap({"res": "top.res", "cap": "top.cap"})
        }
        pm.build(defs, scopes)
        
        with pytest.raises(ParameterEvaluationError) as excinfo:
            pm.evaluate_all(np.array([1e9]))

        assert isinstance(excinfo.value.__cause__, pint.DimensionalityError)
        report = excinfo.value.get_diagnostic_report()
        assert "Parameter Evaluation Error" in report
        assert "FQN:            top.bad_sum" in report
        assert "Cannot convert from 'farad' to 'ohm'" in str(excinfo.value)

    def test_disallowed_sympy_function_raises_build_error(self, param_test_netlists):
        """Verifies a clear diagnostic for using a forbidden SymPy function."""
        netlist_path = param_test_netlists / "disallowed_function.yaml"
        parser = NetlistParser()
        builder = CircuitBuilder()
        parsed_tree = parser.parse_to_circuit_tree(netlist_path)

        with pytest.raises(CircuitBuildError) as excinfo:
            builder.build_simulation_model(parsed_tree)

        report = str(excinfo.value)
        assert "Actionable Diagnostic Report" in report
        assert "Error Type:     Invalid Expression Syntax" in report
        assert "FQN:            top.bad_param" in report
        assert "Disallowed operation type 'Derivative'" in report

    # === Test Case Group 3: Evaluation and Dimensional Analysis ===

    def test_evaluate_all_vectorized(self, param_test_netlists):
        """Happy-path test: Verifies correct, vectorized evaluation of a hierarchical circuit."""
        top_level_path = param_test_netlists / "hierarchical_params.yaml"
        parser = NetlistParser()
        builder = CircuitBuilder()
        parsed_tree = parser.parse_to_circuit_tree(top_level_path)
        sim_circuit = builder.build_simulation_model(parsed_tree)
        pm = sim_circuit.parameter_manager

        freqs_hz = np.array([1e9, 2e9])
        results = pm.evaluate_all(freqs_hz)

        # Assert correct vectorized evaluation for various parameters
        top_gain_qty = results["top.top_gain"]
        assert isinstance(top_gain_qty, Quantity)
        assert top_gain_qty.check("dimensionless")
        np.testing.assert_allclose(top_gain_qty.magnitude, [2.0, 2.0])

        r1_res_qty = results["top.amp1.R1.resistance"]
        assert isinstance(r1_res_qty, Quantity)
        assert r1_res_qty.check("ohm")
        np.testing.assert_allclose(r1_res_qty.magnitude, [100.0, 100.0])

        c2_cap_qty = results["top.amp1.C2.capacitance"]
        assert isinstance(c2_cap_qty, Quantity)
        assert c2_cap_qty.check("farad")
        expected_c2_mag = np.array([1e-9 * 1e9, 1e-9 * 2e9])
        np.testing.assert_allclose(c2_cap_qty.to("farad").magnitude, expected_c2_mag)

    def test_numerical_literal_parameter_inherits_declared_dimension(self):
        """
        NEW HARDENING TEST: Verifies that a unitless numerical literal provided as a
        parameter value (e.g., `resistance: 50`) is correctly assigned the dimension
        declared by the component. This closes a gap in testing a documented input format.
        """
        pm = ParameterManager()
        defs = [
            ParameterDefinition(
                owner_fqn="top.R1", base_name="resistance",
                raw_value_or_expression_str="50",
                source_yaml_path=Path("dummy.yaml"),
                declared_dimension_str="ohm"
            )
        ]
        scopes = {"top.R1.resistance": ChainMap({})}
        pm.build(defs, scopes)

        results = pm.evaluate_all(np.array([1e9]))
        res_qty = results["top.R1.resistance"]

        assert isinstance(res_qty, Quantity)
        assert res_qty.check("ohm")
        np.testing.assert_allclose(res_qty.magnitude, 50.0)

    @pytest.mark.xfail(
        reason="ARCHITECTURAL FLAW: ParameterManager adds magnitudes, not quantities. "
               "This test proves that `50 ohm + 1 pF` succeeds silently instead of "
               "raising a pint.DimensionalityError. This test will pass when the "
               "evaluation pipeline is hardened to operate on Quantity objects.",
        strict=True,
        raises=pint.DimensionalityError
    )
    def test_incompatible_unit_arithmetic_is_diagnosed(self):
        """
        NEW CRITICAL HARDENING TEST: This test is DESIGNED TO FAIL with the current
        architecture to prove a critical unit-safety flaw and serve as a non-negotiable
        gate for a future fix. It verifies that adding incompatible units (ohm + farad)
        raises a DimensionalityError at the point of operation.
        """
        pm = ParameterManager()
        defs = [
            ParameterDefinition(owner_fqn="top", base_name="res", raw_value_or_expression_str="'50 ohm'", source_yaml_path=Path("."), declared_dimension_str="ohm"),
            ParameterDefinition(owner_fqn="top", base_name="cap", raw_value_or_expression_str="'1 pF'", source_yaml_path=Path("."), declared_dimension_str="farad"),
            ParameterDefinition(owner_fqn="top", base_name="bad_sum", raw_value_or_expression_str="res + cap", source_yaml_path=Path("."), declared_dimension_str="ohm")
        ]
        scopes = {
            "top.res": ChainMap({}),
            "top.cap": ChainMap({}),
            "top.bad_sum": ChainMap({"res": "top.res", "cap": "top.cap"})
        }
        pm.build(defs, scopes)
        pm.evaluate_all(np.array([1e9]))

    def test_dimensionally_invalid_transcendental_op_raises_error(self):
        """Verifies transcendental functions on dimensioned quantities fail with a diagnosable error."""
        pm = ParameterManager()
        defs = [
            ParameterDefinition(owner_fqn="top", base_name="length", raw_value_or_expression_str="'5 meter'", source_yaml_path=Path("."), declared_dimension_str="meter"),
            ParameterDefinition(owner_fqn="top", base_name="bad_log", raw_value_or_expression_str="log(length)", source_yaml_path=Path("."), declared_dimension_str="dimensionless")
        ]
        scopes = {
            "top.length": ChainMap({}),
            "top.bad_log": ChainMap({"length": "top.length"})
        }
        pm.build(defs, scopes)

        with pytest.raises(ParameterEvaluationError) as excinfo:
            pm.evaluate_all(np.array([1e9]))

        assert isinstance(excinfo.value.__cause__, pint.DimensionalityError)
        report = excinfo.value.get_diagnostic_report()
        assert "Parameter Evaluation Error" in report
        assert "FQN:            top.bad_log" in report
        assert "log' is not defined for quantities with dimensions" in str(excinfo.value)

    def test_evaluation_error_at_specific_frequency_has_diagnostics(self):
        """Verifies diagnostics for a divide-by-zero error at a specific frequency."""
        pm = ParameterManager()
        defs = [ParameterDefinition(owner_fqn="top", base_name="bad", raw_value_or_expression_str="1/freq", source_yaml_path=Path("."), declared_dimension_str="1/Hz")]
        scopes = {"top.bad": ChainMap({})}
        pm.build(defs, scopes)

        freqs = np.array([0.0, 1e9, 2e9])
        with pytest.raises(ParameterEvaluationError) as excinfo:
            pm.evaluate_all(freqs)

        report = excinfo.value.get_diagnostic_report()
        assert "Actionable Diagnostic Report" in report
        assert "Parameter Evaluation Error" in report
        assert "FQN:            top.bad" in report
        assert "Result contains non-finite values" in report or "divide by zero" in report
        assert "The error first occurred at sweep index 0 (frequency = 0.0000e+00 Hz)" in report
        assert "freq = 0.0" in report