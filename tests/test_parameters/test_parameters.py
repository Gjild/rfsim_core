# tests/test_parameters/test_parameters.py

"""
Definitive Validation Suite for the RFSim Core Parameter Expression Architecture.

This module provides a mercilessly rigorous and comprehensive suite of tests for the
ParameterManager and its related subsystems (ExpressionPreprocessor, CircuitBuilder
integration). Its primary purpose is to serve as the ultimate gatekeeper, validating
the three foundational pillars of the parameter architecture:

1.  **Correctness by Construction:** Verifying that the system correctly resolves
    hierarchical dependencies, handles overrides, and produces numerically accurate,
    vectorized results.

2.  **Intrinsic Unit Safety:** Proving that the `eval()`-based evaluation pipeline,
    operating on `pint.Quantity` objects, makes dimensionally-incompatible
    operations impossible to perform silently. This is the most critical feature
    of the architecture.

3.  **Actionable Diagnostics:** Ensuring that every conceivable user error—from
    syntax mistakes and unresolved symbols to circular dependencies and invalid
    runtime operations—is caught and reported with a clear, user-friendly, and
    diagnosable error report that guides the user to a solution.

The tests are organized into logical groups that build upon each other, starting
with the internal build process, moving to correct evaluation, and finally,
validating the robustness of the error handling. This file is the primary
guarantee that the parameter system has no showstopper flaws.
"""

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
    """
    Helper fixture to create a standard and focused set of YAML files for parameter tests.
    This revised version corrects the flawed lexical scoping assumption in the hierarchical test.
    """
    netlist_dir = tmp_path / "param_netlists"
    netlist_dir.mkdir()

    # --- Netlist 1: A comprehensive hierarchical design for "happy path" testing (CORRECTED) ---
    (netlist_dir / "hierarchical_params.yaml").write_text("""
circuit_name: TopCircuit
ground_net: gnd
parameters:
  top_gain: 2.0
  override_cap_value: "2 pF"
components:
  - id: amp1
    type: Subcircuit
    definition_file: ./sub_amp.yaml
    ports: {IN: p_in, OUT: p_out}
    parameters:
      gain_in: top_gain
      base_cap: override_cap_value
      L1.inductance: "10 nH"
ports:
  - {id: p_in, reference_impedance: '50 ohm'}
  - {id: p_out, reference_impedance: '50 ohm'}
""")
    (netlist_dir / "sub_amp.yaml").write_text("""
circuit_name: SubAmp
ground_net: gnd
parameters:
  gain_in: 1.0
  base_cap: "1 pF"
  C_slope: "1e-21 F/Hz"
  R_base: '50 ohm' 
  derived_capacitance: "base_cap * gain_in"
components:
  - id: R1
    type: Resistor
    ports: {0: IN, 1: gnd}
    parameters: {resistance: "R_base * gain_in"}
  - id: L1
    type: Inductor
    ports: {0: IN, 1: OUT}
    parameters: {inductance: "1 nH"}
  - id: C1
    type: Capacitor
    ports: {0: OUT, 1: gnd}
    parameters: {capacitance: derived_capacitance}
  - id: C2
    type: Capacitor
    ports: {0: IN, 1: gnd}
    parameters: {capacitance: "C_slope * freq"}
ports:
  - {id: IN, reference_impedance: '50 ohm'}
  - {id: OUT, reference_impedance: '50 ohm'}
""")

    # --- The rest of the error-case netlists remain the same, with one modification ---
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
  - {id: R1, type: Resistor, ports: {0: in, 1: gnd}, parameters: {resistance: "1 ohm"}}
""")
    
    # MODIFIED: Changed to a syntactically valid but undefined function
    (netlist_dir / "disallowed_function.yaml").write_text("""
circuit_name: DisallowedFunc
components:
  - id: R1
    type: Resistor
    ports: {0: in, 1: gnd}
    parameters: {resistance: "my_unsupported_function(1) * 1 ohm"}
""")
    
    (netlist_dir / "dimensional_error.yaml").write_text("""
circuit_name: DimensionalError
parameters:
  my_res: '50 ohm'
  my_cap: '1 pF'
  bad_param: "my_res + my_cap"
components:
  - {id: R1, type: Resistor, ports: {0: in, 1: gnd}, parameters: {resistance: bad_param}}
""")
    
    return netlist_dir


@pytest.fixture(scope="module")
def param_test_netlists(tmp_path_factory):
    """Module-scoped fixture to create the test netlist files once per module."""
    base_path = tmp_path_factory.mktemp("param_tests")
    return create_param_netlists(base_path)


class TestParameterManager:
    """
    Rigorously tests the ParameterManager's entire lifecycle: FQN-based definition,
    dependency analysis, scope resolution, vectorized evaluation, and error diagnostics.
    """

    # =========================================================================
    # === Test Group 1: Build Process & Dependency Analysis
    # =========================================================================

    def test_preprocessor_resolves_fqns_correctly(self):
        """
        Unit Test: Verifies the `ExpressionPreprocessor` correctly uses the `ast`
        transformer to resolve local names to their canonical Fully Qualified Names (FQNs).
        """
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

    def test_evaluation_order_is_correctly_computed(self, param_test_netlists):
        """
        Integration Test: Verifies that the internal evaluation order is topologically
        sorted, ensuring parameters are calculated before their dependents.
        """
        top_level_path = param_test_netlists / "hierarchical_params.yaml"
        parser = NetlistParser()
        builder = CircuitBuilder()
        parsed_tree = parser.parse_to_circuit_tree(top_level_path)
        sim_circuit = builder.build_simulation_model(parsed_tree)
        pm = sim_circuit.parameter_manager
        
        order = pm._evaluation_order
        
        assert order.index("top.override_cap_value") < order.index("top.amp1.base_cap")
        assert order.index("top.amp1.base_cap") < order.index("top.amp1.derived_capacitance")
        assert order.index("top.amp1.gain_in") < order.index("top.amp1.derived_capacitance")
        assert order.index("top.amp1.derived_capacitance") < order.index("top.amp1.C1.capacitance")
        assert order.index("freq") < order.index("top.amp1.C2.capacitance")

    def test_get_external_dependencies_of_scope(self, param_test_netlists):
        """
        Integration Test: Verifies correct identification of all external dependencies
        for a subcircuit's scope.
        """
        top_level_path = param_test_netlists / "hierarchical_params.yaml"
        parser = NetlistParser()
        builder = CircuitBuilder()
        parsed_tree = parser.parse_to_circuit_tree(top_level_path)
        sim_circuit = builder.build_simulation_model(parsed_tree)
        pm = sim_circuit.parameter_manager

        subcircuit_fqns = {p.fqn for p in pm.get_all_fqn_definitions() if p.owner_fqn.startswith("top.amp1")}
        
        const_ext, freq_ext = pm.get_external_dependencies_of_scope(subcircuit_fqns)

        assert const_ext == {"top.top_gain", "top.override_cap_value"}
        assert "top.amp1.C2.capacitance" in subcircuit_fqns
        assert pm._dependency_graph.has_edge('freq', 'top.amp1.C2.capacitance')


    # =========================================================================
    # === Test Group 2: Correct Evaluation & Intrinsic Unit Safety
    # =========================================================================

    def test_evaluate_all_handles_hierarchy_overrides_and_vectorization(self, param_test_netlists):
        """
        Happy-Path Test: Verifies correct, vectorized evaluation of a complex
        hierarchical circuit, including parameter overrides and frequency dependency.
        """
        top_level_path = param_test_netlists / "hierarchical_params.yaml"
        parser = NetlistParser()
        builder = CircuitBuilder()
        parsed_tree = parser.parse_to_circuit_tree(top_level_path)
        sim_circuit = builder.build_simulation_model(parsed_tree)
        pm = sim_circuit.parameter_manager

        freqs_hz = np.array([1e9, 2e9])
        results = pm.evaluate_all(freqs_hz)

        r1_res_qty = results["top.amp1.R1.resistance"]
        assert r1_res_qty.check("ohm")
        np.testing.assert_allclose(r1_res_qty.magnitude, [100.0, 100.0])

        l1_ind_qty = results["top.amp1.L1.inductance"]
        assert l1_ind_qty.check("henry")
        np.testing.assert_allclose(l1_ind_qty.to("nH").magnitude, [10.0, 10.0])

        c1_cap_qty = results["top.amp1.C1.capacitance"]
        assert c1_cap_qty.check("farad")
        expected_c1 = (2e-12) * (2.0)
        np.testing.assert_allclose(c1_cap_qty.to("farad").magnitude, [expected_c1, expected_c1])

        c2_cap_qty = results["top.amp1.C2.capacitance"]
        assert c2_cap_qty.check("farad")
        expected_c2_mag = 1e-21 * freqs_hz
        np.testing.assert_allclose(c2_cap_qty.to("farad").magnitude, expected_c2_mag)

    def test_numerical_literal_parameter_is_assigned_declared_dimension(self):
        """
        Edge Case Test: Verifies that a unitless numerical literal is correctly
        assigned the dimension declared by the component.
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
        pm.build(defs, {})
        results = pm.evaluate_all(np.array([1e9]))
        res_qty = results["top.R1.resistance"]

        assert isinstance(res_qty, Quantity)
        assert res_qty.check("ohm")
        np.testing.assert_allclose(res_qty.magnitude, 50.0)

    # REVISED: This test now correctly validates the constant evaluation path.
    def test_constant_expression_error_is_caught_during_build(self):
        """
        Unit Safety Test: Verifies that an error in a *constant* expression is
        caught during the `pm.build()` phase due to eager evaluation.
        """
        pm = ParameterManager()
        defs = [
            ParameterDefinition(
                owner_fqn="top", base_name="length", raw_value_or_expression_str="5 meter",
                source_yaml_path=Path("."), declared_dimension_str="meter"
            ),
            ParameterDefinition(
                owner_fqn="top", base_name="bad_log", raw_value_or_expression_str="log(length)",
                source_yaml_path=Path("."), declared_dimension_str="dimensionless"
            )
        ]
        scopes = {"top.bad_log": ChainMap({"length": "top.length"}), "top.length": ChainMap({})}
        
        with pytest.raises(ParameterEvaluationError) as excinfo:
            pm.build(defs, scopes)

        assert isinstance(excinfo.value.__cause__, pint.DimensionalityError)
        report = excinfo.value.get_diagnostic_report()
        assert "Parameter Evaluation Error" in report
        assert "FQN:            top.bad_log" in report
        assert "log' is not defined for quantities with dimensions" in str(excinfo.value).lower()
        
    # NEW: This test correctly validates the dynamic evaluation path.
    def test_dynamic_expression_error_is_caught_during_evaluation(self):
        """
        Unit Safety Test: Verifies that an error in a *frequency-dependent*
        expression is caught during the `pm.evaluate_all()` phase.
        """
        pm = ParameterManager()
        defs = [
            ParameterDefinition(
                owner_fqn="top", base_name="bad_log_of_freq", raw_value_or_expression_str="log(freq)",
                source_yaml_path=Path("."), declared_dimension_str="dimensionless"
            )
        ]
        pm.build(defs, {"top.bad_log_of_freq": ChainMap({})})
        
        with pytest.raises(ParameterEvaluationError) as excinfo:
            pm.evaluate_all(np.array([1e9]))

        assert isinstance(excinfo.value.__cause__, pint.DimensionalityError)
        report = excinfo.value.get_diagnostic_report()
        assert "Parameter Evaluation Error" in report
        assert "FQN:            top.bad_log_of_freq" in report


    # =========================================================================
    # === Test Group 3: Error Diagnostics & Negative Testing
    # =========================================================================

    def test_unresolved_symbol_raises_diagnosable_error(self, param_test_netlists):
        """Diagnostic Test: Verifies a clear diagnostic for an unresolved symbol."""
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

    def test_circular_dependency_raises_diagnosable_error(self, param_test_netlists):
        """Diagnostic Test: Verifies a clear diagnostic for a circular dependency."""
        netlist_path = param_test_netlists / "circular_dependency.yaml"
        parser = NetlistParser()
        builder = CircuitBuilder()
        parsed_tree = parser.parse_to_circuit_tree(netlist_path)

        with pytest.raises(CircuitBuildError) as excinfo:
            builder.build_simulation_model(parsed_tree)

        report = str(excinfo.value)
        assert "Actionable Diagnostic Report" in report
        assert "Error Type:     Circular Parameter Dependency" in report
        assert "top.param_A" in report and "top.param_B" in report and "top.param_C" in report
        assert "->" in report

    def test_incompatible_unit_arithmetic_is_diagnosed(self, param_test_netlists):
        """
        CRITICAL ARCHITECTURAL VERIFICATION: Proves the `eval`-based architecture is
        intrinsically unit-safe.
        """
        parser = NetlistParser()
        builder = CircuitBuilder()
        parsed_tree = parser.parse_to_circuit_tree(param_test_netlists / "dimensional_error.yaml")
        
        with pytest.raises(CircuitBuildError) as excinfo:
            builder.build_simulation_model(parsed_tree)

        report = str(excinfo.value)
        assert "Actionable Diagnostic Report" in report
        assert "Error Type:     Parameter Evaluation Error" in report
        assert "FQN:            top.bad_param" in report
        report_lower = report.lower()
        assert "cannot convert" in report_lower
        assert "ohm" in report_lower and "farad" in report_lower
        assert isinstance(excinfo.value.__cause__, ParameterEvaluationError)
        assert isinstance(excinfo.value.__cause__.__cause__, pint.DimensionalityError)
        
    # REVISED: This test now uses a valid stimulus and asserts the correct error type.
    def test_undefined_function_raises_diagnosable_error(self, param_test_netlists):
        """
        Diagnostic Test: Verifies a clear diagnostic for using a function not defined
        in the numerical evaluation scope.
        """
        netlist_path = param_test_netlists / "disallowed_function.yaml"
        parser = NetlistParser()
        builder = CircuitBuilder()
        parsed_tree = parser.parse_to_circuit_tree(netlist_path)

        with pytest.raises(CircuitBuildError) as excinfo:
            builder.build_simulation_model(parsed_tree)

        report = str(excinfo.value)
        assert "Actionable Diagnostic Report" in report
        assert "Error Type:     Parameter Evaluation Error" in report
        assert "FQN:            top.R1.resistance" in report
        assert "name 'my_unsupported_function' is not defined" in report
        
    # REVISED: This test now correctly asserts the GENERIC report for a holistic error.
    def test_holistic_evaluation_error_has_generic_diagnostics(self):
        """
        Diagnostic Test: Verifies that a holistic evaluation error (like a
        dimensionality mismatch) provides a clear, generic report when per-index
        information is not applicable.
        """
        pm = ParameterManager()
        defs = [ParameterDefinition(
            owner_fqn="top", base_name="bad", 
            raw_value_or_expression_str="1/freq", 
            source_yaml_path=Path("."), declared_dimension_str="ohm" # Intentionally wrong dimension
        )]
        pm.build(defs, {"top.bad": ChainMap({})})

        freqs = np.array([1e9, 2e9])
        with pytest.raises(ParameterEvaluationError) as excinfo:
            pm.evaluate_all(freqs)

        report = excinfo.value.get_diagnostic_report()
        assert "Actionable Diagnostic Report" in report
        assert "Parameter Evaluation Error" in report
        assert "FQN:            top.bad" in report
        report_lower = report.lower()
        
        assert "cannot convert" in report_lower
        assert "hertz" in report_lower
        assert "ohm" in report_lower
        
        # This is the key change: We assert that the detailed per-index string is NOT present.
        assert "the error first occurred at sweep index" not in report_lower

    # NEW: This test validates the detailed, per-index diagnostic reporting.
    def test_numerical_error_at_specific_frequency_has_diagnostics(self):
        """
        Diagnostic Test: Verifies detailed, per-index diagnostics for a numerical
        runtime error (e.g., division by zero). This validates the enhanced error
        handling logic that finds specific failure points.
        """
        pm = ParameterManager()
        defs = [ParameterDefinition(
            owner_fqn="top", base_name="bad",
            # This expression will cause a ZeroDivisionError at freq = 1e9 Hz
            raw_value_or_expression_str="1 / (freq - 1e9 Hz)",
            source_yaml_path=Path("."), declared_dimension_str="second" # Correct dimension
        )]
        pm.build(defs, {"top.bad": ChainMap({})})

        # The frequency sweep must include the point of failure.
        freqs = np.array([0.5e9, 1e9, 1.5e9])
        
        with pytest.raises(ParameterEvaluationError) as excinfo:
            pm.evaluate_all(freqs)

        report = excinfo.value.get_diagnostic_report()
        assert "Actionable Diagnostic Report" in report
        assert "Parameter Evaluation Error" in report
        assert "FQN:            top.bad" in report
        assert "Expression resulted in a non-finite value (inf)" in report

        # This is the critical assertion for the detailed report.
        assert "The error first occurred at sweep index 1 (frequency = 1.0000e+09 Hz)" in report
        # Verify it reports the correct number of failing points.
        assert "This error occurred at 1 total frequency points." in report