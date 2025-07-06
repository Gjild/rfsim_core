# tests/test_parameters/test_parameters.py

"""
Definitive Validation Suite for the RFSim Core Parameter Evaluation Architecture.

**Architectural Alignment (Post-Phase 10):**
This test suite has been fundamentally refactored to validate the new, unified,
`eval`-based parameter evaluation pipeline. It no longer tests the obsolete
`ast`/`sympy` parsing path. Its primary purpose is to serve as the ultimate
gatekeeper for the new architecture, validating its three foundational pillars:

1.  **A Single, Consistent Language:** Verifying that all parameter expressions, which
    must now be syntactically valid Python, are correctly evaluated. This includes
    the new requirement for explicit `Quantity()` construction for literals with units.

2.  **Intrinsic Unit Safety & Graph-Based Constant Validation:** Proving that the `eval()`-based
    pipeline makes dimensionally-incompatible operations impossible and that errors in
    constant-valued expressions are caught and diagnosed at build time via a robust,
    graph-based dependency analysis.

3.  **Actionable Diagnostics:** Ensuring that every conceivable user error—from `SyntaxError`
    and `NameError` (unresolved symbol) to `pint.DimensionalityError` and runtime
    `ZeroDivisionError`—is caught and reported with a clear, user-friendly, and
    diagnosable error report that guides the user to a solution.

The tests are organized into logical groups that build upon each other, starting
with the internal build process, moving to correct evaluation, and finally,
validating the robustness of the error handling. This file is the primary
guarantee that the new, simplified parameter system is correct and robust.
"""

import pytest
from pathlib import Path
import numpy as np
import pint

# --- Core RFSim Imports for Testing ---
from rfsim_core.parameters import (
    ParameterManager,
    ParameterDefinition,
    ParameterEvaluationError,
    CircularParameterDependencyError,
)
from rfsim_core.errors import CircuitBuildError
from rfsim_core.circuit_builder import CircuitBuilder
from rfsim_core.parser import NetlistParser
from rfsim_core.units import ureg, Quantity


def create_param_netlists(tmp_path: Path) -> Path:
    """
    Helper fixture to create a standard set of YAML files for parameter tests.
    This version is REVISED to use the new, `eval`-safe syntax where unit literals
    are explicitly constructed with `Quantity()`.
    """
    netlist_dir = tmp_path / "param_netlists"
    netlist_dir.mkdir()

    # --- Netlist 1: A comprehensive hierarchical design for "happy path" testing ---
    (netlist_dir / "hierarchical_params.yaml").write_text("""
circuit_name: TopCircuit
ground_net: gnd
parameters:
  top_gain: 2.0 # Simple dimensionless literal is OK.
  # --- START OF FIX: Use the explicit dictionary format for parameters with dimensions. ---
  override_cap_value: {expression: "Quantity('2 pF')", dimension: "farad"}
  # --- END OF FIX ---
components:
  - id: amp1
    type: Subcircuit
    definition_file: ./sub_amp.yaml
    ports: {IN: p_in, OUT: p_out}
    parameters:
      gain_in: top_gain
      base_cap: override_cap_value
      L1.inductance: "Quantity('10 nH')"
ports:
  - {id: p_in, reference_impedance: "Quantity('50 ohm')"}
  - {id: p_out, reference_impedance: "Quantity('50 ohm')"}
""")
    # --- START OF FIX: Apply the same explicit format to all test netlists. ---
    (netlist_dir / "sub_amp.yaml").write_text("""
circuit_name: SubAmp
ground_net: gnd
parameters:
  gain_in: 1.0
  base_cap: {expression: "Quantity('1 pF')", dimension: "farad"}
  C_slope: {expression: "Quantity('1e-21 F/Hz')", dimension: "farad / hertz"}
  R_base: {expression: "Quantity('50 ohm')", dimension: "ohm"}
  
  # --- START OF THE FIX ---
  # Any derived parameter that has units MUST declare its expected dimension.
  derived_capacitance: {expression: "base_cap * gain_in", dimension: "farad"}
  # --- END OF THE FIX ---

components:
  - id: R1
    type: Resistor
    ports: {0: IN, 1: gnd}
    # This expression is now fine because R_base has a declared dimension.
    parameters: {resistance: "R_base * gain_in"}
  - id: L1
    type: Inductor
    ports: {0: IN, 1: OUT}
    parameters: {inductance: "Quantity('1 nH')"}
  - id: C1
    type: Capacitor
    ports: {0: OUT, 1: gnd}
    # This is now fine because derived_capacitance has a declared dimension.
    parameters: {capacitance: derived_capacitance}
  - id: C2
    type: Capacitor
    ports: {0: IN, 1: gnd}
    # This expression containing 'freq' is a dynamic parameter, so it is handled
    # by a different part of the evaluation logic. Its dimensionality is checked
    # against the component's declared parameter dimension ('farad').
    parameters: {capacitance: "C_slope * freq"}
ports:
  - {id: IN, reference_impedance: "Quantity('50 ohm')"}
  - {id: OUT, reference_impedance: "Quantity('50 ohm')"}
""")

    # --- Error-case netlists ---
    (netlist_dir / "unresolved_symbol.yaml").write_text("""
circuit_name: UnresolvedSymbol
components:
  - id: R1
    type: Resistor
    ports: {0: in, 1: out}
    parameters:
      resistance: "Quantity('50 ohm') * undefined_gain"
""")

    (netlist_dir / "circular_dependency.yaml").write_text("""
circuit_name: CircularDep
parameters:
  param_A: "param_B * 2"
  param_B: "param_C / 3"
  param_C: "param_A + 1"
components:
  - {id: R1, type: Resistor, ports: {0: in, 1: gnd}, parameters: {resistance: "Quantity('1 ohm')"}}
""")

    (netlist_dir / "undefined_function.yaml").write_text("""
circuit_name: UndefinedFunc
components:
  - id: R1
    type: Resistor
    ports: {0: in, 1: gnd}
    parameters: {resistance: "my_unsupported_function(1) * Quantity('1 ohm')"}
""")

    (netlist_dir / "dimensional_error.yaml").write_text("""
circuit_name: DimensionalError
parameters:
  my_res: {expression: "Quantity('50 ohm')", dimension: "ohm"}
  my_cap: {expression: "Quantity('1 pF')", dimension: "farad"}
  bad_param: "my_res + my_cap" # This will now be correctly reached and fail.
components:
  - {id: R1, type: Resistor, ports: {0: in, 1: gnd}, parameters: {resistance: bad_param}}
""")
    # --- END OF FIX ---

    # REVISED: This netlist now tests a valid dimensional expression that has a numerical error.
    (netlist_dir / "build_time_eval_error.yaml").write_text("""
circuit_name: BuildTimeError
parameters:
  # This expression is dimensionally valid (log of dimensionless) but mathematically invalid.
  # It is a constant expression, so it MUST be caught during the build process.
  invalid_param: "np.log(-1.0)"
components:
  - id: R1
    type: Resistor
    ports: {0: in, 1: gnd}
    parameters: {resistance: "Quantity('1 ohm')"}
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
        assert "freq" not in order

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

    def test_constant_expression_error_is_caught_during_build(self, param_test_netlists):
        """
        Unit Safety Test (REVISED): Verifies that a mathematically invalid operation in
        a *constant* expression is caught during the `CircuitBuilder.build()` phase.
        """
        parser = NetlistParser()
        builder = CircuitBuilder()
        parsed_tree = parser.parse_to_circuit_tree(param_test_netlists / "build_time_eval_error.yaml")

        with pytest.raises(CircuitBuildError) as excinfo:
            builder.build_simulation_model(parsed_tree)

        report = str(excinfo.value)
        assert "Actionable Diagnostic Report" in report
        assert "Error Type:     Parameter Evaluation Error" in report
        assert "FQN:            top.invalid_param" in report
        
        report_lower = report.lower()
        assert "non-finite value" in report_lower
        
        # Assert on the exception chain for completeness
        assert isinstance(excinfo.value.__cause__, ParameterEvaluationError)
        assert isinstance(excinfo.value.__cause__.__cause__, ValueError)

    def test_dynamic_expression_error_is_caught_during_evaluation(self):
        pm = ParameterManager()
        defs = [
            ParameterDefinition(
                owner_fqn="top", base_name="bad_log_of_freq", raw_value_or_expression_str="np.log(freq)",
                source_yaml_path=Path("."), declared_dimension_str="dimensionless"
            )
        ]
        pm.build(defs, {})

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
        parser = NetlistParser()
        builder = CircuitBuilder()
        parsed_tree = parser.parse_to_circuit_tree(param_test_netlists / "unresolved_symbol.yaml")

        with pytest.raises(CircuitBuildError) as excinfo:
            builder.build_simulation_model(parsed_tree)

        report = str(excinfo.value)
        assert "Actionable Diagnostic Report" in report
        assert "Error Type:     Parameter Evaluation Error" in report
        assert "FQN:            top.R1.resistance" in report
        assert "name 'undefined_gain' is not defined" in report

    def test_circular_dependency_raises_diagnosable_error(self, param_test_netlists):
        """Diagnostic Test (REVISED): Verifies a clear diagnostic for a circular dependency."""
        netlist_path = param_test_netlists / "circular_dependency.yaml"
        parser = NetlistParser()
        builder = CircuitBuilder()
        parsed_tree = parser.parse_to_circuit_tree(netlist_path)

        with pytest.raises(CircuitBuildError) as excinfo:
            builder.build_simulation_model(parsed_tree)

        report = str(excinfo.value)
        assert "Actionable Diagnostic Report" in report
        assert "Error Type:     Circular Parameter Dependency" in report
        # Make the test robust against the order of the cycle.
        assert "top.param_A" in report
        assert "top.param_B" in report
        assert "top.param_C" in report

    def test_incompatible_unit_arithmetic_is_diagnosed(self, param_test_netlists):
        parser = NetlistParser()
        builder = CircuitBuilder()
        parsed_tree = parser.parse_to_circuit_tree(param_test_netlists / "dimensional_error.yaml")

        with pytest.raises(CircuitBuildError) as excinfo:
            builder.build_simulation_model(parsed_tree)

        report = str(excinfo.value)
        assert "Actionable Diagnostic Report" in report
        assert "Error Type:     Parameter Evaluation Error" in report
        # The build will fail on the correct parameter now.
        assert "FQN:            top.bad_param" in report
        report_lower = report.lower()
        assert "cannot convert" in report_lower
        assert "ohm" in report_lower and "farad" in report_lower
        assert isinstance(excinfo.value.__cause__, ParameterEvaluationError)
        assert isinstance(excinfo.value.__cause__.__cause__, pint.DimensionalityError)

    def test_undefined_function_raises_diagnosable_error(self, param_test_netlists):
        netlist_path = param_test_netlists / "undefined_function.yaml"
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

    def test_holistic_evaluation_error_has_generic_diagnostics(self):
        pm = ParameterManager()
        defs = [ParameterDefinition(
            owner_fqn="top", base_name="bad",
            raw_value_or_expression_str="1/freq",
            source_yaml_path=Path("."), declared_dimension_str="ohm"
        )]
        pm.build(defs, {})

        freqs = np.array([1e9, 2e9])
        with pytest.raises(ParameterEvaluationError) as excinfo:
            pm.evaluate_all(freqs)

        report = excinfo.value.get_diagnostic_report()
        assert "Actionable Diagnostic Report" in report
        assert "Parameter Evaluation Error" in report
        assert "FQN:            top.bad" in report
        report_lower = report.lower()
        assert "cannot convert" in report_lower
        assert "1 / hertz" in report_lower
        assert "ohm" in report_lower
        assert "the error first occurred at sweep index" not in report_lower

    def test_numerical_error_at_specific_frequency_has_diagnostics(self):
        pm = ParameterManager()
        defs = [ParameterDefinition(
            owner_fqn="top", base_name="bad",
            raw_value_or_expression_str="1 / (freq - Quantity('1e9 Hz'))",
            source_yaml_path=Path("."), declared_dimension_str="second"
        )]
        
        pm.build(defs, {})

        freqs = np.array([0.5e9, 1e9, 1.5e9])
        with pytest.raises(ParameterEvaluationError) as excinfo:
            pm.evaluate_all(freqs)

        report = excinfo.value.get_diagnostic_report()
        assert "Actionable Diagnostic Report" in report
        assert "Parameter Evaluation Error" in report
        assert "FQN:            top.bad" in report
        assert "Expression resulted in a non-finite value (inf)" in report
        assert "The error first occurred at sweep index 1 (frequency = 1.0000e+09 Hz)" in report
        assert "This error occurred at 1 total frequency point." in report