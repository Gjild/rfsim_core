# tests/components/test_components.py

import pytest
import numpy as np
import pint

from rfsim_core.units import ureg, Quantity, ADMITTANCE_DIMENSIONALITY
from rfsim_core.constants import LARGE_ADMITTANCE_SIEMENS
from rfsim_core.data_structures import Circuit
from rfsim_core.parser import NetlistParser
from rfsim_core.circuit_builder import CircuitBuilder
from rfsim_core.cache import SimulationCache

# --- Foundational Imports for the New Architecture ---
from rfsim_core.components.base import ComponentBase, DCBehaviorType
from rfsim_core.components.capabilities import IMnaContributor, IDcContributor, ComponentCapability
from rfsim_core.components.elements import Resistor, Capacitor, Inductor
from rfsim_core.components.subcircuit import SubcircuitInstance
# Correctly import the single, canonical ComponentError
from rfsim_core.components.exceptions import ComponentError

# --- Imports for Testing the Simulation Executive's Behavior ---
from rfsim_core.simulation.context import SimulationContext
from rfsim_core.simulation.engine import SimulationEngine
from rfsim_core.simulation.exceptions import SingleLevelSimulationFailure

# --- Import Fixtures ---
# conftest.py contains the test harness and definitions for malicious components
from conftest import InheritingComponent, TwoPortBase


class TestFrameworkContractEnforcement:
    """
    Verifies the simulation framework correctly enforces its API
    contracts on all components, especially malicious or incompetent plugins.
    """

    def test_stamping_unpopulated_subcircuit_raises_diagnosable_error(self, tmp_path):
        """
        Verifies that attempting to stamp a subcircuit whose cache has not been
        populated by the simulation executive results in a clear, diagnosable error.
        """
        # 1. Create a valid hierarchical netlist
        (tmp_path / "top.yaml").write_text("""
        circuit_name: Top
        components:
          - id: X1
            type: Subcircuit
            definition_file: ./sub.yaml
            ports: {p1: net1, p2: gnd}
        ports:
          - {id: net1, reference_impedance: "50 ohm"}
        """)
        (tmp_path / "sub.yaml").write_text("""
        circuit_name: Sub
        components:
          - id: R1
            type: Resistor
            ports: {0: p1, 1: p2}
            parameters: {resistance: 50.0}
        ports:
          - {id: p1, reference_impedance: "50 ohm"}
          - {id: p2, reference_impedance: "50 ohm"}
        """)

        # 2. Build the circuit model
        parser = NetlistParser()
        builder = CircuitBuilder()
        parsed = parser.parse_to_circuit_tree(tmp_path / "top.yaml")
        circuit = builder.build_simulation_model(parsed)
        sub_instance = circuit.sim_components['X1']

        # 3. Manually ensure the subcircuit's cache is NOT populated
        assert isinstance(sub_instance, SubcircuitInstance)
        assert sub_instance.cached_y_parameters_ac is None

        # 4. Directly invoke the simulation engine's single-level run
        freqs = np.array([1e9])
        cache = SimulationCache()
        context = SimulationContext(circuit, freqs, cache)
        engine = SimulationEngine(context)

        # 5. Assert that the simulation fails with a SingleLevelSimulationFailure,
        #    which is the error contract for the engine's single-level run method.
        with pytest.raises(SingleLevelSimulationFailure) as excinfo:
            # --- FIX: Call the correct method to create the intended failure state ---
            # Directly call the single-level simulation on the top circuit. This bypasses
            # the recursive cache population step, ensuring X1's cache is empty, which
            # is the condition this test is designed to verify.
            engine._run_single_level_simulation(circuit)

        # 6. Deeply inspect the error chain to verify the root cause
        # The _run_single_level_simulation correctly wraps the ComponentError in a
        # SingleLevelSimulationFailure. We inspect the original error.
        cause = excinfo.value.original_error
        assert isinstance(cause, ComponentError), "The root cause should be a ComponentError"
        assert cause.component_fqn == "top.X1"
        # The first capability to be called in the flow is the DC contributor.
        assert "DC analysis results cache was not populated" in cause.details

        # Verify the top-level report is correctly formatted
        report = excinfo.value.get_diagnostic_report()
        assert "Simulation Failure in Hierarchical Context" in report
        assert "--- Details of the Root Cause ---" in report
        assert "Component Simulation Error" in report
        # The FQN of the subcircuit where the error occurred should be present
        assert "FQN:            top.X1" in report

    def test_framework_rejects_non_quantity_return(self, run_bad_comp_test_harness):
        """
        Framework rejects MNA contributions that are not pint.Quantity objects.
        """
        component_yaml = """
          - id: U1
            type: BadComponentReturnsNumber
            ports: {0: p1, 1: gnd}
        """
        cause = run_bad_comp_test_harness(component_yaml)
        assert isinstance(cause, ComponentError)
        assert cause.component_fqn == "top.U1"
        # Assert on the new, more precise error message from the framework's validation
        assert "is not a pint.Quantity object" in cause.details
        assert "but type 'ndarray'" in cause.details


    def test_framework_rejects_wrong_dimension_return(self, run_bad_comp_test_harness):
        """
        Framework rejects MNA contributions with incorrect physical dimensions.
        """
        component_yaml = """
          - id: U1
            type: BadComponentReturnsWrongDimension
            ports: {0: p1, 1: gnd}
        """
        cause = run_bad_comp_test_harness(component_yaml)
        assert isinstance(cause, ComponentError)
        assert isinstance(cause.__cause__, pint.DimensionalityError)
        assert cause.component_fqn == "top.U1"
        # --- FIX: Make the assertion robust to pint's string formatting ---
        # Check that the error message contains the expected dimension and a key part
        # of the base-unit representation of the incorrect dimension ([impedance]).
        assert "Expected [admittance]" in cause.details
        assert "got [mass] * [length] ** 2" in cause.details

    def test_framework_rejects_mismatched_stamp_shape(self, run_bad_comp_test_harness):
        """
        Framework rejects MNA contributions with incorrect shapes (non-vectorized).
        """
        component_yaml = """
          - id: U1
            type: BadComponentReturnsWrongShape
            ports: {0: p1, 1: gnd}
        """
        cause = run_bad_comp_test_harness(component_yaml)
        assert isinstance(cause, ComponentError)
        # --- FIX: Assert on the correct root cause and error message ---
        # The true root cause is an IndexError when the assembler tries to slice the
        # non-vectorized array.
        assert isinstance(cause.__cause__, IndexError)
        assert cause.component_fqn == "top.U1"
        # Check that the details string contains the relevant information.
        assert "Stamping failed" in cause.details
        assert "too many indices for array" in cause.details


    def test_framework_wraps_unexpected_component_exceptions(self, run_bad_comp_test_harness):
        """
        Framework wraps unexpected exceptions from component capabilities.
        """
        component_yaml = """
          - id: U1
            type: BadComponentRaisesRandomError
            ports: {0: p1, 1: gnd}
        """
        cause = run_bad_comp_test_harness(component_yaml)
        assert isinstance(cause, ComponentError)
        assert isinstance(cause.__cause__, ValueError)
        assert str(cause.__cause__) == "Something went wrong inside the component!"
        assert cause.component_fqn == "top.U1"
        assert "Something went wrong" in cause.details


class TestCapabilitySystemFramework:
    """
    Verifies the core mechanics of the capability discovery and querying system.
    """

    def test_declare_capabilities_discovers_correctly(self):
        """
        Verifies ComponentBase.declare_capabilities finds all nested classes with @provides.
        """
        caps = Resistor.declare_capabilities()
        assert IMnaContributor in caps
        assert IDcContributor in caps
        assert caps[IMnaContributor] is Resistor.MnaContributor
        assert caps[IDcContributor] is Resistor.DcContributor
        assert len(caps) == 2

    def test_capability_discovery_respects_inheritance(self):
        """
        Verifies capability discovery correctly inspects the MRO and finds parent capabilities.
        """
        caps = InheritingComponent.declare_capabilities()
        assert IMnaContributor in caps
        assert IDcContributor in caps
        assert caps[IMnaContributor] is InheritingComponent.MnaContributor
        assert caps[IDcContributor] is TwoPortBase.DcContributor

    def test_get_capability_returns_instance_and_is_cached(self, tmp_path):
        """
        Verifies get_capability returns a valid instance and that subsequent
        calls return the exact same object, proving the instance-level cache works.
        """
        (tmp_path / "r_test.yaml").write_text("""
        circuit_name: CapabilityTest
        components:
          - id: R1
            type: Resistor
            ports: {0: p1, 1: gnd}
            parameters: {resistance: 50.0}
        """)
        parser = NetlistParser()
        builder = CircuitBuilder()
        circuit = builder.build_simulation_model(parser.parse_to_circuit_tree(tmp_path / "r_test.yaml"))
        resistor_instance = circuit.sim_components['R1']

        mna_cap_1 = resistor_instance.get_capability(IMnaContributor)
        assert isinstance(mna_cap_1, Resistor.MnaContributor)

        mna_cap_2 = resistor_instance.get_capability(IMnaContributor)
        assert id(mna_cap_1) == id(mna_cap_2)

        class IUnsupportedCapability(ComponentCapability): pass
        assert resistor_instance.get_capability(IUnsupportedCapability) is None


class TestLeafElementInternalContracts:
    """
    Verifies that leaf elements correctly enforce their own physical constraints.
    """

    @pytest.fixture
    def mock_component(self, mock_ir_data):
        """
        Creates a mock ComponentBase instance to pass as context. This is a concrete
        class that implements all abstract methods, making it instantiable and robust.
        """
        class MockComponent(ComponentBase):
            def __init__(self):
                # The component_type is now set in the base class init
                super().__init__(
                    instance_id="comp",
                    parameter_manager=None,
                    parent_hierarchical_id="mock",
                    raw_ir_data=mock_ir_data
                )
            @classmethod
            def declare_parameters(cls): return {"value": "ohm"}
            @classmethod
            def declare_ports(cls): return [0, 1]
            def is_structurally_open(self, resolved_constant_params): return False
        return MockComponent()

    @pytest.mark.parametrize(
        "element_class, param_name, param_dim, invalid_value, expected_msg",
        [
            # --- Resistor Test Cases ---
            (Resistor, "resistance", "ohm", -50 * ureg.ohm, "non-negative"),
            (Resistor, "resistance", "ohm", (50+1j) * ureg.ohm, "must be real"),

            # --- Capacitor Test Cases ---
            (Capacitor, "capacitance", "farad", -1 * ureg.pF, "non-negative"),
            (Capacitor, "capacitance", "farad", (1+1j) * ureg.pF, "must be real"),

            # --- Inductor Test Cases ---
            (Inductor, "inductance", "henry", -1 * ureg.nH, "non-negative"),
            (Inductor, "inductance", "henry", (1+1j) * ureg.nH, "must be real"),
        ]
    )
    def test_leaf_element_capability_rejects_invalid_values(
            self, mock_component, element_class, param_name, param_dim, invalid_value, expected_msg
        ):
        """
        Verifies that MNA capability methods correctly raise a diagnosable ComponentError
        when provided with parameters that violate fundamental physical constraints.
        """
        # ARRANGE: Set up the component, its parameters, and test conditions.
        mna_contributor = element_class.MnaContributor()
        freqs = np.array([1e9])

        # This approach preserves the real or complex nature of the original test value,
        # ensuring the correct validation path is triggered in the code under test.
        value_mag = np.asarray(invalid_value.magnitude)
        vectorized_mag = np.full(freqs.shape, value_mag, dtype=value_mag.dtype)
        vectorized_invalid_value = Quantity(vectorized_mag, invalid_value.units)

        # Mock the component's contract to declare the correct parameter name AND dimension.
        type(mock_component).declare_parameters = classmethod(
            lambda cls: {param_name: param_dim}
        )
        assert mock_component.parameter_fqns[0] == f"mock.comp.{param_name}"

        params = {mock_component.parameter_fqns[0]: vectorized_invalid_value}

        # ACT & ASSERT
        with pytest.raises(ComponentError) as excinfo:
            mna_contributor.get_mna_stamps(mock_component, freqs, params)

        # Assert that the correct error was raised by checking its message.
        assert expected_msg in excinfo.value.details


class TestLeafElementCapabilities:
    """
    Verifies the numerical and dimensional correctness of the capabilities for each leaf element.
    """

    @pytest.fixture
    def freqs(self):
        return np.array([1e9, 2e9])

    @pytest.fixture
    def mock_r(self, mock_ir_data):
        class MockR(Resistor):
             def __init__(self): super().__init__(instance_id="R1", parent_hierarchical_id="mock", parameter_manager=None, raw_ir_data=mock_ir_data)
        return MockR()

    @pytest.fixture
    def mock_l(self, mock_ir_data):
        class MockL(Inductor):
            def __init__(self): super().__init__(instance_id="L1", parent_hierarchical_id="mock", parameter_manager=None, raw_ir_data=mock_ir_data)
        return MockL()

    @pytest.fixture
    def mock_c(self, mock_ir_data):
        class MockC(Capacitor):
            def __init__(self): super().__init__(instance_id="C1", parent_hierarchical_id="mock", parameter_manager=None, raw_ir_data=mock_ir_data)
        return MockC()

    def test_resistor_mna_contribution(self, mock_r, freqs):
        r_val = Quantity(np.full_like(freqs, 50.0), "ohm")
        params = {mock_r.parameter_fqns[0]: r_val}
        stamps = mock_r.get_capability(IMnaContributor).get_mna_stamps(mock_r, freqs, params)

        assert len(stamps) == 1
        y_qty, _ = stamps[0]

        assert isinstance(y_qty, Quantity)
        assert y_qty.dimensionality == ADMITTANCE_DIMENSIONALITY
        # Make the shape assertion more robust by tying it to the input frequencies
        assert y_qty.shape == (len(freqs), 2, 2)

        expected_y = 1.0 / 50.0
        base_stamp = np.array([[expected_y, -expected_y], [-expected_y, expected_y]])
        expected_stamp_vectorized = np.array([base_stamp] * len(freqs))
        np.testing.assert_allclose(y_qty.magnitude, expected_stamp_vectorized)
        
    @pytest.mark.parametrize("r_val, expected_y", [
        (0 * ureg.ohm, LARGE_ADMITTANCE_SIEMENS),
        (np.inf * ureg.ohm, 0.0),
    ])
    def test_resistor_mna_ideal_boundaries(self, mock_r, r_val, expected_y):
        freqs = np.array([1e9])
        vectorized_r_val = Quantity(np.full_like(freqs, r_val.magnitude), r_val.units)
        params = {mock_r.parameter_fqns[0]: vectorized_r_val}

        stamps = mock_r.get_capability(IMnaContributor).get_mna_stamps(mock_r, freqs, params)
        y_qty, _ = stamps[0]

        expected_stamp = np.array([[expected_y, -expected_y], [-expected_y, expected_y]])
        np.testing.assert_allclose(y_qty.to("siemens").magnitude[0], expected_stamp, rtol=1e-9)

    @pytest.mark.parametrize("r_val, expected_type, expected_y_val", [
        (0 * ureg.ohm, DCBehaviorType.SHORT_CIRCUIT, None),
        (np.inf * ureg.ohm, DCBehaviorType.OPEN_CIRCUIT, None),
        (50 * ureg.ohm, DCBehaviorType.ADMITTANCE, 1.0/50.0),
    ])
    def test_resistor_dc_contribution(self, mock_r, r_val, expected_type, expected_y_val):
        params = {mock_r.parameter_fqns[0]: Quantity(np.array([r_val.magnitude]), r_val.units)}
        b_type, y_qty = mock_r.get_capability(IDcContributor).get_dc_behavior(mock_r, params)

        assert b_type == expected_type
        if expected_y_val is not None:
            assert y_qty is not None
            assert y_qty.dimensionality == ADMITTANCE_DIMENSIONALITY
            np.testing.assert_allclose(y_qty.to("siemens").magnitude, expected_y_val)
        else:
            assert y_qty is None

    def test_inductor_mna_contribution(self, mock_l, freqs):
        l_val = Quantity(10e-9, "henry")
        params = {mock_l.parameter_fqns[0]: l_val}
        stamps = mock_l.get_capability(IMnaContributor).get_mna_stamps(mock_l, freqs, params)
        y_qty, _ = stamps[0]

        assert y_qty.dimensionality == ADMITTANCE_DIMENSIONALITY
        omega = 2 * np.pi * freqs
        expected_y_mag = 1.0 / (omega * l_val.to("H").magnitude)
        expected_stamps = -1j * np.array([[[y, -y], [-y, y]] for y in expected_y_mag])
        np.testing.assert_allclose(y_qty.to("S").magnitude, expected_stamps, rtol=1e-9)

    def test_capacitor_mna_contribution(self, mock_c, freqs):
        c_val = Quantity(1e-12, "farad")
        params = {mock_c.parameter_fqns[0]: c_val}
        stamps = mock_c.get_capability(IMnaContributor).get_mna_stamps(mock_c, freqs, params)
        y_qty, _ = stamps[0]

        assert y_qty.dimensionality == ADMITTANCE_DIMENSIONALITY
        omega = 2 * np.pi * freqs
        expected_y_mag = omega * c_val.to("F").magnitude
        expected_stamps = 1j * np.array([[[y, -y], [-y, y]] for y in expected_y_mag])
        np.testing.assert_allclose(y_qty.to("S").magnitude, expected_stamps, rtol=1e-9)

    @pytest.mark.parametrize("c_val, expected_type", [
        (0 * ureg.farad, DCBehaviorType.OPEN_CIRCUIT),
        (1e-12 * ureg.farad, DCBehaviorType.OPEN_CIRCUIT),
        (np.inf * ureg.farad, DCBehaviorType.SHORT_CIRCUIT),
    ])
    def test_capacitor_dc_contribution(self, mock_c, c_val, expected_type):
        params = {mock_c.parameter_fqns[0]: Quantity(np.array([c_val.magnitude]), c_val.units)}
        b_type, y_qty = mock_c.get_capability(IDcContributor).get_dc_behavior(mock_c, params)
        assert b_type == expected_type
        assert y_qty is None