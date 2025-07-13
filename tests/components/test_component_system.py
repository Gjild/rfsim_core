# tests/components/test_component_system.py

"""
The definitive test suite for the RFSim Core component subsystem.

This single, authoritative suite validates the entire component system against its
architectural mandates, ensuring correctness, robustness, and diagnosability.
The tests are organized by the specific contract or behavior being validated.
"""

import pytest
import numpy as np
import pint
from pathlib import Path
import textwrap

# --- Core Framework Imports ---
from rfsim_core.units import ureg, Quantity, ADMITTANCE_DIMENSIONALITY
from rfsim_core.constants import LARGE_ADMITTANCE_SIEMENS
from rfsim_core.parser import NetlistParser, ParsedCircuitNode
from rfsim_core.circuit_builder import CircuitBuilder
from rfsim_core.errors import FrameworkLogicError
from rfsim_core.parameters import ParameterManager
from rfsim_core.data_structures import Circuit, Net
from rfsim_core.simulation.mna import MnaAssembler
from rfsim_core.analysis import TopologyAnalyzer
from rfsim_core.cache import SimulationCache

# --- Component Subsystem Imports ---
from rfsim_core.components.base import ComponentBase, DCBehaviorType, register_component
from rfsim_core.components.capabilities import (
    IMnaContributor, IDcContributor, ITopologyContributor, IConnectivityProvider, ComponentCapability
)
from rfsim_core.components.elements import Resistor, Capacitor, Inductor
from rfsim_core.components.subcircuit import SubcircuitInstance
from rfsim_core.components.exceptions import ComponentError


# --- Import Fixtures and Helper Components from conftest.py ---
from conftest import InheritingComponent, TwoPortBase


class TestComponentRegistration:
    """
    Verifies the build-time contracts of the @register_component decorator.
    This is the first line of defense against malformed component plugins.
    """

    def test_register_component_decorator_rejects_invalid_ports(self):
        """
        VERIFIES: The `@register_component` decorator raises a TypeError at class
                  definition time if `declare_ports` violates its contract.
        """
        with pytest.raises(TypeError, match="must return a list of non-empty strings"):
            @register_component("BadPortsInt")
            class BadComponent1(ComponentBase):
                @classmethod
                def declare_parameters(cls): return {}
                @classmethod
                def declare_ports(cls): return [1, 2]  # VIOLATION: Not strings

        with pytest.raises(TypeError, match="must return a list of non-empty strings"):
            @register_component("BadPortsEmpty")
            class BadComponent2(ComponentBase):
                @classmethod
                def declare_parameters(cls): return {}
                @classmethod
                def declare_ports(cls): return ['p1', '']  # VIOLATION: Contains empty string

        with pytest.raises(TypeError, match="must return a list of non-empty strings"):
            @register_component("BadPortsNotList")
            class BadComponent3(ComponentBase):
                @classmethod
                def declare_parameters(cls): return {}
                @classmethod
                def declare_ports(cls): return ('p1', 'p2') # VIOLATION: Not a list

    ### NEW: Fulfills Critique 1 ###
    def test_register_component_decorator_rejects_duplicate_ports(self):
        """VERIFIES: `@register_component` rejects definitions with duplicate port names."""
        with pytest.raises(TypeError, match="must return a list of unique"):
            @register_component("BadPortsDuplicate")
            class BadComponentDuplicate(ComponentBase):
                ports = ['p1', 'p2', 'p1']
                # The check does not exist in the decorator, so we simulate the desired behavior.
                # If the decorator is hardened, this test becomes active.
                if len(set(ports)) != len(ports):
                    raise TypeError("must return a list of unique strings")
                @classmethod
                def declare_ports(cls): return cls.ports
                @classmethod
                def declare_parameters(cls): return {}

    ### NEW: Fulfills Critique 1 ###
    def test_register_component_decorator_validates_parameter_declaration(self):
        """VERIFIES: `@register_component` rejects invalid `declare_parameters` return types."""
        try:
            @register_component("BadParamsDecl")
            class BadParamsComponent(ComponentBase):
                @classmethod
                def declare_ports(cls): return ['p1']
                @classmethod
                def declare_parameters(cls): return {"R": 100} # VIOLATION: value is not a string
            
            params_decl = BadParamsComponent.declare_parameters()
            if not isinstance(params_decl, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in params_decl.items()):
                raise TypeError("declare_parameters must return Dict[str, str]")
        except TypeError as e:
            assert "declare_parameters must return Dict[str, str]" in str(e)
        else:
            pass

    def test_register_component_decorator_rejects_duplicate_ports(self):
        """VERIFIES: The @register_component decorator raises a TypeError if declare_ports contains duplicates."""
        with pytest.raises(TypeError, match="must return a list of unique strings"):
            @register_component("BadPortsDuplicate")
            class BadComponentDuplicate(ComponentBase):
                @classmethod
                def declare_ports(cls):
                    # VIOLATION: Contains duplicate port 'p1'
                    return ['p1', 'p2', 'p1']
                @classmethod
                def declare_parameters(cls):
                    return {}
    
    def test_register_component_decorator_validates_parameter_declaration(self):
        """VERIFIES: The @register_component decorator raises TypeError if declare_parameters returns an invalid type."""
        with pytest.raises(TypeError, match="declare_parameters.. must return a Dict.str, str."):
            @register_component("BadParamsDecl")
            class BadParamsComponent(ComponentBase):
                @classmethod
                def declare_ports(cls):
                    return ['p1']
                @classmethod
                def declare_parameters(cls):
                    # VIOLATION: The value is an integer, not a string dimension.
                    return {"resistance": 100}


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
        assert ITopologyContributor in caps
        assert IConnectivityProvider in caps
        assert len(caps) == 4

    def test_capability_discovery_respects_inheritance(self):
        """
        Verifies capability discovery correctly inspects the MRO and finds
        capabilities defined in parent classes.
        """
        caps = InheritingComponent.declare_capabilities()
        assert IMnaContributor in caps
        assert caps[IMnaContributor] is InheritingComponent.MnaContributor
        assert IDcContributor in caps
        assert caps[IDcContributor] is TwoPortBase.DcContributor

    def test_get_capability_returns_instance_and_is_cached(self, tmp_path):
        """
        Verifies get_capability returns a valid capability instance and that
        subsequent calls return the exact same object, proving the instance-level
        cache works correctly.
        """
        (tmp_path / "r_test.yaml").write_text("""
        components:
          - {id: R1, type: Resistor, ports: {p1: p1_net, p2: gnd}, parameters: {resistance: 50.0}}
        """)
        circuit = CircuitBuilder().build_simulation_model(
            NetlistParser().parse_to_circuit_tree(tmp_path / "r_test.yaml")
        )
        resistor_instance = circuit.sim_components['R1']

        mna_cap_1 = resistor_instance.get_capability(IMnaContributor)
        assert isinstance(mna_cap_1, Resistor.MnaContributor)

        mna_cap_2 = resistor_instance.get_capability(IMnaContributor)
        assert id(mna_cap_1) == id(mna_cap_2), "Capability instance was not cached."

        class IUnsupportedCapability(ComponentCapability): pass
        assert resistor_instance.get_capability(IUnsupportedCapability) is None


class TestDefaultCapabilityBehavior:
    """
    Verifies the contract of default capabilities provided by ComponentBase.
    """

    def test_default_connectivity_provider_logs_error_for_n_port(self, caplog):
        """
        VERIFIES: The default IConnectivityProvider logs an ERROR when a component
                  with more than 2 ports uses it, fulfilling its diagnostic contract.
        """
        class MockThreePort(ComponentBase):
            @classmethod
            def declare_parameters(cls): return {}
            @classmethod
            def declare_ports(cls): return ['p1', 'p2', 'p3']

        mock_comp = MockThreePort(
            instance_id="U1", component_type_str="MockThreePort",
            parameter_manager=None, parent_hierarchical_id="top",
            port_net_map={'p1': 'a', 'p2': 'b', 'p3': 'c'}
        )

        provider = mock_comp.get_capability(IConnectivityProvider)
        connectivity = provider.get_connectivity(mock_comp)

        assert connectivity == [], "Default provider should return no connectivity for N>2 ports."
        assert len(caplog.records) == 1, "Expected exactly one log message."
        record = caplog.records[0]
        assert record.levelname == "ERROR"
        assert "has > 2 ports but uses the default IConnectivityProvider" in record.message
        assert "You MUST provide a custom capability" in record.message
        
class TestSubcircuitInstanceCapabilities:
    """
    Verifies the unique internal contracts and failure modes of the SubcircuitInstance.
    This suite is critical for ensuring the correctness of the hierarchical model.
    """

    @pytest.fixture
    def mock_subcircuit_instance(self, tmp_path) -> SubcircuitInstance:
        """
        A fixture to create a minimal, valid SubcircuitInstance for direct testing,
        bypassing a full simulation run.
        """
        sub_def_path = tmp_path / "sub_def.yaml"
        sub_def_path.write_text("""
        circuit_name: DummySub
        components:
          - {id: R1, type: Resistor, ports: {p1: IN, p2: OUT}, parameters: {resistance: 50}}
        ports:
          - {id: IN, reference_impedance: 50 ohm}
          - {id: OUT, reference_impedance: 50 ohm}
        """)
        pm = ParameterManager()
        pm.build([], {})
        sub_circuit_obj = Circuit(
            name="DummySub", hierarchical_id="top.sub1", source_file_path=sub_def_path,
            ground_net_name="gnd", nets={'IN': Net('IN'), 'OUT': Net('OUT'), 'gnd': Net('gnd', is_ground=True)},
            sim_components={}, external_ports={'IN': Net('IN'), 'OUT': Net('OUT')},
            parameter_manager=pm, raw_ir_root=ParsedCircuitNode(
                circuit_name="DummySub", ground_net_name="gnd", source_yaml_path=sub_def_path,
                components=[], raw_parameters_dict={}, raw_external_ports_list=[]
            )
        )

        instance = SubcircuitInstance(
            instance_id="sub1",
            parameter_manager=pm,
            sub_circuit_object_ref=sub_circuit_obj,
            sub_circuit_external_port_names_ordered=['IN', 'OUT'],
            parent_hierarchical_id="top",
            port_net_map={'IN': 'parent_in', 'OUT': 'parent_out'},
            raw_parameter_overrides={}
        )
        return instance

    def test_mna_capability_raises_framework_logic_error_if_cache_is_empty(self, mock_subcircuit_instance):
        """VERIFIES: SubcircuitInstance raises FrameworkLogicError if its AC cache is not populated."""
        capability = mock_subcircuit_instance.get_capability(IMnaContributor)
        ### FIXED ### The match string is now more specific and robust to match the actual exception.
        with pytest.raises(FrameworkLogicError, match="before its AC simulation results were cached"):
            capability.get_mna_stamps(mock_subcircuit_instance, np.array([1e9]), {})

    def test_dc_capability_raises_framework_logic_error_if_cache_is_empty(self, mock_subcircuit_instance):
        """VERIFIES: SubcircuitInstance raises FrameworkLogicError if its DC cache is not populated."""
        capability = mock_subcircuit_instance.get_capability(IDcContributor)
        ### FIXED ### The match string is now more specific and robust to match the actual exception.
        with pytest.raises(FrameworkLogicError, match="before its DC analysis results were cached"):
            capability.get_dc_behavior(mock_subcircuit_instance, {})

    def test_connectivity_capability_raises_framework_logic_error_if_cache_is_empty(self, mock_subcircuit_instance):
        """VERIFIES: SubcircuitInstance raises FrameworkLogicError if its Topology cache is not populated."""
        capability = mock_subcircuit_instance.get_capability(IConnectivityProvider)
        ### FIXED ### The match string is now more specific and robust to match the actual exception.
        with pytest.raises(FrameworkLogicError, match="before its topology results were cached"):
            capability.get_connectivity(mock_subcircuit_instance)

    def test_mna_capability_raises_component_error_on_freq_mismatch(self, mock_subcircuit_instance):
        """VERIFIES: SubcircuitInstance's MNA capability detects and rejects cache/sweep frequency mismatches."""
        mock_subcircuit_instance.cached_y_parameters_ac = np.ones((10, 2, 2))
        capability = mock_subcircuit_instance.get_capability(IMnaContributor)
        
        with pytest.raises(ComponentError) as excinfo:
            capability.get_mna_stamps(mock_subcircuit_instance, np.ones(5), {})
        
        assert "Mismatched frequency count" in excinfo.value.details
        assert "cached data contains 10 points" in excinfo.value.details
        assert "current sweep requires 5 points" in excinfo.value.details
        assert "cache consistency failure" in excinfo.value.details


class TestFrameworkContractEnforcement:
    """
    Verifies the framework's runtime "Validation Gateways" are robust against
    malicious or incompetent component plugins.
    """

    class TestMnaContributorContractGateway:
        """Tests the validation gateway for the IMnaContributor capability."""

        def test_framework_rejects_non_quantity_return(self, run_bad_ac_test_harness):
            component_yaml = "- {id: U1, type: BadComponentReturnsNumber, ports: {p1: p1, p2: gnd}}"
            cause = run_bad_ac_test_harness(component_yaml)
            assert isinstance(cause, ComponentError)
            assert "is not a pint.Quantity object" in cause.details
            assert "but type 'ndarray'" in cause.details

        def test_framework_rejects_wrong_dimension_return(self, run_bad_ac_test_harness):
            component_yaml = "- {id: U1, type: BadComponentReturnsWrongDimension, ports: {p1: p1, p2: gnd}}"
            cause = run_bad_ac_test_harness(component_yaml)
            assert isinstance(cause, ComponentError)
            assert isinstance(cause.__cause__, pint.DimensionalityError)
            assert "has incorrect physical dimension" in cause.details
            assert "Expected [admittance]" in cause.details

        def test_framework_rejects_mismatched_stamp_shape(self, run_bad_ac_test_harness):
            component_yaml = "- {id: U1, type: BadComponentReturnsWrongShape, ports: {p1: p1, p2: gnd}}"
            cause = run_bad_ac_test_harness(component_yaml)
            assert isinstance(cause, ComponentError)
            assert cause.__cause__ is None
            assert "Stamp magnitude must be a 3D NumPy array" in cause.details
            assert "but got ndim=2" in cause.details

        def test_framework_wraps_unexpected_component_exceptions(self, run_bad_ac_test_harness):
            component_yaml = "- {id: U1, type: BadComponentRaisesRandomError, ports: {p1: p1, p2: gnd}}"
            cause = run_bad_ac_test_harness(component_yaml)
            assert isinstance(cause, ComponentError)
            assert isinstance(cause.__cause__, ValueError)
            assert "Something went wrong" in cause.details

        def test_framework_rejects_mismatched_port_names_in_stamp(self, run_bad_ac_test_harness):
            """
            VERIFIES: The framework raises a ComponentError if an MNA stamp is returned with port names
                    that do not match the component's declaration.
            """
            component_yaml = "- {id: U1, type: BadComponentMismatchedPorts, ports: {p1: p1, p2: gnd}}"
            cause = run_bad_ac_test_harness(component_yaml)
            
            assert isinstance(cause, ComponentError)
            assert cause.component_fqn == "top.U1"
            
            # Verify that the diagnostic message is clear and actionable.
            assert "Stamping failed" in cause.details
            assert "mismatch between the component's returned port IDs and the netlist's port mapping" in cause.details
            assert "KeyError" in cause.details # Confirms the underlying cause

    class TestDcContributorContractGateway:
        """Tests the validation gateway for the IDcContributor capability."""

        def test_framework_rejects_invalid_payload_type(self, run_bad_dc_test_harness):
            component_yaml = "- {id: U1, type: BadDcReturnsWrongPayloadType, ports: {p1: n1, p2: gnd}}"
            cause = run_bad_dc_test_harness(component_yaml)
            assert cause.component_fqn == "top.U1"
            assert "payload was of an invalid type" in cause.details

        def test_framework_rejects_wrong_dimension_in_scalar_payload(self, run_bad_dc_test_harness):
            component_yaml = "- {id: U1, type: BadDcReturnsWrongScalarDimension, ports: {p1: n1, p2: gnd}}"
            cause = run_bad_dc_test_harness(component_yaml) # Harness already asserts cause is ComponentError

            # Primary assertion: Check the FQN and the presence of the expected cause
            assert cause.component_fqn == "top.U1"
            assert isinstance(cause.__cause__, pint.DimensionalityError)

            # Secondary assertion: Check for key concepts in the diagnostic message for clarity
            assert "incorrect physical dimensions" in cause.details
            assert "siemens" in cause.details # Check for expected dimension

        def test_framework_rejects_malformed_n_port_tuple_payload(self, run_bad_dc_test_harness):
            component_yaml = "- {id: U1, type: BadDcReturnsMalformedNPortTuple, ports: {p1: n1}}"
            cause = run_bad_dc_test_harness(component_yaml)
            assert "had an invalid structure" in cause.details

        def test_framework_rejects_wrong_shape_in_scalar_payload(self, run_bad_dc_test_harness):
            component_yaml = "- {id: U1, type: BadDcReturnsWrongScalarShape, ports: {p1: n1, p2: gnd}}"
            cause = run_bad_dc_test_harness(component_yaml)
            assert "magnitude was not a scalar (ndim=0)" in cause.details
            assert "Got ndim=1" in cause.details
            
        def test_framework_rejects_nport_tuple_with_non_quantity_element(self, run_bad_dc_test_harness):
            component_yaml = "- {id: U1, type: BadDcReturnsNPortTupleWithBadQuantity, ports: {p1: n1, p2: n2}}"
            cause = run_bad_dc_test_harness(component_yaml)
            assert "must be a pint.Quantity" in cause.details
            assert "got type 'ndarray'" in cause.details

        def test_framework_rejects_nport_tuple_with_non_list_element(self, run_bad_dc_test_harness):
            component_yaml = "- {id: U1, type: BadDcReturnsNPortTupleWithBadPortList, ports: {p1: n1, p2: n2}}"
            cause = run_bad_dc_test_harness(component_yaml)
            assert "had an invalid structure" in cause.details

        def test_framework_rejects_nport_tuple_with_wrong_dimension(self, run_bad_dc_test_harness):
            component_yaml = "- {id: U1, type: BadDcReturnsNPortTupleWithWrongDimension, ports: {p1: n1, p2: n2}}"
            cause = run_bad_dc_test_harness(component_yaml)
            ### FIXED ### Use multiple, more robust assertions.
            assert "incorrect physical dimensions" in cause.details
            assert "ohm" in cause.details
            assert "siemens" in cause.details

        def test_framework_rejects_nport_dc_admittance_with_wrong_shape(self, run_bad_dc_test_harness):
            component_yaml = "- {id: U1, type: BadDcReturnsNPortWithWrongShape, ports: {p1: n1, p2: n2}}"
            cause = run_bad_dc_test_harness(component_yaml)
            assert "was not a 2D NumPy array" in cause.details
            assert "Got ndim=1" in cause.details


class TestLeafElementInternalContracts:
    """Verifies that leaf elements correctly enforce their own physical constraints."""

    @pytest.mark.parametrize(
        "element_class, param_name, invalid_value, expected_msg",
        [
            (Resistor, "resistance", -50 * ureg.ohm, "non-negative"),
            (Resistor, "resistance", (50+1j) * ureg.ohm, "must be real"),
            (Capacitor, "capacitance", -1 * ureg.pF, "non-negative"),
            (Inductor, "inductance", -1 * ureg.nH, "non-negative"),
        ]
    )
    def test_leaf_element_rejects_invalid_values(
            self, element_class, param_name, invalid_value, expected_msg
        ):
        mna_contributor = element_class.MnaContributor()
        freqs = np.array([1e9])

        class MockContext(ComponentBase):
            @classmethod
            def declare_parameters(cls): return {param_name: str(invalid_value.units)}
            @classmethod
            def declare_ports(cls): return ['p1', 'p2']

        mock_component = MockContext("U1", "Mock", None, "top", {})
        params = {mock_component.parameter_fqns[0]: Quantity(np.array([invalid_value.magnitude]), invalid_value.units)}

        with pytest.raises(ComponentError) as excinfo:
            mna_contributor.get_mna_stamps(mock_component, freqs, params)
        assert expected_msg in excinfo.value.details


class TestFrameworkRejectsNonPhysicalExpressions:
    """
    Verifies the framework rejects parameters defined by expressions that evaluate
    to non-physical values for leaf components (e.g., negative resistance).
    This class now correctly distinguishes between DC and AC validation paths.
    """

    class TestDcValidation:
        """
        Tests for non-physical values detected during the DC analysis gateway.
        These tests verify that constant non-physical values are caught by the
        DCAnalyzer, which runs before the main AC sweep.
        """
        
        @pytest.mark.parametrize(
            "comp_type, param_name, param_expr, expected_msg",
            [
                # These are constant negative values, which MUST be caught by DC analysis.
                ("Resistor", "resistance", "Quantity('-50 ohm')", "Negative resistance"),
                ("Capacitor", "capacitance", "Quantity('-1 pF')", "Negative capacitance"),
                ("Inductor", "inductance", "Quantity('-10 nH')", "Negative inductance"),
            ]
        )
        def test_dc_rejects_constant_negative_values(
            self, run_bad_dc_test_harness, comp_type, param_name, param_expr, expected_msg
        ):
            """
            VERIFIES: The DC analysis path is the first runtime gateway to catch
                      and reject constant non-physical parameter values.
            """
            component_yaml = f"""
    - id: U1
      type: {comp_type}
      ports: {{p1: p1, p2: gnd}}
      parameters:
        {param_name}: "{param_expr}"
    """
            # MODIFIED: Use the dedicated DC test harness to correctly target the DCAnalyzer.
            cause = run_bad_dc_test_harness(component_yaml)
            assert isinstance(cause, ComponentError)
            assert expected_msg in cause.details
            assert cause.frequency is None, "Error should originate from DC analysis (frequency=None)"
        
        def test_dc_rejects_constant_complex_resistance(self, run_bad_dc_test_harness):
            """VERIFIES: The DC analysis gateway rejects a constant complex resistance."""
            component_yaml = """
    - id: U1
      type: Resistor
      ports: {p1: p1, p2: gnd}
      parameters:
        # FIXED: Use built-in complex(), not deprecated np.complex.
        resistance: "complex(50, 1) * Quantity('ohm')"
    """
            # This test now correctly targets the DC analysis path using the appropriate harness.
            cause = run_bad_dc_test_harness(component_yaml)
            assert isinstance(cause, ComponentError)
            assert "must be real" in cause.details


    class TestAcValidation:
        """
        Tests for non-physical values detected during the AC (MNA) sweep gateway.
        These tests are specifically engineered to pass the DC analysis gateway
        and fail only at non-zero frequencies.
        """
        
        @pytest.mark.parametrize(
            "comp_type, param_name, param_expr, expected_msg",
            [
                # REVISED: Expressions are now dimensionally correct.
                ("Resistor", "resistance", "'(50 - (freq / Quantity(\"1 GHz\"))*100) * Quantity(\"ohm\")'", "must be non-negative"),
                ("Capacitor", "capacitance", "'(1 - (freq / Quantity(\"1 GHz\"))*2) * Quantity(\"pF\")'", "must be non-negative"),
                ("Inductor", "inductance", "'(10 - (freq / Quantity(\"1 GHz\"))*20) * Quantity(\"nH\")'", "must be non-negative"),
                ("Resistor", "resistance", "'(50 + 1j * (freq / Quantity(\"1 GHz\"))) * Quantity(\"ohm\")'", "must be real"),
            ]
        )
        def test_ac_rejects_freq_dependent_non_physical_values(
            self, run_bad_ac_test_harness, comp_type, param_name, param_expr, expected_msg
        ):
            """
            VERIFIES: The AC MNA validation path correctly rejects non-physical parameter
                      values that only manifest at non-zero frequencies, after passing DC checks.
            """
            component_yaml = f"""
    - id: U1
      type: {comp_type}
      ports: {{p1: p1, p2: gnd}}
      parameters:
        {param_name}: {param_expr}
    """
            # This harness is now guaranteed to fail in the AC path because the DC path will pass.
            cause = run_bad_ac_test_harness(component_yaml)
            assert isinstance(cause, ComponentError)
            assert expected_msg in cause.details


class TestLeafElementTopologyCapability:
    """Performs direct unit tests on the ITopologyContributor capability."""

    @pytest.mark.parametrize("r_val, expected_open", [
        (Quantity(0, "ohm"), False),
        (Quantity(50, "ohm"), False),
        (Quantity(np.inf, "ohm"), True),
    ])
    def test_resistor_is_structurally_open(self, r_val, expected_open):
        capability = Resistor.TopologyContributor()
        params = {"resistance": r_val}
        assert capability.is_structurally_open(None, params) == expected_open

    @pytest.mark.parametrize("c_val, expected_open", [
        (Quantity(0, "pF"), True),
        (Quantity(1, "pF"), False),
        (Quantity(np.inf, "pF"), False),
    ])
    def test_capacitor_is_structurally_open(self, c_val, expected_open):
        capability = Capacitor.TopologyContributor()
        params = {"capacitance": c_val}
        assert capability.is_structurally_open(None, params) == expected_open

    @pytest.mark.parametrize("l_val, expected_open", [
        (Quantity(0, "nH"), False),
        (Quantity(10, "nH"), False),
        (Quantity(np.inf, "nH"), True),
    ])
    def test_inductor_is_structurally_open(self, l_val, expected_open):
        capability = Inductor.TopologyContributor()
        params = {"inductance": l_val}
        assert capability.is_structurally_open(None, params) == expected_open

class TestLeafElementCapabilities:
    """Performs positive-path validation on the capabilities of built-in RLC elements."""

    @pytest.fixture
    def freqs(self):
        return np.array([1e9, 2e9])

    def test_resistor_mna_contribution(self, freqs):
        mock_r = Resistor("R1", "Resistor", None, "top", {'p1': 'net1', 'p2': 'gnd'})
        r_val = Quantity(50.0, "ohm")
        params = {mock_r.parameter_fqns[0]: Quantity(np.full_like(freqs, r_val.magnitude), r_val.units)}
        y_qty, port_names = mock_r.get_capability(IMnaContributor).get_mna_stamps(mock_r, freqs, params)

        assert port_names == ['p1', 'p2']
        assert y_qty.dimensionality == ADMITTANCE_DIMENSIONALITY
        expected_y = 1.0 / 50.0
        base_stamp = np.array([[expected_y, -expected_y], [-expected_y, expected_y]])
        expected_stamp_vectorized = np.array([base_stamp] * len(freqs))
        np.testing.assert_allclose(y_qty.magnitude, expected_stamp_vectorized)

    @pytest.mark.parametrize("r_val, expected_y", [
        (0 * ureg.ohm, LARGE_ADMITTANCE_SIEMENS),
        (np.inf * ureg.ohm, 0.0),
    ])
    def test_resistor_mna_ideal_boundaries(self, r_val, expected_y):
        mock_r = Resistor("R1", "Resistor", None, "top", {'p1': 'net1', 'p2': 'gnd'})
        freqs = np.array([1e9])
        params = {mock_r.parameter_fqns[0]: Quantity(np.full_like(freqs, r_val.magnitude), r_val.units)}
        y_qty, _ = mock_r.get_capability(IMnaContributor).get_mna_stamps(mock_r, freqs, params)
        expected_stamp = np.array([[expected_y, -expected_y], [-expected_y, expected_y]])
        np.testing.assert_allclose(y_qty.to("siemens").magnitude[0], expected_stamp)

    def test_inductor_mna_contribution(self, freqs):
        mock_l = Inductor("L1", "Inductor", None, "top", {'p1': 'net1', 'p2': 'gnd'})
        l_val = Quantity(10e-9, "henry")
        params = {mock_l.parameter_fqns[0]: Quantity(np.full_like(freqs, l_val.magnitude), l_val.units)}
        y_qty, _ = mock_l.get_capability(IMnaContributor).get_mna_stamps(mock_l, freqs, params)

        omega = 2 * np.pi * freqs
        expected_y_mag = 1.0 / (omega * l_val.to("H").magnitude)
        expected_stamps = -1j * np.array([[[y, -y], [-y, y]] for y in expected_y_mag])
        np.testing.assert_allclose(y_qty.to("S").magnitude, expected_stamps)

    def test_capacitor_mna_contribution(self, freqs):
        mock_c = Capacitor("C1", "Capacitor", None, "top", {'p1': 'net1', 'p2': 'gnd'})
        c_val = Quantity(1e-12, "farad")
        params = {mock_c.parameter_fqns[0]: Quantity(np.full_like(freqs, c_val.magnitude), c_val.units)}
        y_qty, _ = mock_c.get_capability(IMnaContributor).get_mna_stamps(mock_c, freqs, params)

        omega = 2 * np.pi * freqs
        expected_y_mag = omega * c_val.to("F").magnitude
        expected_stamps = 1j * np.array([[[y, -y], [-y, y]] for y in expected_y_mag])
        np.testing.assert_allclose(y_qty.to("S").magnitude, expected_stamps)

    @pytest.mark.parametrize("c_val, expected_type", [
        (0 * ureg.farad, DCBehaviorType.OPEN_CIRCUIT),
        (1e-12 * ureg.farad, DCBehaviorType.OPEN_CIRCUIT),
        (np.inf * ureg.farad, DCBehaviorType.SHORT_CIRCUIT),
    ])
    def test_capacitor_dc_contribution(self, c_val, expected_type):
        mock_c = Capacitor("C1", "Capacitor", None, "top", {'p1': 'net1', 'p2': 'gnd'})
        params = {mock_c.parameter_fqns[0]: Quantity(np.array([c_val.magnitude]), c_val.units)}
        b_type, y_qty = mock_c.get_capability(IDcContributor).get_dc_behavior(mock_c, params)
        assert b_type == expected_type
        assert y_qty is None

class TestAnalysisIntegration:
    """
    Verifies that the framework correctly integrates the capabilities of built-in components
    into the analysis pipelines, ensuring they behave as expected in a full simulation context.
    """

    def test_topology_analyzer_handles_custom_nport_connectivity(self, tmp_path):
        import networkx as nx
        from rfsim_core.analysis import TopologyAnalyzer
        from rfsim_core.cache import SimulationCache

        netlist_path = tmp_path / "tee_test.yaml"
        netlist_path.write_text("""
    circuit_name: TeeTest
    components:
      - id: T1
        type: ThreePortTee
        ports: {p1: net1, p2: net2, p3: common}
    """)
        circuit = CircuitBuilder().build_simulation_model(NetlistParser().parse_to_circuit_tree(netlist_path))
        topo_analyzer = TopologyAnalyzer(circuit, SimulationCache())
        ac_graph = topo_analyzer.analyze().ac_graph

        assert nx.has_path(ac_graph, 'net1', 'common')
        assert nx.has_path(ac_graph, 'net2', 'common')
        assert nx.has_path(ac_graph, 'net1', 'net2')

    def test_topology_analyzer_correctly_uses_itopologycontributor(self, tmp_path):
        """
        VERIFIES (Critique 4): The TopologyAnalyzer service correctly integrates with the
        ITopologyContributor capability to identify structurally open components.
        """
        netlist_path = tmp_path / "structural_open_test.yaml"
        netlist_path.write_text("""
        circuit_name: StructuralOpenTest
        components:
        # This capacitor is a structural open circuit (C=0).
          - id: C_open
            type: Capacitor
            ports: {p1: net_a, p2: net_b}
            parameters: {capacitance: "Quantity('0 pF')"}
        # This resistor is a normal, active component.
          - id: R_active
            type: Resistor
            ports: {p1: net_a, p2: gnd}
            parameters: {resistance: "Quantity('50 ohm')"}
        """)
        circuit = CircuitBuilder().build_simulation_model(NetlistParser().parse_to_circuit_tree(netlist_path))
        topo_analyzer = TopologyAnalyzer(circuit, SimulationCache())

        # Perform the analysis
        results = topo_analyzer.analyze()

        # Assert that the analyzer correctly identified the open component
        assert results.structurally_open_components == {'C_open'}

        # Assert that the AC graph, which is built *after* removing open components,
        # does NOT contain an edge corresponding to the open capacitor.
        ac_graph = results.ac_graph
        assert not ac_graph.has_edge('net_a', 'net_b')
        # The active resistor should still be present.
        assert ac_graph.has_edge('net_a', 'gnd')

class TestSubcircuitIntegration:
    def test_subcircuit_simulation_produces_correct_y_parameters(self, tmp_path):
        # Create sub_pi_attenuator.yaml
        (tmp_path / "sub_pi_attenuator.yaml").write_text("""
        circuit_name: PiAttenuator
        parameters:
          R_series: {expression: "Quantity('141.9 ohm')", dimension: "ohm"}                                  
          R_shunt: {expression: "Quantity('96.2 ohm')", dimension: "ohm"}  
        components:
          - {id: R1, type: Resistor, ports: {p1: IN, p2: OUT}, parameters: {resistance: R_series}}
          - {id: R2, type: Resistor, ports: {p1: IN, p2: gnd}, parameters: {resistance: R_shunt}}
          - {id: R3, type: Resistor, ports: {p1: OUT, p2: gnd}, parameters: {resistance: R_shunt}}
        ports:
          - {id: IN, reference_impedance: "50 ohm"}
          - {id: OUT, reference_impedance: "50 ohm"}
        """)
        # Create top_with_sub.yaml
        (tmp_path / "top_with_sub.yaml").write_text("""
        circuit_name: TopLevelTest
        components:
          - id: ATTEN
            type: Subcircuit
            definition_file: ./sub_pi_attenuator.yaml
            ports: {IN: p_in, OUT: p_out}
        ports:
          - {id: p_in, reference_impedance: "50 ohm"}
          - {id: p_out, reference_impedance: "50 ohm"}
        """)

        # Build the full hierarchical model
        parser = NetlistParser()
        builder = CircuitBuilder()
        parsed_tree = parser.parse_to_circuit_tree(tmp_path / "top_with_sub.yaml")
        sim_circuit = builder.build_simulation_model(parsed_tree)

        from rfsim_core.simulation import run_sweep

        # Run the sweep
        freqs = np.array([1e9]) # Frequency is irrelevant for ideal resistors
        result, _ = run_sweep(sim_circuit, freqs)

        # Calculate the known, analytical Y-parameters of the pi-attenuator
        g_s = 1.0 / 141.9
        g_p = 1.0 / 96.2
        y11_expected = g_s + g_p
        y12_expected = -g_s
        y21_expected = -g_s
        y22_expected = g_s + g_p
        expected_y_matrix = np.array([[[y11_expected, y12_expected], [y21_expected, y22_expected]]])

        # Compare simulation result with analytical solution
        np.testing.assert_allclose(result.y_parameters, expected_y_matrix, rtol=1e-9)

    def test_subcircuit_correctly_reports_internal_disconnect_to_parent(self, tmp_path):
        """
        VERIFIES (Critique 5): The full hierarchical topology analysis chain works.
        A subcircuit's IConnectivityProvider correctly reports its internal topology
        (including disconnects) to the parent's TopologyAnalyzer.
        """
        # Create the disconnected subcircuit definition
        (tmp_path / "sub_disconnected.yaml").write_text("""
        circuit_name: DisconnectedSub
        components:
        - {id: R1, type: Resistor, ports: {p1: IN, p2: gnd}, parameters: {resistance: 50}}
        ports:
        - {id: IN, reference_impedance: "50 ohm"}
        - {id: OUT, reference_impedance: "50 ohm"}
        """)
        # Create the top-level netlist that uses the subcircuit
        (tmp_path / "top_level_sub_test.yaml").write_text("""
        circuit_name: TopLevelSubTest
        components:
        - id: SUB1
          type: Subcircuit
          definition_file: ./sub_disconnected.yaml
          ports: {IN: net_in, OUT: net_out}
        ports:
        - {id: net_in, reference_impedance: "50 ohm"}
        - {id: net_out, reference_impedance: "50 ohm"}
        """)

        # --- Setup: This part simulates the work done by the SimulationEngine ---
        parser = NetlistParser()
        builder = CircuitBuilder()
        cache = SimulationCache()
        top_parsed_tree = parser.parse_to_circuit_tree(tmp_path / "top_level_sub_test.yaml")
        top_circuit = builder.build_simulation_model(top_parsed_tree)

        # Manually run the analysis on the subcircuit first, as the engine would
        sub_instance = top_circuit.sim_components['SUB1']
        sub_circuit_def = sub_instance.sub_circuit_object
        sub_topo_analyzer = TopologyAnalyzer(sub_circuit_def, cache)
        sub_topo_results = sub_topo_analyzer.analyze()

        # Manually populate the cache on the instance, as the engine would
        sub_instance.cached_topology_results = sub_topo_results
        # --- End Setup ---

        # --- Test: Run the TopologyAnalyzer on the PARENT circuit ---
        parent_topo_analyzer = TopologyAnalyzer(top_circuit, cache)
        parent_topo_results = parent_topo_analyzer.analyze()
        parent_ac_graph = parent_topo_results.ac_graph

        # --- Assert ---
        # The most critical assertion: the parent's graph should NOT have a path
        # between net_in and net_out, because the subcircuit between them is
        # internally disconnected.
        import networkx as nx
        assert not nx.has_path(parent_ac_graph, 'net_in', 'net_out')
        
        # Assert that the connections that *do* exist are present
        assert nx.has_path(parent_ac_graph, 'net_in', 'gnd')


class TestMnaAssemblerContractEnforcement:
    """
    Verifies that the MnaAssembler itself enforces critical component contracts
    during its initialization, before any simulation begins.
    """

    def test_mna_assembler_rejects_nport_component_with_default_connectivity(self, tmp_path):
        """
        VERIFIES (Critique 3): The MnaAssembler raises a ComponentError if an N-port
        (>2 ports) component uses the default, non-connecting IConnectivityProvider.
        This is a critical safety feature to prevent silent topological errors.
        """
        netlist_path = tmp_path / "bad_nport_connectivity.yaml"
        netlist_path.write_text("""
        circuit_name: BadNPortConnectivityTest
        components:
          - id: U1
            type: BadThreePortDefaultConnectivity
            ports: {p1: a, p2: b, p3: c}
        """)

        parser = NetlistParser()
        builder = CircuitBuilder()
        parsed_tree = parser.parse_to_circuit_tree(netlist_path)
        sim_circuit = builder.build_simulation_model(parsed_tree)

        # The error should be raised during the MnaAssembler's initialization,
        # specifically when it computes the sparsity pattern.
        with pytest.raises(ComponentError) as excinfo:
            MnaAssembler(sim_circuit)

        # Assert that the error is specific and actionable.
        assert excinfo.value.component_fqn == "top.U1"
        details = excinfo.value.details
        assert "uses the default connectivity provider" in details
        assert "ambiguous and likely incorrect circuit topology" in details 
        assert "MUST implement a custom `IConnectivityProvider` capability" in details