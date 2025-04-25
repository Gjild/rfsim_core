# tests/test_components.py
import pytest
import numpy as np
from pint import DimensionalityError

# Use the application's unit registry
from rfsim_core import ureg, Quantity, pint
from rfsim_core.components import Resistor, Capacitor, Inductor, ComponentError
from rfsim_core.parameters import ParameterManager
from rfsim_core.data_structures import Component as ComponentData
from rfsim_core.circuit_builder import CircuitBuilder, CircuitBuildError
from rfsim_core import ParameterError

# Helper for comparing quantities
def assert_quantity_close(q1, q2, **kwargs):
    """Asserts that two Pint quantities are close in magnitude and have same units."""
    assert q1.units == q2.units, f"Units differ: {q1.units} vs {q2.units}"
    np.testing.assert_allclose(q1.magnitude, q2.magnitude, **kwargs)

# --- Fixtures ---
@pytest.fixture
def param_manager():
    """Basic ParameterManager fixture."""
    return ParameterManager({"global_res": "1 kohm", "global_cap": "10 pF"})

@pytest.fixture
def builder():
    """CircuitBuilder fixture."""
    return CircuitBuilder()

# --- Resistor Tests ---
def test_resistor_creation_and_params():
    params = {"resistance": Quantity("50 ohm")}
    r = Resistor("R1", "Resistor", params)
    assert r.instance_id == "R1"
    assert r.component_type == "Resistor"
    assert r.resistance == Quantity("50 ohm")
    assert r.get_parameter("resistance") == Quantity("50 ohm")

def test_resistor_admittance():
    params = {"resistance": Quantity("50 ohm")}
    r = Resistor("R1", "Resistor", params)
    freq = 1e9 # 1 GHz
    expected_adm = Quantity(1.0 / 50.0, ureg.siemens).astype(np.complex128)
    actual_adm = r.get_admittance(freq)
    assert_quantity_close(actual_adm, expected_adm)
    # Test array frequency
    freq_arr = np.array([1e9, 2e9])
    expected_adm_arr_val = np.array([1.0/50.0 + 0j, 1.0/50.0 + 0j])
    expected_adm_arr = Quantity(expected_adm_arr_val, ureg.siemens)
    actual_adm_arr = r.get_admittance(freq_arr)
    assert_quantity_close(actual_adm_arr, expected_adm_arr)

def test_resistor_zero_resistance():
    params = {"resistance": Quantity("0 ohm")}
    r = Resistor("R0", "Resistor", params) # Should warn on creation
    freq = 1e9
    expected_adm = Quantity(np.inf + 0j, ureg.siemens)
    actual_adm = r.get_admittance(freq)
    # assert actual_adm.magnitude == np.inf # Direct comparison with inf is tricky
    assert np.isinf(actual_adm.magnitude)
    assert actual_adm.units == ureg.siemens

def test_resistor_negative_resistance():
     params = {"resistance": Quantity("-50 ohm")}
     with pytest.raises(ComponentError, match="Resistance cannot be negative"):
         Resistor("Rneg", "Resistor", params)

# --- Capacitor Tests ---
def test_capacitor_creation_and_params():
    params = {"capacitance": Quantity("1 pF")}
    c = Capacitor("C1", "Capacitor", params)
    assert c.instance_id == "C1"
    assert c.capacitance == Quantity("1 pF")

def test_capacitor_admittance():
    cap_val = 1e-12 # 1 pF
    params = {"capacitance": Quantity(cap_val, ureg.farad)}
    c = Capacitor("C1", "Capacitor", params)
    freq = 1e9 # 1 GHz
    omega = 2 * np.pi * freq
    expected_adm_val = 1j * omega * cap_val
    expected_adm = Quantity(expected_adm_val, ureg.siemens)
    actual_adm = c.get_admittance(freq)
    assert_quantity_close(actual_adm, expected_adm, rtol=1e-7)
    # Test array frequency
    freq_arr = np.array([1e9, 2e9])
    omega_arr = 2 * np.pi * freq_arr
    expected_adm_arr_val = 1j * omega_arr * cap_val
    expected_adm_arr = Quantity(expected_adm_arr_val, ureg.siemens)
    actual_adm_arr = c.get_admittance(freq_arr)
    assert_quantity_close(actual_adm_arr, expected_adm_arr, rtol=1e-7)

# --- Inductor Tests ---
def test_inductor_creation_and_params():
    params = {"inductance": Quantity("1 nH")}
    l = Inductor("L1", "Inductor", params)
    assert l.instance_id == "L1"
    assert l.inductance == Quantity("1 nH")

def test_inductor_admittance():
    ind_val = 1e-9 # 1 nH
    params = {"inductance": Quantity(ind_val, ureg.henry)}
    l = Inductor("L1", "Inductor", params)
    freq = 1e9 # 1 GHz
    omega = 2 * np.pi * freq
    expected_imp_val = 1j * omega * ind_val
    expected_adm_val = 1.0 / expected_imp_val
    expected_adm = Quantity(expected_adm_val, ureg.siemens)
    actual_adm = l.get_admittance(freq)
    assert_quantity_close(actual_adm, expected_adm, rtol=1e-7)
    # Test array frequency
    freq_arr = np.array([1e9, 2e9])
    omega_arr = 2 * np.pi * freq_arr
    expected_imp_arr_val = 1j * omega_arr * ind_val
    expected_adm_arr_val = 1.0 / expected_imp_arr_val
    expected_adm_arr = Quantity(expected_adm_arr_val, ureg.siemens)
    actual_adm_arr = l.get_admittance(freq_arr)
    assert_quantity_close(actual_adm_arr, expected_adm_arr, rtol=1e-7)

def test_inductor_admittance_at_dc():
    # THIS TEST SHOULD FAIL! DC SIMULATION IS FUNDAMENTALLY SEPARATE FROM AC!
    params = {"inductance": Quantity(1e-6, ureg.henry)}
    l = Inductor("L1", "Inductor", params)
    freq = 0.0
    expected_adm = Quantity(0.0 + 0.0j, ureg.siemens) # Expect zero admittance at DC
    with pytest.raises(ZeroDivisionError) as excinfo:
        actual_adm = l.get_admittance(freq)

def test_inductor_zero_inductance():
    params = {"inductance": Quantity("0 H")}
    l = Inductor("L0", "Inductor", params) # Should warn
    freq = 1e9
    expected_adm = Quantity(np.inf + 0j, ureg.siemens)
    with pytest.raises(ZeroDivisionError) as excinfo:
        actual_adm = l.get_admittance(freq)
    #assert np.isinf(actual_adm.magnitude)
    #assert actual_adm.units == ureg.siemens
    # Check at DC (should still be inf based on code, though physically debatable)
    #actual_adm_dc = l.get_admittance(0.0)
    #assert np.isinf(actual_adm_dc.magnitude) # Current code returns inf even at DC for L=0

# --- Parameter Validation Tests (via CircuitBuilder) ---

def test_builder_valid_component_params(builder, param_manager):
    comp_data = ComponentData(
        instance_id="R1",
        component_type="Resistor",
        parameters={"resistance": "50 ohm"} # Literal value
    )
    processed = builder._process_component_parameters(
        comp_data, Resistor.declare_parameters(), param_manager
    )
    assert "resistance" in processed
    assert processed["resistance"] == Quantity("50 ohm")

def test_builder_valid_global_ref(builder, param_manager):
    comp_data = ComponentData(
        instance_id="R2",
        component_type="Resistor",
        parameters={"resistance": "global_res"} # Reference global
    )
    processed = builder._process_component_parameters(
        comp_data, Resistor.declare_parameters(), param_manager
    )
    assert processed["resistance"] == Quantity("1 kohm")

def test_builder_missing_parameter(builder, param_manager):
    comp_data = ComponentData(
        instance_id="R_bad",
        component_type="Resistor",
        parameters={} # Missing resistance
    )
    with pytest.raises(ParameterError, match="Required parameter 'resistance' missing"):
        builder._process_component_parameters(
            comp_data, Resistor.declare_parameters(), param_manager
        )

def test_builder_invalid_unit_literal(builder, param_manager):
    comp_data = ComponentData(
        instance_id="R_bad",
        component_type="Resistor",
        parameters={"resistance": "50 meter"} # Wrong unit dimension
    )
    with pytest.raises(DimensionalityError) as excinfo:
         builder._process_component_parameters(
            comp_data, Resistor.declare_parameters(), param_manager
        )
    # Check specific details of the error message if needed
    assert "Dimensionality mismatch" in str(excinfo.value)
    assert "meter" in str(excinfo.value) # Dimension of meters
    assert "ohm" in str(excinfo.value) # Expected dimension

def test_builder_invalid_unit_global_ref(builder, param_manager):
    comp_data = ComponentData(
        instance_id="R_bad",
        component_type="Resistor",
        parameters={"resistance": "global_cap"} # Refers to capacitance param
    )
    with pytest.raises(pint.DimensionalityError) as excinfo:
         builder._process_component_parameters(
            comp_data, Resistor.declare_parameters(), param_manager
        )
    assert "picofarad" in str(excinfo.value)
    assert "ohm" in str(excinfo.value)

def test_builder_unparseable_literal(builder, param_manager):
     comp_data = ComponentData(
        instance_id="C_bad",
        component_type="Capacitor",
        parameters={"capacitance": "10 picoFaraday"} # Invalid unit name
    )
     with pytest.raises(ParameterError, match="Error parsing literal value"):
          builder._process_component_parameters(
             comp_data, Capacitor.declare_parameters(), param_manager
         )

def test_builder_unknown_component_type(builder, param_manager):
    # Create a dummy ParsedCircuitData
    class MockParsedCircuit:
        name = "TestCircuit"
        parameter_manager = param_manager
        components = {
            "X1": ComponentData(instance_id="X1", component_type="NonExistent", parameters={}, ports={})
        }
    parsed_data = MockParsedCircuit()
    with pytest.raises(CircuitBuildError) as excinfo:
        builder.build_circuit(parsed_data)
    assert "Unknown component type 'NonExistent'" in str(excinfo.value)

def test_full_build_process(parser, builder, valid_yaml_string):
     """Test parsing then building"""
     parsed_circuit = parser.parse(valid_yaml_string)
     # Should have raw components
     assert not hasattr(parsed_circuit, 'sim_components')
     assert isinstance(parsed_circuit.components['R1'], ComponentData)

     # Now build
     built_circuit = builder.build_circuit(parsed_circuit)

     # Should have sim_components attribute populated
     assert hasattr(built_circuit, 'sim_components')
     assert len(built_circuit.sim_components) == 3
     assert isinstance(built_circuit.sim_components['R1'], Resistor)
     assert isinstance(built_circuit.sim_components['C1'], Capacitor)
     assert isinstance(built_circuit.sim_components['L1'], Inductor)

     # Check if parameters were processed correctly
     r1_sim = built_circuit.sim_components['R1']
     assert r1_sim.get_parameter('resistance') == Quantity('1 kohm')

     c1_sim = built_circuit.sim_components['C1']
     # 'default_cap' was '10 pF' globally
     assert c1_sim.get_parameter('capacitance') == Quantity('10 pF')

     l1_sim = built_circuit.sim_components['L1']
     assert l1_sim.get_parameter('inductance') == Quantity('1 uH')