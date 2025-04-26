# tests/test_components.py
import pytest
import numpy as np
from pint import DimensionalityError

# Use the application's unit registry
# Adjust the import path based on your project structure if necessary
# Assuming 'src' is in your PYTHONPATH or you installed the package
from rfsim_core import ureg, Quantity, pint
from rfsim_core.components import Resistor, Capacitor, Inductor, ComponentError
from rfsim_core.parameters import ParameterManager, ParameterError
from rfsim_core.data_structures import Component as ComponentData
from rfsim_core.circuit_builder import CircuitBuilder, CircuitBuildError
from rfsim_core.parser import NetlistParser # Needed for full_build_process test

# Helper for comparing quantities with arrays
def assert_quantity_close(q1, q2, rtol=1e-5, atol=1e-8, **kwargs):
    """
    Asserts that two Pint quantities are close in magnitude (element-wise for arrays)
    and have same units. Handles potential NaNs and Infs separately.
    """
    assert q1.units == q2.units, f"Units differ: {q1.units} vs {q2.units}"

    m1 = q1.magnitude
    m2 = q2.magnitude

    # Ensure they are numpy arrays for consistent handling
    if not isinstance(m1, np.ndarray):
        m1 = np.array(m1)
    if not isinstance(m2, np.ndarray):
        m2 = np.array(m2)

    assert m1.shape == m2.shape, f"Shapes differ: {m1.shape} vs {m2.shape}"

    # Handle special values: NaN
    nan_mask1 = np.isnan(m1)
    nan_mask2 = np.isnan(m2)
    assert np.array_equal(nan_mask1, nan_mask2), "NaN locations differ"

    # Handle special values: Inf
    inf_mask1 = np.isinf(m1)
    inf_mask2 = np.isinf(m2)
    # Check if infinities match (including sign)
    assert np.array_equal(inf_mask1, inf_mask2), "Infinity locations differ"
    if np.any(inf_mask1):
         assert np.array_equal(m1[inf_mask1], m2[inf_mask1]), "Infinity signs differ"


    # Compare finite values
    finite_mask = ~nan_mask1 & ~inf_mask1
    if np.any(finite_mask):
        np.testing.assert_allclose(
            m1[finite_mask], m2[finite_mask], rtol=rtol, atol=atol, **kwargs
        )

# --- Fixtures ---
@pytest.fixture
def param_manager():
    """Basic ParameterManager fixture."""
    return ParameterManager({"global_res": "1 kohm", "global_cap": "10 pF"})

@pytest.fixture
def builder():
    """CircuitBuilder fixture."""
    return CircuitBuilder()

@pytest.fixture
def parser():
    """NetlistParser fixture."""
    return NetlistParser()

@pytest.fixture
def valid_yaml_string_for_build():
    """Provides a basic valid YAML string suitable for builder tests."""
    # Added sweep section to make parsing valid for Phase 4
    return """
circuit_name: Test RLC Circuit Build
sweep: { type: list, points: ['1 GHz'] }
parameters:
  supply_voltage: '5 V'
  default_cap: '10 pF'
components:
  - type: Resistor
    id: R1
    ports: { p1: net1, p2: gnd }
    parameters:
      resistance: '1kohm'
  - type: Capacitor
    id: C1
    ports: { p1: net1, p2: gnd }
    parameters:
      capacitance: default_cap
  - type: Inductor
    id: L1
    ports: { p1: net1, p2: 'out' }
    parameters:
      inductance: '1 uH'
ports:
  - id: out
    reference_impedance: '50 ohm'
"""


# --- Resistor Tests ---
def test_resistor_creation_and_params():
    """ Test basic resistor instantiation and parameter access. """
    params = {"resistance": Quantity("50 ohm")}
    r = Resistor("R1", "Resistor", params)
    assert r.instance_id == "R1"
    assert r.component_type == "Resistor"
    assert r.resistance == Quantity("50 ohm")
    assert r.get_parameter("resistance") == Quantity("50 ohm")

def test_resistor_admittance_vectorized():
    """ Test resistor admittance calculation with frequency array. """
    params = {"resistance": Quantity("50 ohm")}
    r = Resistor("R1", "Resistor", params)
    freq_arr = np.array([1e9, 2e9, 3e9])
    # Expected admittance is constant 1/R for all frequencies
    expected_adm_val = np.array([1.0/50.0 + 0j] * len(freq_arr))
    expected_adm = Quantity(expected_adm_val, ureg.siemens)

    actual_adm = r.get_admittance(freq_arr)

    assert actual_adm.shape == freq_arr.shape
    assert_quantity_close(actual_adm, expected_adm)

def test_resistor_zero_resistance_vectorized():
    """ Test zero resistance resistor admittance calculation with frequency array. """
    params = {"resistance": Quantity("0 ohm")}
    r = Resistor("R0", "Resistor", params) # Should warn on creation
    freq_arr = np.array([1e9, 2e9])
    # Expected admittance is infinite for all frequencies
    expected_adm_val = np.array([np.inf + 0j] * len(freq_arr))
    expected_adm = Quantity(expected_adm_val, ureg.siemens)

    actual_adm = r.get_admittance(freq_arr)

    assert actual_adm.shape == freq_arr.shape
    assert_quantity_close(actual_adm, expected_adm) # Helper handles inf comparison


def test_resistor_negative_resistance():
     """ Test instantiation fails with negative resistance. """
     params = {"resistance": Quantity("-50 ohm")}
     with pytest.raises(ComponentError, match="Resistance cannot be negative"):
         Resistor("Rneg", "Resistor", params)


# --- Capacitor Tests ---
def test_capacitor_creation_and_params():
    """ Test basic capacitor instantiation and parameter access. """
    params = {"capacitance": Quantity("1 pF")}
    c = Capacitor("C1", "Capacitor", params)
    assert c.instance_id == "C1"
    assert c.capacitance == Quantity("1 pF")

def test_capacitor_admittance_vectorized():
    """ Test capacitor admittance calculation with frequency array. """
    cap_val = 1e-12 # 1 pF
    params = {"capacitance": Quantity(cap_val, ureg.farad)}
    c = Capacitor("C1", "Capacitor", params)
    freq_arr = np.array([1e9, 2e9, 0.0]) # Include F=0
    omega_arr = 2 * np.pi * freq_arr
    expected_adm_arr_val = 1j * omega_arr * cap_val
    expected_adm_arr = Quantity(expected_adm_arr_val, ureg.siemens)

    actual_adm_arr = c.get_admittance(freq_arr)

    assert actual_adm_arr.shape == freq_arr.shape
    # Y = jwC -> Y=0 at F=0
    assert_quantity_close(actual_adm_arr, expected_adm_arr, rtol=1e-7, atol=1e-12)


# --- Inductor Tests ---
def test_inductor_creation_and_params():
    """ Test basic inductor instantiation and parameter access. """
    params = {"inductance": Quantity("1 nH")}
    l = Inductor("L1", "Inductor", params)
    assert l.instance_id == "L1"
    assert l.inductance == Quantity("1 nH")

def test_inductor_admittance_vectorized():
    """ Test inductor admittance calculation with frequency array (including F=0). """
    ind_val = 1e-9 # 1 nH
    params = {"inductance": Quantity(ind_val, ureg.henry)}
    l = Inductor("L1", "Inductor", params)
    freq_arr = np.array([0.0, 1e9, 2e9]) # Include DC

    # Expected values
    expected_adm_arr_val = np.empty_like(freq_arr, dtype=complex)
    # F=0, L>0 -> Inf admittance (short) based on AC model limit
    expected_adm_arr_val[0] = np.inf + 0j
    # F>0 cases
    ac_mask = freq_arr > 0
    omega_arr_ac = 2 * np.pi * freq_arr[ac_mask]
    expected_adm_arr_val[ac_mask] = 1.0 / (1j * omega_arr_ac * ind_val)

    expected_adm_arr = Quantity(expected_adm_arr_val, ureg.siemens)
    actual_adm_arr = l.get_admittance(freq_arr)

    assert actual_adm_arr.shape == freq_arr.shape
    # assert_quantity_close handles the inf comparison
    assert_quantity_close(actual_adm_arr, expected_adm_arr, rtol=1e-7)


def test_inductor_zero_inductance_vectorized():
    """ Test zero inductance calculation with frequency array (including F=0). """
    params = {"inductance": Quantity("0 H")}
    l = Inductor("L0", "Inductor", params) # Should warn
    freq_arr = np.array([0.0, 1e9, 2e9]) # Include DC
    # L=0 -> infinite admittance for all frequencies based on AC model limit
    expected_adm_val = np.array([np.inf + 0j] * len(freq_arr))
    expected_adm = Quantity(expected_adm_val, ureg.siemens)

    actual_adm = l.get_admittance(freq_arr)

    assert actual_adm.shape == freq_arr.shape
    assert_quantity_close(actual_adm, expected_adm)


# --- Parameter Validation Tests (via CircuitBuilder) ---

def test_builder_valid_component_params(builder, param_manager):
    """ Test processing valid literal parameters. """
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
    """ Test processing valid global parameter references. """
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
    """ Test error handling for missing required parameters. """
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
    """ Test error handling for literal parameter with incorrect dimensions. """
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
    assert "meter" in str(excinfo.value) # Actual unit
    assert "ohm" in str(excinfo.value) # Expected unit/dimension

def test_builder_invalid_unit_global_ref(builder, param_manager):
    """ Test error handling for global ref with incorrect dimensions. """
    comp_data = ComponentData(
        instance_id="R_bad",
        component_type="Resistor",
        parameters={"resistance": "global_cap"} # Refers to capacitance param
    )
    with pytest.raises(pint.DimensionalityError) as excinfo:
         builder._process_component_parameters(
            comp_data, Resistor.declare_parameters(), param_manager
        )
    assert "picofarad" in str(excinfo.value) # Actual unit from global_cap
    assert "ohm" in str(excinfo.value) # Expected unit/dimension

def test_builder_unparseable_literal(builder, param_manager):
    """ Test error handling for unparseable literal parameter values. """
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
    """ Test error handling for unregistered component types during build. """
    # Create a dummy ParsedCircuitData object that includes a sweep section
    class MockParsedCircuit:
        name = "TestCircuit"
        parameter_manager = param_manager
        # Needs sweep to be considered valid by builder (even if not used here)
        frequency_sweep_hz = np.array([1e9])
        components = {
            "X1": ComponentData(instance_id="X1", component_type="NonExistent", parameters={}, ports={})
        }
        nets = {} # Need basic structure
        external_ports = {}
        external_port_impedances = {}
        ground_net_name = 'gnd'

    parsed_data = MockParsedCircuit()

    with pytest.raises(CircuitBuildError) as excinfo:
        builder.build_circuit(parsed_data)
    assert "Unknown component type 'NonExistent'" in str(excinfo.value)

def test_full_build_process(parser, builder, valid_yaml_string_for_build):
     """ Test parsing then building a valid circuit definition. """
     parsed_circuit = parser.parse(valid_yaml_string_for_build)
     # Should have raw components and frequency sweep
     assert not hasattr(parsed_circuit, 'sim_components')
     assert isinstance(parsed_circuit.components['R1'], ComponentData)
     assert hasattr(parsed_circuit, 'frequency_sweep_hz')

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