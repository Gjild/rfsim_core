# tests/test_components.py
import pytest
import numpy as np
from pint import DimensionalityError
import logging
from typing import List, Tuple, Any # Added

from rfsim_core import ureg, Quantity, pint
from rfsim_core.components import (
    Resistor, Capacitor, Inductor, ComponentError, ComponentBase, # Added ComponentBase
    LARGE_ADMITTANCE_SIEMENS
)
# Import new elements needed for StampInfo
from rfsim_core.components.base import StampInfo
from rfsim_core.components.elements import PORT_1, PORT_2 # Standard port names

from rfsim_core.parameters import ParameterManager, ParameterError
from rfsim_core.data_structures import Circuit as ParsedCircuitData
from rfsim_core.data_structures import Component as ComponentData
from rfsim_core.data_structures import Net
from rfsim_core.circuit_builder import CircuitBuilder, CircuitBuildError

# Helper for comparing complex matrices (can reuse from test_simulation)
def assert_matrix_close(m1, m2, rtol=1e-5, atol=1e-8, msg=''):
    """Asserts element-wise closeness for complex numpy arrays."""
    assert isinstance(m1, np.ndarray), f"m1 is not a numpy array (type: {type(m1)}) {msg}"
    assert isinstance(m2, np.ndarray), f"m2 is not a numpy array (type: {type(m2)}) {msg}"
    assert m1.shape == m2.shape, f"Shapes differ: {m1.shape} vs {m2.shape} {msg}"
    nan_m1 = np.isnan(m1)
    nan_m2 = np.isnan(m2)
    if np.any(nan_m1) or np.any(nan_m2):
        assert np.array_equal(nan_m1, nan_m2), f"NaN patterns differ {msg}"
        np.testing.assert_allclose(m1[~nan_m1], m2[~nan_m2], rtol=rtol, atol=atol, err_msg=msg)
    else:
        np.testing.assert_allclose(m1, m2, rtol=rtol, atol=atol, err_msg=msg)

# Helper for comparing StampInfo
def assert_stamp_info_list_close(
    stamp_info_list: List[StampInfo],
    expected_stamps: List[Tuple[np.ndarray, List[str | int]]],
    expected_dim: str = 'siemens',
    rtol=1e-5, atol=1e-8, msg=''
):
    """Asserts correctness of a list of StampInfo tuples."""
    assert len(stamp_info_list) == len(expected_stamps), \
        f"Number of stamps differ. Got {len(stamp_info_list)}, expected {len(expected_stamps)}. {msg}"

    for idx, (stamp_info, expected_stamp) in enumerate(zip(stamp_info_list, expected_stamps)):
        matrix_qty, port_ids = stamp_info
        expected_matrix_mag, expected_ports = expected_stamp
        current_msg = f"Stamp index {idx}. {msg}"

        # 1. Check Port IDs
        assert port_ids == expected_ports, f"Port ID mismatch. Got {port_ids}, expected {expected_ports}. {current_msg}"

        # 2. Check Matrix Quantity Type and Dimension
        assert isinstance(matrix_qty, Quantity), f"Stamp matrix is not a Quantity (type: {type(matrix_qty)}). {current_msg}"
        assert matrix_qty.check(expected_dim), f"Stamp matrix has wrong dimension. Got {matrix_qty.dimensionality}, expected {expected_dim}. {current_msg}"

        # 3. Check Matrix Magnitude
        # Convert to Siemens for comparison if dim is admittance
        if matrix_qty.check('siemens'):
            matrix_mag = matrix_qty.to(ureg.siemens).magnitude
        else:
            matrix_mag = matrix_qty.to_base_units().magnitude # Compare in base units otherwise

        assert isinstance(matrix_mag, np.ndarray), f"Stamp matrix magnitude is not a numpy array (type: {type(matrix_mag)}). {current_msg}"

        # Use the matrix comparison helper
        assert_matrix_close(matrix_mag, expected_matrix_mag, rtol=rtol, atol=atol, msg=f"Stamp matrix magnitude mismatch. {current_msg}")


# Helper for scalar/1D Quantities (Keep for simple parameter checks)
def assert_quantity_close(q1, q2, rtol=1e-5, atol=1e-8, msg=''):
    assert isinstance(q1, Quantity), f"q1 is not a Quantity (type: {type(q1)}) {msg}"
    assert isinstance(q2, Quantity), f"q2 is not a Quantity (type: {type(q2)}) {msg}"
    assert q1.is_compatible_with(q2), f"Quantities incompatible: {q1.dimensionality} vs {q2.dimensionality} {msg}"
    m1 = q1.magnitude
    m2 = q2.to(q1.units).magnitude
    if not isinstance(m1, np.ndarray): m1 = np.array(m1)
    if not isinstance(m2, np.ndarray): m2 = np.array(m2)
    assert m1.shape == m2.shape, f"Shapes differ: {m1.shape} vs {m2.shape} {msg}"
    np.testing.assert_allclose(m1, m2, rtol=rtol, atol=atol, err_msg=msg)


# --- Fixtures ---
@pytest.fixture
def param_manager():
    return ParameterManager({"global_res": "1 kohm", "global_cap": "10 pF"})

@pytest.fixture
def builder():
    return CircuitBuilder()

# --- Component Declaration Tests ---

@pytest.mark.parametrize("CompClass, expected_ports", [
    (Resistor, [PORT_1, PORT_2]),
    (Capacitor, [PORT_1, PORT_2]),
    (Inductor, [PORT_1, PORT_2]),
])
def test_component_port_declaration(CompClass, expected_ports):
    """Verify components declare correct ports."""
    assert CompClass.declare_ports() == expected_ports

@pytest.mark.parametrize("CompClass, expected_connectivity", [
    (Resistor, [(PORT_1, PORT_2)]),
    (Capacitor, [(PORT_1, PORT_2)]),
    (Inductor, [(PORT_1, PORT_2)]),
    # Add tests for future multi-terminal components here
])
def test_component_connectivity_declaration(CompClass, expected_connectivity):
    """Verify components declare correct connectivity for sparsity."""
    # Need to handle potential ordering differences in tuples if not guaranteed
    result_conn = CompClass.declare_connectivity()
    assert set(tuple(sorted(pair)) for pair in result_conn) == \
           set(tuple(sorted(pair)) for pair in expected_connectivity)


# --- Resistor Tests ---
def test_resistor_creation_and_params():
    """ Test basic resistor instantiation and parameter access. """
    params = {"resistance": Quantity("50 ohm")}
    r = Resistor("R1", "Resistor", params)
    assert r.instance_id == "R1"
    assert r.component_type == "Resistor"
    # Access parameter via internal dict or getter if preferred
    assert r.get_parameter("resistance") == Quantity("50 ohm")
    # Internal attribute access might change, use getter if possible
    assert r._params["resistance"] == Quantity("50 ohm") # Check internal storage too


def test_resistor_stamps_vectorized():
    """ Test resistor MNA stamps calculation with frequency array (F>0). """
    R_val = 50.0
    params = {"resistance": Quantity(R_val, ureg.ohm)}
    r = Resistor("R1", "Resistor", params)
    freq_arr = np.array([1e9, 2e9, 3e9])

    # Expected stamp matrix magnitude (broadcast)
    G = 1.0 / R_val
    y_val = G + 0j
    num_freqs = len(freq_arr)
    expected_mag = np.zeros((num_freqs, 2, 2), dtype=np.complex128)
    expected_mag[:, 0, 0] = y_val
    expected_mag[:, 0, 1] = -y_val
    expected_mag[:, 1, 0] = -y_val
    expected_mag[:, 1, 1] = y_val

    expected_stamps = [(expected_mag, [PORT_1, PORT_2])] # List containing one stamp tuple

    actual_stamp_info_list = r.get_mna_stamps(freq_arr)

    assert_stamp_info_list_close(actual_stamp_info_list, expected_stamps)

def test_resistor_stamps_scalar_freq():
    """ Test resistor MNA stamps calculation with scalar frequency (F>0). """
    R_val = 50.0
    params = {"resistance": Quantity(R_val, ureg.ohm)}
    r = Resistor("R1", "Resistor", params)
    freq_scalar = np.array([1e9]) # Pass as 1-element array

    # Expected stamp matrix magnitude (scalar)
    G = 1.0 / R_val
    y_val = G + 0j
    expected_mag = np.array([[y_val, -y_val], [-y_val, y_val]], dtype=np.complex128)
    expected_stamps = [(expected_mag, [PORT_1, PORT_2])]

    actual_stamp_info_list = r.get_mna_stamps(freq_scalar)

    assert_stamp_info_list_close(actual_stamp_info_list, expected_stamps)


def test_resistor_zero_resistance_stamps_vectorized(caplog):
    """ Test zero resistance resistor stamps (uses LARGE_ADMITTANCE_SIEMENS). """
    params = {"resistance": Quantity("0 ohm")}
    with caplog.at_level(logging.WARNING):
        r = Resistor("R0", "Resistor", params)
    assert any("Component 'R0' has zero resistance" in message for message in caplog.text.splitlines())

    freq_arr = np.array([1e9, 2e9])
    y_val = LARGE_ADMITTANCE_SIEMENS + 0j
    num_freqs = len(freq_arr)
    expected_mag = np.zeros((num_freqs, 2, 2), dtype=np.complex128)
    expected_mag[:, 0, 0] = y_val
    expected_mag[:, 0, 1] = -y_val
    expected_mag[:, 1, 0] = -y_val
    expected_mag[:, 1, 1] = y_val
    expected_stamps = [(expected_mag, [PORT_1, PORT_2])]

    actual_stamp_info_list = r.get_mna_stamps(freq_arr)
    assert_stamp_info_list_close(actual_stamp_info_list, expected_stamps)


def test_resistor_negative_resistance():
     """ Test instantiation fails with negative resistance (check in __init__). """
     params = {"resistance": Quantity("-50 ohm")}
     with pytest.raises(ComponentError, match="Resistance must be real and non-negative"):
         Resistor("Rneg", "Resistor", params)

def test_resistor_get_stamps_requires_numpy_array():
    params = {"resistance": Quantity("50 ohm")}
    r = Resistor("R1", "Resistor", params)
    with pytest.raises(TypeError, match="freq_hz must be a NumPy array"):
        r.get_mna_stamps(1e9) # Pass float


# --- Capacitor Tests ---
def test_capacitor_creation_and_params():
    params = {"capacitance": Quantity("1 pF")}
    c = Capacitor("C1", "Capacitor", params)
    assert c.instance_id == "C1"
    assert c.get_parameter("capacitance") == Quantity("1 pF")

def test_capacitor_stamps_vectorized():
    cap_val = 1e-12 # 1 pF
    params = {"capacitance": Quantity(cap_val, ureg.farad)}
    c = Capacitor("C1", "Capacitor", params)
    freq_arr = np.array([1e9, 2e9, 3e9])
    omega_arr = 2 * np.pi * freq_arr
    y_val_array = 1j * omega_arr * cap_val

    num_freqs = len(freq_arr)
    expected_mag = np.zeros((num_freqs, 2, 2), dtype=np.complex128)
    expected_mag[:, 0, 0] = y_val_array
    expected_mag[:, 0, 1] = -y_val_array
    expected_mag[:, 1, 0] = -y_val_array
    expected_mag[:, 1, 1] = y_val_array
    expected_stamps = [(expected_mag, [PORT_1, PORT_2])]

    actual_stamp_info_list = c.get_mna_stamps(freq_arr)
    assert_stamp_info_list_close(actual_stamp_info_list, expected_stamps, rtol=1e-7, atol=1e-12)


def test_capacitor_zero_capacitance_stamps():
    params = {"capacitance": Quantity("0 F")}
    c = Capacitor("C0", "Capacitor", params)
    freq_arr = np.array([1e9, 2e9])

    num_freqs = len(freq_arr)
    expected_mag = np.zeros((num_freqs, 2, 2), dtype=np.complex128) # Expect all zeros
    expected_stamps = [(expected_mag, [PORT_1, PORT_2])]

    actual_stamp_info_list = c.get_mna_stamps(freq_arr)
    assert_stamp_info_list_close(actual_stamp_info_list, expected_stamps)

def test_capacitor_negative_capacitance():
    params = {"capacitance": Quantity("-1 pF")}
    with pytest.raises(ComponentError, match="Capacitance must be real and non-negative"):
        Capacitor("Cneg", "Capacitor", params)

def test_capacitor_get_stamps_requires_numpy_array():
    params = {"capacitance": Quantity("1 pF")}
    c = Capacitor("C1", "Capacitor", params)
    with pytest.raises(TypeError, match="freq_hz must be a NumPy array"):
        c.get_mna_stamps(1e9)


# --- Inductor Tests ---
def test_inductor_creation_and_params():
    params = {"inductance": Quantity("1 nH")}
    l = Inductor("L1", "Inductor", params)
    assert l.instance_id == "L1"
    assert l.get_parameter("inductance") == Quantity("1 nH")

def test_inductor_stamps_vectorized():
    ind_val = 1e-9 # 1 nH
    params = {"inductance": Quantity(ind_val, ureg.henry)}
    l = Inductor("L1", "Inductor", params)
    freq_arr = np.array([1e9, 2e9, 3e9])
    omega_arr = 2 * np.pi * freq_arr
    y_val_array = 1.0 / (1j * omega_arr * ind_val)

    num_freqs = len(freq_arr)
    expected_mag = np.zeros((num_freqs, 2, 2), dtype=np.complex128)
    expected_mag[:, 0, 0] = y_val_array
    expected_mag[:, 0, 1] = -y_val_array
    expected_mag[:, 1, 0] = -y_val_array
    expected_mag[:, 1, 1] = y_val_array
    expected_stamps = [(expected_mag, [PORT_1, PORT_2])]

    actual_stamp_info_list = l.get_mna_stamps(freq_arr)
    assert_stamp_info_list_close(actual_stamp_info_list, expected_stamps, rtol=1e-7)


def test_inductor_zero_inductance_stamps_vectorized(caplog):
    params = {"inductance": Quantity("0 H")}
    with caplog.at_level(logging.WARNING):
        l = Inductor("L0", "Inductor", params)
    assert any("Component 'L0' has zero inductance" in message for message in caplog.text.splitlines())

    freq_arr = np.array([1e9, 2e9])
    y_val = LARGE_ADMITTANCE_SIEMENS + 0j
    num_freqs = len(freq_arr)
    expected_mag = np.zeros((num_freqs, 2, 2), dtype=np.complex128)
    expected_mag[:, 0, 0] = y_val
    expected_mag[:, 0, 1] = -y_val
    expected_mag[:, 1, 0] = -y_val
    expected_mag[:, 1, 1] = y_val
    expected_stamps = [(expected_mag, [PORT_1, PORT_2])]

    actual_stamp_info_list = l.get_mna_stamps(freq_arr)
    assert_stamp_info_list_close(actual_stamp_info_list, expected_stamps)


def test_inductor_negative_inductance():
    params = {"inductance": Quantity("-1 nH")}
    with pytest.raises(ComponentError, match="Inductance must be real and non-negative"):
        Inductor("Lneg", "Inductor", params)

def test_inductor_get_stamps_requires_numpy_array():
    params = {"inductance": Quantity("1 nH")}
    l = Inductor("L1", "Inductor", params)
    with pytest.raises(TypeError, match="freq_hz must be a NumPy array"):
        l.get_mna_stamps(1e9)

def test_inductor_get_stamps_requires_positive_freq():
    """ Test that get_mna_stamps fails if F<=0 is passed (component responsibility). """
    params = {"inductance": Quantity("1 nH")}
    l = Inductor("L1", "Inductor", params)
    freq_arr_zero = np.array([0.0, 1e9])
    freq_arr_neg = np.array([-1e9, 1e9])
    with pytest.raises(ComponentError, match="AC analysis frequency must be > 0 Hz"):
        l.get_mna_stamps(freq_arr_zero)
    with pytest.raises(ComponentError, match="AC analysis frequency must be > 0 Hz"):
        l.get_mna_stamps(freq_arr_neg)


# --- Parameter and Build Validation Tests ---
# These tests check CircuitBuilder which now performs port validation too.

def test_builder_valid_component_params(builder, param_manager):
    # This test focuses on parameter processing, unchanged fundamentally
    comp_data = ComponentData(
        instance_id="R1", component_type="Resistor",
        parameters={"resistance": "50 ohm"},
        ports={PORT_1: 'n1', PORT_2: 'n2'} # Need valid ports for builder
    )
    # Ensure RLC types declare their ports
    Resistor.declare_ports()
    processed = builder._process_component_parameters(comp_data, Resistor.declare_parameters(), param_manager)
    assert "resistance" in processed
    assert processed["resistance"] == Quantity("50 ohm")

def test_builder_valid_global_ref(builder, param_manager):
    # This test focuses on parameter processing, unchanged fundamentally
    comp_data = ComponentData(
        instance_id="R2", component_type="Resistor",
        parameters={"resistance": "global_res"},
        ports={PORT_1: 'n1', PORT_2: 'n2'} # Need valid ports for builder
    )
    processed = builder._process_component_parameters(comp_data, Resistor.declare_parameters(), param_manager)
    assert processed["resistance"] == Quantity("1 kohm")

def test_builder_missing_parameter(builder, param_manager):
    comp_data = ComponentData(
        instance_id="R_bad", component_type="Resistor",
        parameters={}, ports={PORT_1: 'n1', PORT_2: 'n2'} # Need valid ports
    )
    with pytest.raises(ParameterError, match="Required parameter 'resistance' missing"):
        builder._process_component_parameters(comp_data, Resistor.declare_parameters(), param_manager)

def test_builder_invalid_unit_literal(builder, param_manager):
    comp_data = ComponentData(
        instance_id="R_bad", component_type="Resistor",
        parameters={"resistance": "50 meter"}, ports={PORT_1: 'n1', PORT_2: 'n2'}
    )
    with pytest.raises(DimensionalityError) as excinfo:
         builder._process_component_parameters(comp_data, Resistor.declare_parameters(), param_manager)
    # Error message check remains the same

def test_builder_invalid_unit_global_ref(builder, param_manager):
    comp_data = ComponentData(
        instance_id="R_bad", component_type="Resistor",
        parameters={"resistance": "global_cap"}, ports={PORT_1: 'n1', PORT_2: 'n2'}
    )
    with pytest.raises(pint.DimensionalityError) as excinfo:
         builder._process_component_parameters(comp_data, Resistor.declare_parameters(), param_manager)
    # Error message check remains the same

def test_builder_unparseable_literal(builder, param_manager):
    comp_data = ComponentData(
        instance_id="C_bad", component_type="Capacitor",
        parameters={"capacitance": "10 picoFaraday"}, ports={PORT_1: 'n1', PORT_2: 'n2'}
    )
    with pytest.raises(ParameterError, match="Error parsing literal value"):
        builder._process_component_parameters(comp_data, Capacitor.declare_parameters(), param_manager)

# --- Full Build Process Tests (including port validation) ---

def test_full_build_process_valid(builder, param_manager):
     """ Test building a valid circuit structure. """
     # Using integer ports as defined in elements.py
     r1_data = ComponentData(instance_id="R1", component_type="Resistor", parameters={"resistance": "1kohm"}, ports={PORT_1:'net1', PORT_2:'gnd'})
     c1_data = ComponentData(instance_id="C1", component_type="Capacitor", parameters={"capacitance": "global_cap"}, ports={PORT_1:'net1', PORT_2:'gnd'})
     l1_data = ComponentData(instance_id="L1", component_type="Inductor", parameters={"inductance": "1 uH"}, ports={PORT_1:'net1', PORT_2:'out'})

     parsed_circuit = ParsedCircuitData(
         name="Test Build", parameter_manager=param_manager,
         components={"R1": r1_data, "C1": c1_data, "L1": l1_data},
         # Add dummy net data for completeness, though builder doesn't use it directly
         nets={'net1': Net, 'gnd': Net, 'out': Net}, # Use Any as placeholder for Net objects
         external_ports={}, external_port_impedances={}, ground_net_name='gnd'
     )

     built_circuit = builder.build_circuit(parsed_circuit) # Should succeed

     assert hasattr(built_circuit, 'sim_components')
     assert len(built_circuit.sim_components) == 3
     assert isinstance(built_circuit.sim_components['R1'], Resistor)
     # ... (parameter checks remain the same)
     r1_sim = built_circuit.sim_components['R1']
     assert r1_sim.get_parameter('resistance') == Quantity('1 kohm')
     c1_sim = built_circuit.sim_components['C1']
     assert c1_sim.get_parameter('capacitance') == Quantity('10 pF')
     l1_sim = built_circuit.sim_components['L1']
     assert l1_sim.get_parameter('inductance') == Quantity('1 uH')


def test_build_invalid_component_port_id(builder, param_manager):
    """ Test build fails if component instance uses undeclared port IDs. """
    # Resistor declares ports [0, 1]
    r1_data = ComponentData(
        instance_id="R_bad_port", component_type="Resistor",
        parameters={"resistance": "1kohm"},
        ports={'p1':'net1', 'p2':'gnd'} # Using strings 'p1', 'p2' instead of ints 0, 1
    )
    parsed_circuit = ParsedCircuitData(
        name="Bad Port ID", parameter_manager=param_manager,
        components={"R_bad_port": r1_data},
        nets={'net1': Net, 'gnd': Net},
        external_ports={}, external_port_impedances={}, ground_net_name='gnd'
    )

    with pytest.raises(CircuitBuildError) as excinfo:
        builder.build_circuit(parsed_circuit)
    assert "uses undeclared ports" in str(excinfo.value)
    assert "'p1'" in str(excinfo.value) or "'p2'" in str(excinfo.value)
    assert "[0, 1]" in str(excinfo.value) # Show the declared ports


def test_build_unknown_component_type(builder, param_manager):
    """ Test build fails for unregistered component types. """
    # Redundant with test_builder_unknown_component_type but good to have full build check
    x1_data = ComponentData(
        instance_id="X1", component_type="NonExistent",
        parameters={}, ports={'a': 'n1', 'b': 'n2'} # Dummy ports
    )
    parsed_circuit = ParsedCircuitData(
        name="Unknown Type Build", parameter_manager=param_manager,
        components={"X1": x1_data},
        nets={'n1': Net, 'n2': Net, 'gnd': Net},
        external_ports={}, external_port_impedances={}, ground_net_name='gnd'
    )
    with pytest.raises(CircuitBuildError) as excinfo:
        builder.build_circuit(parsed_circuit)
    # Note: This check might now happen earlier in the parser if registry is checked there.
    # If parser catches it, this test might need adjustment or removal.
    # Let's assume builder re-checks or parser doesn't have registry access.
    assert "Uknown Component Type: NonExistent" in str(excinfo.value)