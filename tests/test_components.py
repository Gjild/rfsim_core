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
        # Use nan_to_num to compare non-NaN parts robustly
        np.testing.assert_allclose(np.nan_to_num(m1), np.nan_to_num(m2), rtol=rtol, atol=atol, err_msg=msg + " (comparing non-NaN parts)")
    else:
        np.testing.assert_allclose(m1, m2, rtol=rtol, atol=atol, err_msg=msg)

# Helper for comparing StampInfo
# tests/test_components.py
import pytest
import numpy as np
from pint import DimensionalityError
import logging
from typing import List, Tuple, Any # Added Any
import copy # For testing builder non-mutation

from rfsim_core import ureg, Quantity, pint
from rfsim_core.components import (
    Resistor, Capacitor, Inductor, ComponentError, ComponentBase, # Added ComponentBase
    LARGE_ADMITTANCE_SIEMENS
)
# Import new elements needed for StampInfo
from rfsim_core.components.base import StampInfo
from rfsim_core.components.elements import PORT_1, PORT_2 # Standard port names

from rfsim_core.parameters import ParameterManager, ParameterError
# Use Circuit for type hinting now
from rfsim_core.data_structures import Circuit, Component as ComponentData, Net, Port
from rfsim_core.circuit_builder import CircuitBuilder, CircuitBuildError

# Helper for comparing complex matrices (unchanged)
def assert_matrix_close(m1, m2, rtol=1e-5, atol=1e-8, msg=''):
    """Asserts element-wise closeness for complex numpy arrays."""
    assert isinstance(m1, np.ndarray), f"m1 is not a numpy array (type: {type(m1)}) {msg}"
    assert isinstance(m2, np.ndarray), f"m2 is not a numpy array (type: {type(m2)}) {msg}"
    assert m1.shape == m2.shape, f"Shapes differ: {m1.shape} vs {m2.shape} {msg}"
    nan_m1 = np.isnan(m1)
    nan_m2 = np.isnan(m2)
    if np.any(nan_m1) or np.any(nan_m2):
        assert np.array_equal(nan_m1, nan_m2), f"NaN patterns differ {msg}"
        # Use nan_to_num to compare non-NaN parts robustly
        np.testing.assert_allclose(np.nan_to_num(m1), np.nan_to_num(m2), rtol=rtol, atol=atol, err_msg=msg + " (comparing non-NaN parts)")
    else:
        np.testing.assert_allclose(m1, m2, rtol=rtol, atol=atol, err_msg=msg)


# Helper for comparing StampInfo (unchanged)
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
        if matrix_qty.check('siemens'): # Check dimension name
            matrix_mag = matrix_qty.to(ureg.siemens).magnitude
        else:
            matrix_mag = matrix_qty.to_base_units().magnitude

        assert isinstance(matrix_mag, np.ndarray), f"Stamp matrix magnitude is not a numpy array (type: {type(matrix_mag)}). {current_msg}"

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
    params = {"resistance": Quantity("50 ohm")}
    r = Resistor("R1", "Resistor", params)
    assert r.instance_id == "R1"
    assert r.get_parameter("resistance") == Quantity("50 ohm")


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

def test_resistor_zero_resistance_stamps_at_f_zero(caplog):
    """ Test zero resistance resistor stamps at F=0 (should still be large admittance). """
    params = {"resistance": Quantity("0 ohm")}
    with caplog.at_level(logging.WARNING):
        r = Resistor("R0", "Resistor", params)

    freq_zero = np.array([0.0])
    y_val = LARGE_ADMITTANCE_SIEMENS + 0j

    expected_mag = np.zeros((1, 2, 2), dtype=np.complex128)
    expected_mag[0, 0, 0] = y_val
    expected_mag[0, 0, 1] = -y_val
    expected_mag[0, 1, 0] = -y_val
    expected_mag[0, 1, 1] = y_val
    expected_stamps = [(expected_mag, [PORT_1, PORT_2])]

    actual_stamp_info_list = r.get_mna_stamps(freq_zero)
    assert_stamp_info_list_close(actual_stamp_info_list, expected_stamps)

def test_resistor_zero_resistance_stamps_at_f_zero(caplog):
    """ Test zero resistance resistor stamps at F=0 (should still be large admittance). """
    params = {"resistance": Quantity("0 ohm")}
    with caplog.at_level(logging.WARNING):
        r = Resistor("R0", "Resistor", params)
    assert any("Component 'R0' has zero resistance" in message for message in caplog.text.splitlines())

    freq_zero = np.array([0.0])
    y_val = LARGE_ADMITTANCE_SIEMENS + 0j
    expected_mag = np.array([[y_val, -y_val], [-y_val, y_val]], dtype=np.complex128)
    expected_stamps = [(expected_mag, [PORT_1, PORT_2])]

    actual_stamp_info_list = r.get_mna_stamps(freq_zero)
    assert_stamp_info_list_close(actual_stamp_info_list, expected_stamps)

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

def test_capacitor_infinite_capacitance_init(caplog):
    params = {"capacitance": Quantity("inf F")}
    with caplog.at_level(logging.WARNING):
         c = Capacitor("Cinf", "Capacitor", params) # Should not raise error
    assert c is not None
    assert "infinite capacitance" in caplog.text

def test_capacitor_infinite_capacitance_stamps(caplog):
    params = {"capacitance": Quantity("inf F")}
    with caplog.at_level(logging.WARNING):
        c = Capacitor("Cinf", "Capacitor", params)

    freq_arr = np.array([1e9, 2e9, 0.0]) # Include F=0
    y_val = LARGE_ADMITTANCE_SIEMENS + 0j
    num_freqs = len(freq_arr)
    expected_mag = np.zeros((num_freqs, 2, 2), dtype=np.complex128)
    expected_mag[:, 0, 0] = y_val
    expected_mag[:, 0, 1] = -y_val
    expected_mag[:, 1, 0] = -y_val
    expected_mag[:, 1, 1] = y_val
    expected_stamps = [(expected_mag, [PORT_1, PORT_2])]

    actual_stamp_info_list = c.get_mna_stamps(freq_arr)
    assert_stamp_info_list_close(actual_stamp_info_list, expected_stamps)

# Test capacitor stamps at F=0 (should be zero admittance for finite C)
def test_capacitor_stamps_at_f_zero():
    cap_val = 1e-12 # 1 pF
    params = {"capacitance": Quantity(cap_val, ureg.farad)}
    c = Capacitor("C1", "Capacitor", params)
    freq_zero = np.array([0.0])

    y_val = 0.0 + 0j # Expect zero admittance
    # --- EXPECT 3D Shape ---
    expected_mag = np.zeros((1, 2, 2), dtype=np.complex128)
    expected_mag[0, 0, 0] = y_val
    expected_mag[0, 0, 1] = -y_val
    expected_mag[0, 1, 0] = -y_val
    expected_mag[0, 1, 1] = y_val
    expected_stamps = [(expected_mag, [PORT_1, PORT_2])]

    actual_stamp_info_list = c.get_mna_stamps(freq_zero)
    assert_stamp_info_list_close(actual_stamp_info_list, expected_stamps)

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

# Test inductor stamps at F=0 (should be large admittance)
def test_inductor_stamps_at_f_zero():
    ind_val = 1e-9 # 1 nH
    params = {"inductance": Quantity(ind_val, ureg.henry)}
    l = Inductor("L1", "Inductor", params)
    freq_zero = np.array([0.0])

    y_val = LARGE_ADMITTANCE_SIEMENS + 0j # Expect large admittance
    # --- EXPECT 3D Shape ---
    expected_mag = np.zeros((1, 2, 2), dtype=np.complex128)
    expected_mag[0, 0, 0] = y_val
    expected_mag[0, 0, 1] = -y_val  
    expected_mag[0, 1, 0] = -y_val
    expected_mag[0, 1, 1] = y_val
    expected_stamps = [(expected_mag, [PORT_1, PORT_2])]

    # This test should now pass with the np.nan_to_num fix
    actual_stamp_info_list = l.get_mna_stamps(freq_zero)
    assert_stamp_info_list_close(actual_stamp_info_list, expected_stamps)

# *Test L=0 at F=0
def test_inductor_zero_inductance_stamps_at_f_zero(caplog):
    params = {"inductance": Quantity("0 H")}
    with caplog.at_level(logging.WARNING):
        l = Inductor("L0", "Inductor", params)
    # ... (log check unchanged) ...

    freq_zero = np.array([0.0])
    y_val = LARGE_ADMITTANCE_SIEMENS + 0j
    # --- EXPECT 3D Shape ---
    expected_mag = np.zeros((1, 2, 2), dtype=np.complex128)
    expected_mag[0, 0, 0] = y_val
    expected_mag[0, 0, 1] = -y_val
    expected_mag[0, 1, 0] = -y_val
    expected_mag[0, 1, 1] = y_val
    expected_stamps = [(expected_mag, [PORT_1, PORT_2])]

    actual_stamp_info_list = l.get_mna_stamps(freq_zero)
    assert_stamp_info_list_close(actual_stamp_info_list, expected_stamps)


# --- Parameter and Build Validation Tests ---
# These tests check CircuitBuilder which now performs port validation too.

# **NEW:** Test builder errors on undeclared parameter provided
def test_builder_undeclared_parameter(builder, param_manager):
    comp_data = ComponentData(
        instance_id="R_extra", component_type="Resistor",
        parameters={"resistance": "50 ohm", "extra_param": "10"}, # Has extra_param
        ports={PORT_1: 'n1', PORT_2: 'n2'}
    )
    # --- EXPECT CircuitBuildError ---
    with pytest.raises(CircuitBuildError) as excinfo:
         parsed_circuit = Circuit(
             name="Extra Param Test", parameter_manager=param_manager,
             components={"R_extra": comp_data},
             nets={'n1': Net(name='n1'), 'n2': Net(name='n2'), 'gnd': Net(name='gnd', is_ground=True)},
             external_ports={}, external_port_impedances={}, ground_net_name='gnd'
         )
         # Connect nets for port validation checks during build
         net1=parsed_circuit.nets['n1']; net2=parsed_circuit.nets['n2']
         # Ensure ports exist before assigning nets (might need to adjust test setup if ComponentData doesn't auto-create Port objects)
         # Assuming ComponentData's ports dict values are Port objects or created implicitly
         # If not, this setup needs more detail. Let's assume they are created by ComponentData for now.
         # It's safer to add ports explicitly if ComponentData doesn't.
         if PORT_1 not in comp_data.ports: comp_data.add_port(PORT_1)
         if PORT_2 not in comp_data.ports: comp_data.add_port(PORT_2)
         comp_data.ports[PORT_1] = net1
         comp_data.ports[PORT_2] = net2


         builder.build_circuit(parsed_circuit)
    # Check the underlying error message within CircuitBuildError
    assert "Parameter validation failed for component 'R_extra'" in str(excinfo.value)
    assert "parameter 'extra_param' which is not declared" in str(excinfo.value)

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
     r1_data = ComponentData(instance_id="R1", component_type="Resistor", parameters={"resistance": "1kohm"})
     c1_data = ComponentData(instance_id="C1", component_type="Capacitor", parameters={"capacitance": "global_cap"})
     l1_data = ComponentData(instance_id="L1", component_type="Inductor", parameters={"inductance": "1 uH"})

     # Add ports explicitly to ComponentData if needed
     r1_data.add_port(PORT_1); r1_data.add_port(PORT_2)
     c1_data.add_port(PORT_1); c1_data.add_port(PORT_2)
     l1_data.add_port(PORT_1); l1_data.add_port(PORT_2)

     net1 = Net(name='net1'); gnd = Net(name='gnd', is_ground=True); out = Net(name='out')

     # Assign ports to nets after creating components and nets
     r1_data.ports[PORT_1].net = net1; r1_data.ports[PORT_2].net = gnd
     c1_data.ports[PORT_1].net = net1; c1_data.ports[PORT_2].net = gnd
     l1_data.ports[PORT_1].net = net1; l1_data.ports[PORT_2].net = out

     parsed_circuit = Circuit(
         name="Test Build", parameter_manager=param_manager,
         components={"R1": r1_data, "C1": c1_data, "L1": l1_data},
         nets={'net1': net1, 'gnd': gnd, 'out': out},
         external_ports={}, external_port_impedances={}, ground_net_name='gnd'
     )
     # Make a copy *only* for checking mutation if needed, but simpler check is better
     original_parsed_circuit = copy.deepcopy(parsed_circuit) # Keep original state

     # --- Call build_circuit ---
     built_circuit = builder.build_circuit(original_parsed_circuit) # Pass the original

     # --- Check returned built_circuit ---
     assert built_circuit is not original_parsed_circuit # Should be a new object
     assert hasattr(built_circuit, 'sim_components')
     assert len(built_circuit.sim_components) == 3
     assert isinstance(built_circuit.sim_components['R1'], Resistor)

     # **REVISED:** Simplified mutation check: Original object should lack sim_components
     assert not hasattr(original_parsed_circuit, 'sim_components'), \
         "Input circuit should not have sim_components added by builder"

     r1_sim = built_circuit.sim_components['R1']
     assert r1_sim.get_parameter('resistance') == Quantity('1 kohm')
     c1_sim = built_circuit.sim_components['C1']
     assert c1_sim.get_parameter('capacitance') == Quantity('10 pF')
     l1_sim = built_circuit.sim_components['L1']
     assert l1_sim.get_parameter('inductance') == Quantity('1 uH')


def test_build_invalid_component_port_id(builder, param_manager):
    """ Test build fails if component instance uses undeclared port IDs. """
    r1_data = ComponentData(
        instance_id="R_bad_port", component_type="Resistor",
        parameters={"resistance": "1kohm"},
        ports={'p1':'net1', 'p2':'gnd'} # Using strings 'p1', 'p2' instead of declared ints 0, 1
    )
     # Need nets for context
    net1 = Net(name='net1'); gnd = Net(name='gnd', is_ground=True)
    r1_data.ports['p1'] = net1; r1_data.ports['p2'] = gnd

    parsed_circuit = Circuit(
        name="Bad Port ID", parameter_manager=param_manager,
        components={"R_bad_port": r1_data},
        nets={'net1': net1, 'gnd': gnd},
        external_ports={}, external_port_impedances={}, ground_net_name='gnd'
    )

    with pytest.raises(CircuitBuildError) as excinfo:
        builder.build_circuit(parsed_circuit)
    assert "uses undeclared ports: ['p1', 'p2']" in str(excinfo.value) # Check exact error message
    assert "Declared ports are: [0, 1]" in str(excinfo.value) # Show the declared ports

# Test build fails if declared ports are missing connections
def test_build_missing_required_port(builder, param_manager):
    """ Test build fails if a component instance omits a declared port connection. """
    # Resistor declares ports [0, 1]
    r1_data = ComponentData(
        instance_id="R_missing_port", component_type="Resistor",
        parameters={"resistance": "1kohm"},
        ports={PORT_1: 'net1'} # Only port 0 is connected, port 1 is missing
    )
    net1 = Net(name='net1'); gnd = Net(name='gnd', is_ground=True) # Dummy gnd needed
    r1_data.ports[PORT_1] = net1

    parsed_circuit = Circuit(
        name="Missing Port", parameter_manager=param_manager,
        components={"R_missing_port": r1_data},
        nets={'net1': net1, 'gnd': gnd},
        external_ports={}, external_port_impedances={}, ground_net_name='gnd'
    )

    with pytest.raises(CircuitBuildError) as excinfo:
        builder.build_circuit(parsed_circuit)
    assert "is missing required connections for declared ports: [1]" in str(excinfo.value) # Port 1 is missing
    assert "Connected ports are: [0]" in str(excinfo.value)

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