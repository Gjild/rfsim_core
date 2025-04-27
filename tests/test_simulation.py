# tests/test_simulation.py
import pytest
import numpy as np
import logging
from rfsim_core import (
    NetlistParser,
    CircuitBuilder,
    run_simulation,
    run_sweep,
    SimulationError,
    MnaInputError,
    SingularMatrixError,
    ureg, Quantity,
    ParsingError, SchemaValidationError,
    CircuitBuildError # Added for testing build failures impacting simulation
)
from rfsim_core.components.elements import PORT_1, PORT_2 # Import standard port names

# Helper for comparing complex matrices (Keep as is)
def assert_matrix_close(m1, m2, rtol=1e-5, atol=1e-8, msg=''):
    # ... (implementation unchanged) ...
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


# --- Fixtures ---
@pytest.fixture
def parser():
    return NetlistParser()

@pytest.fixture
def builder():
    return CircuitBuilder()

# --- Helper Function to Parse and Build ---
def parse_and_build(parser, builder, yaml_netlist):
    """ Parses YAML and builds the circuit. Returns (built_circuit, freq_array). """
    circuit_data, freq_array = parser.parse(yaml_netlist)
    built_circuit = builder.build_circuit(circuit_data)
    return built_circuit, freq_array

# --- Single Frequency Tests (using run_simulation helper) ---

@pytest.fixture
def freq_1ghz():
    return 1e9

# Test cases need to use the correct port identifiers (0, 1) for RLC components

def test_single_freq_resistor_series(parser, builder, freq_1ghz):
    yaml_netlist = f"""
sweep: {{ type: list, points: ['1 GHz'] }}
components:
  - type: Resistor
    id: R1
    ports: {{ {PORT_1}: P1, {PORT_2}: P2 }} # Use integer ports
    parameters: {{resistance: '100 ohm'}}
ports:
  - {{id: P1, reference_impedance: '50 ohm'}}
  - {{id: P2, reference_impedance: '50 ohm'}}
"""
    built_circuit, _ = parse_and_build(parser, builder, yaml_netlist)
    y_matrix = run_simulation(built_circuit, freq_1ghz)
    # Analytical expectation remains the same
    R = 100.0
    G = 1.0 / R
    Y_expected = np.array([[ G, -G], [-G,  G]], dtype=complex)
    assert_matrix_close(y_matrix, Y_expected, rtol=1e-6, msg="Series R intrinsic Y")

def test_single_freq_resistor_shunt(parser, builder, freq_1ghz):
    yaml_netlist = f"""
sweep: {{ type: list, points: ['1 GHz'] }}
components:
  - type: Resistor
    id: R1
    ports: {{ {PORT_1}: P1, {PORT_2}: gnd }} # Use integer ports
    parameters: {{resistance: '25 ohm'}}
ports:
  - {{id: P1, reference_impedance: '50 ohm'}}
ground_net: 'gnd'
"""
    built_circuit, _ = parse_and_build(parser, builder, yaml_netlist)
    y_matrix = run_simulation(built_circuit, freq_1ghz)
    # Analytical expectation remains the same
    R = 25.0
    G = 1.0 / R
    Y_expected = np.array([[G]], dtype=complex)
    assert_matrix_close(y_matrix, Y_expected, msg="Shunt R intrinsic Y")


def test_single_freq_voltage_divider(parser, builder, freq_1ghz):
    yaml_netlist = f"""
sweep: {{ type: list, points: ['1 GHz'] }}
components:
  - type: Resistor
    id: R1
    ports: {{ {PORT_1}: P1, {PORT_2}: mid }} # Use integer ports
    parameters: {{resistance: '50 ohm'}}
  - type: Resistor
    id: R2
    ports: {{ {PORT_1}: mid, {PORT_2}: gnd }} # Use integer ports
    parameters: {{resistance: '50 ohm'}}
ports:
  - {{id: P1, reference_impedance: '50 ohm'}}
ground_net: 'gnd'
"""
    built_circuit, _ = parse_and_build(parser, builder, yaml_netlist)
    y_matrix = run_simulation(built_circuit, freq_1ghz)
    # Analytical expectation remains the same
    R1 = 50.0
    R2 = 50.0
    Y_expected = np.array([[1.0 / (R1 + R2)]], dtype=complex)
    assert_matrix_close(y_matrix, Y_expected, msg="Voltage Divider intrinsic Y")


def test_single_freq_lc_circuit(parser, builder, freq_1ghz):
    # Resonance test (adjust C value if needed based on float precision)
    C_val_str = '5.0660591821168885721939731604863819452179387336123274422804515947086 pF'
    yaml_netlist = f"""
sweep: {{ type: list, points: ['1 GHz'] }}
components:
  - type: Inductor
    id: L1
    ports: {{ {PORT_1}: P1, {PORT_2}: mid }} # Use integer ports
    parameters: {{inductance: '5 nH'}}
  - type: Capacitor
    id: C1
    ports: {{ {PORT_1}: mid, {PORT_2}: P2 }} # Use integer ports
    parameters: {{capacitance: '{C_val_str}'}}
ports:
  - {{id: P1, reference_impedance: '50 ohm'}}
  - {{id: P2, reference_impedance: '50 ohm'}}
"""
    built_circuit, _ = parse_and_build(parser, builder, yaml_netlist)
    y_matrix = run_simulation(built_circuit, freq_1ghz)
    # Analytical expectation (singularity) remains the same
    magnitude_threshold = 1e12
    acceptable = np.isnan(y_matrix) | (np.abs(y_matrix) > magnitude_threshold)
    assert np.all(acceptable), f"Y-matrix has unexpected entries at resonance:\n{y_matrix}"


def test_single_freq_lc_circuit_off_resonance(parser, builder):
    freq_off = 0.5e9
    L_val_str = '7.957747 nH'
    C_val_str = '3.183099 pF'
    yaml_netlist = f"""
sweep: {{ type: list, points: ['{freq_off/1e9} GHz'] }}
components:
  - type: Inductor
    id: L1
    ports: {{ {PORT_1}: P1, {PORT_2}: mid }} # Use integer ports
    parameters: {{inductance: '{L_val_str}'}}
  - type: Capacitor
    id: C1
    ports: {{ {PORT_1}: mid, {PORT_2}: P2 }} # Use integer ports
    parameters: {{capacitance: '{C_val_str}'}}
ports:
  - {{id: P1, reference_impedance: '50 ohm'}}
  - {{id: P2, reference_impedance: '50 ohm'}}
"""
    built_circuit, _ = parse_and_build(parser, builder, yaml_netlist)
    y_matrix = run_simulation(built_circuit, freq_off)
    # Analytical expectation remains the same
    freq = freq_off
    L_val = ureg.Quantity(L_val_str).to(ureg.henry).magnitude
    C_val = ureg.Quantity(C_val_str).to(ureg.farad).magnitude
    omega = 2 * np.pi * freq
    ZL = 1j * omega * L_val
    ZC = 1 / (1j * omega * C_val)
    Z_series = ZL + ZC
    Y_series = 1.0 / Z_series
    Y_expected = Y_series * np.array([[1, -1], [-1, 1]], dtype=complex)
    assert_matrix_close(y_matrix, Y_expected, rtol=1e-5, msg="Series LC off-resonance intrinsic Y")


# --- Sweep Specific Tests ---

def test_sweep_simple_resistor_shunt(parser, builder):
    yaml_netlist = f"""
sweep: {{ type: list, points: ['1 MHz', '500 MHz', '1 GHz'] }}
components:
  - type: Resistor
    id: R1
    ports: {{ {PORT_1}: P1, {PORT_2}: gnd }} # Use integer ports
    parameters: {{resistance: '25 ohm'}}
ports:
  - {{id: P1, reference_impedance: '50 ohm'}}
ground_net: 'gnd'
"""
    built_circuit, freq_array = parse_and_build(parser, builder, yaml_netlist)
    freqs_out, y_matrices = run_sweep(built_circuit, freq_array)
    # Analytical expectations remain the same
    assert len(freqs_out) == 3
    assert y_matrices.shape == (3, 1, 1)
    np.testing.assert_allclose(freqs_out, [1e6, 5e8, 1e9])
    R = 25.0
    G = 1.0 / R
    Y_expected = np.array([[G]], dtype=complex)
    for i in range(len(freqs_out)):
        assert not np.any(np.isnan(y_matrices[i,:,:]))
        assert_matrix_close(y_matrices[i,:,:], Y_expected, msg=f"Shunt R @ {freqs_out[i]:.2e} Hz")


def test_sweep_lc_circuit_detailed(parser, builder):
    # Resonance test (adjust C value if needed based on float precision)
    C_val_str = '5.0660591821168885721939731604863819452179387336123274422804515947086 pF'
    L_val_str = '5 nH'
    yaml_netlist = f"""
sweep: {{ type: linear, start: '0.8 GHz', stop: '1.2 GHz', num_points: 5 }}
components:
  - type: Inductor
    id: L1
    ports: {{ {PORT_1}: P1, {PORT_2}: mid }} # Use integer ports
    parameters: {{inductance: '{L_val_str}'}}
  - type: Capacitor
    id: C1
    ports: {{ {PORT_1}: mid, {PORT_2}: P2 }} # Use integer ports
    parameters: {{capacitance: '{C_val_str}'}}
ports:
  - {{id: P1, reference_impedance: '50 ohm'}}
  - {{id: P2, reference_impedance: '50 ohm'}}
"""
    built_circuit, freq_array = parse_and_build(parser, builder, yaml_netlist)
    freqs_out, y_matrices = run_sweep(built_circuit, freq_array)
    # Analytical expectations remain the same
    assert len(freqs_out) == 5
    assert y_matrices.shape == (5, 2, 2)
    res_freq_idx = np.argmin(np.abs(freqs_out - 1e9))
    assert np.isclose(freqs_out[res_freq_idx], 1.0e9), "Sweep should include resonance"

    L_val = ureg.Quantity(L_val_str).to(ureg.henry).magnitude
    C_val = ureg.Quantity(C_val_str).to(ureg.farad).magnitude

    for i, freq in enumerate(freqs_out):
        assert freq > 0
        omega = 2 * np.pi * freq
        ZL = 1j * omega * L_val
        ZC = 1 / (1j * omega * C_val)
        Z_series = ZL + ZC
        magnitude_threshold = 1e12
        if np.isclose(freq, 1e9):
            assert np.all(np.isnan(y_matrices[i])) | np.all((np.abs(y_matrices[i])) > magnitude_threshold)
        else:
            assert not np.any(np.isnan(y_matrices[i])), f"Did not expect NaN off resonance freq={freq:.4e} Hz"
            Y_series = 1.0 / Z_series
            Y_expected = Y_series * np.array([[1, -1], [-1, 1]], dtype=complex)
            assert_matrix_close(y_matrices[i], Y_expected, rtol=1e-5, atol=1e-6, msg=f"Series LC @ {freq:.2e} Hz")


# --- Error Handling Tests ---

def test_sweep_invalid_frequency_input(parser, builder):
    # Test contents remain the same, checking input validation of run_sweep
    yaml_netlist = f"""
sweep: {{ type: list, points: ['1 GHz'] }}
components:
  - type: Resistor
    id: R1
    ports: {{ {PORT_1}: P1, {PORT_2}: gnd }} # Use integer ports
    parameters: {{resistance: '50 ohm'}}
ports:
  - {{id: P1, reference_impedance: '50 ohm'}}
ground_net: 'gnd'
"""
    built_circuit, _ = parse_and_build(parser, builder, yaml_netlist)
    with pytest.raises(MnaInputError, match="Frequency sweep array cannot be empty"):
        run_sweep(built_circuit, np.array([]))
    with pytest.raises(MnaInputError, match="must be provided as a 1D NumPy array"):
        run_sweep(built_circuit, np.array([[1e6, 2e6]]))
    with pytest.raises(MnaInputError, match="All frequencies in the sweep must be > 0 Hz"):
        run_sweep(built_circuit, np.array([0.0, 1e9]))
    with pytest.raises(MnaInputError, match="All frequencies in the sweep must be > 0 Hz"):
        run_sweep(built_circuit, np.array([0.0]))
    with pytest.raises(MnaInputError, match="All frequencies in the sweep must be > 0 Hz"):
        run_sweep(built_circuit, np.array([-1e9, 1e9]))

def test_circuit_not_built_error(parser):
    # Test contents remain the same
    yaml_netlist = f"""
sweep: {{ type: list, points: ['1 MHz'] }}
components:
  - type: Resistor
    id: R1
    ports: {{ {PORT_1}: P1, {PORT_2}: gnd }}
    parameters: {{resistance: '1k'}}
ports: [{{id: P1, reference_impedance: '50'}}]
"""
    circuit, freq_array = parser.parse(yaml_netlist)
    assert not hasattr(circuit, 'sim_components')
    with pytest.raises(MnaInputError, match="Circuit object must be processed by CircuitBuilder first"):
        run_sweep(circuit, freq_array)

def test_run_simulation_helper_f_zero_error(parser, builder):
    # Test contents remain the same
    yaml_netlist = f"""
sweep: {{ type: list, points: ['1 GHz'] }}
components:
  - type: Resistor
    id: R1
    ports: {{ {PORT_1}: P1, {PORT_2}: gnd }}
    parameters: {{resistance: '50 ohm'}}
ports:
  - {{id: P1, reference_impedance: '50 ohm'}}
ground_net: 'gnd'
"""
    built_circuit, _ = parse_and_build(parser, builder, yaml_netlist)
    with pytest.raises(MnaInputError, match="Single frequency simulation requires freq_hz > 0"):
        run_simulation(built_circuit, 0.0)
    with pytest.raises(MnaInputError, match="Single frequency simulation requires freq_hz > 0"):
        run_simulation(built_circuit, -1e6)

def test_pi_network_intrinsic_y(parser, builder, freq_1ghz):
    # Ensure correct port IDs are used
    yaml_netlist = f"""
sweep: {{ type: list, points: ['1 GHz'] }}
components:
  - type: Resistor
    id: R1
    ports: {{ {PORT_1}: P1, {PORT_2}: P2 }}
    parameters: {{resistance: '100 ohm'}}
  - type: Resistor
    id: R2
    ports: {{ {PORT_1}: P1, {PORT_2}: gnd }}
    parameters: {{resistance: '50 ohm'}}
  - type: Resistor
    id: R3
    ports: {{ {PORT_1}: P2, {PORT_2}: gnd }}
    parameters: {{resistance: '50 ohm'}}
ports:
  - {{id: P1, reference_impedance: '50 ohm'}}
  - {{id: P2, reference_impedance: '50 ohm'}}
ground_net: 'gnd'
"""
    built_circuit, _ = parse_and_build(parser, builder, yaml_netlist)
    y_matrix = run_simulation(built_circuit, freq_1ghz)
    # Analytical expectation remains the same
    Y1 = 1.0 / 100.0; Y2 = 1.0 / 50.0; Y3 = 1.0 / 50.0
    Y_expected = np.array([[Y1 + Y2, -Y1], [-Y1, Y1 + Y3]], dtype=complex)
    assert_matrix_close(y_matrix, Y_expected, msg="Pi network intrinsic Y")


def test_t_network_intrinsic_y(parser, builder, freq_1ghz):
    # Ensure correct port IDs are used
    yaml_netlist = f"""
sweep: {{ type: list, points: ['1 GHz'] }}
components:
  - type: Resistor
    id: R1
    ports: {{ {PORT_1}: P1, {PORT_2}: mid }}
    parameters: {{resistance: '50 ohm'}}
  - type: Resistor
    id: R2
    ports: {{ {PORT_1}: mid, {PORT_2}: P2 }}
    parameters: {{resistance: '50 ohm'}}
  - type: Resistor
    id: R3
    ports: {{ {PORT_1}: mid, {PORT_2}: gnd }}
    parameters: {{resistance: '25 ohm'}}
ports:
  - {{id: P1, reference_impedance: '50 ohm'}}
  - {{id: P2, reference_impedance: '50 ohm'}}
ground_net: 'gnd'
"""
    built_circuit, _ = parse_and_build(parser, builder, yaml_netlist)
    y_matrix = run_simulation(built_circuit, freq_1ghz)
    # Analytical expectation remains the same
    G1 = 1.0 / 50.0; G2 = 1.0 / 50.0; G3 = 1.0 / 25.0
    SumG = G1 + G2 + G3
    Y_expected = (1.0 / SumG) * np.array([
        [G1*(G2+G3), -G1*G2],
        [-G1*G2, G2*(G1+G3)]
    ], dtype=complex)
    assert_matrix_close(y_matrix, Y_expected, rtol=1e-6, msg="T network intrinsic Y")

# --- Tests for New Semantic Validation ---
# These tests ensure the parser/builder catches topological errors

def test_semantic_validation_floating_internal_net(parser, builder, caplog):
    """ Test warning for internal net connected only once. """
    yaml_netlist = f"""
sweep: {{ type: list, points: ['1 GHz'] }}
components:
  - type: Resistor
    id: R1
    ports: {{ {PORT_1}: P1, {PORT_2}: floating_net }} # Only R1 connects to floating_net
    parameters: {{resistance: '50 ohm'}}
ports:
  - {{id: P1, reference_impedance: '50 ohm'}}
ground_net: 'gnd' # Need ground defined even if not used by R1
"""
    # Parsing should WARN but SUCCEED
    with caplog.at_level(logging.WARNING):
         built_circuit, freq_array = parse_and_build(parser, builder, yaml_netlist)

    assert "Internal net 'floating_net' is only connected to one component port" in caplog.text
    assert "component 'R1' port '1'" in caplog.text # Adjust port ID if needed
    assert built_circuit is not None # Build should still complete

    # Simulation might fail or produce weird results depending on how MNA handles it
    # Let's just check build completes for now. Add simulation check if needed.
    # with pytest.raises(SingularMatrixError): # Or maybe it solves okay?
    #     run_sweep(built_circuit, freq_array)


def test_semantic_validation_unconnected_external_port(parser, builder):
    """ Test error for external port defined but not connected to any component. """
    yaml_netlist = f"""
sweep: {{ type: list, points: ['1 GHz'] }}
components:
  - type: Resistor # Dummy component to make netlist non-empty
    id: R_dummy
    ports: {{ {PORT_1}: n1, {PORT_2}: gnd }}
    parameters: {{resistance: '1k'}}
ports:
  - {{id: P1, reference_impedance: '50 ohm'}} # P1 is defined...
  - {{id: P2, reference_impedance: '50 ohm'}} # ...but P2 is not connected to R_dummy
ground_net: 'gnd'
"""
    # Parsing should FAIL during semantic validation
    with pytest.raises(ParsingError) as excinfo:
        parse_and_build(parser, builder, yaml_netlist)
    assert "External port 'P2' is defined but the net name was never used by any component" in str(excinfo.value)

def test_semantic_validation_duplicate_component_id(parser, builder):
    """ Test error for duplicate component IDs. """
    yaml_netlist = f"""
sweep: {{ type: list, points: ['1 GHz'] }}
components:
  - type: Resistor
    id: R1
    ports: {{ {PORT_1}: P1, {PORT_2}: gnd }}
    parameters: {{resistance: '50 ohm'}}
  - type: Resistor
    id: R1 # Duplicate ID
    ports: {{ {PORT_1}: P1, {PORT_2}: gnd }}
    parameters: {{resistance: '100 ohm'}}
ports:
  - {{id: P1, reference_impedance: '50 ohm'}}
ground_net: 'gnd'
"""
    with pytest.raises(ParsingError) as excinfo:
        parse_and_build(parser, builder, yaml_netlist)
    assert "Duplicate component ID 'R1' found" in str(excinfo.value)

def test_builder_invalid_port_id_usage(parser, builder):
    """ Test build fails if component instance uses undeclared port IDs (using full parse/build). """
    yaml_netlist = f"""
sweep: {{ type: list, points: ['1 GHz'] }}
components:
  - type: Resistor # Resistor expects ports 0, 1
    id: R_bad_port
    ports: {{ p1: P1, p2: gnd }} # Using string IDs 'p1','p2'
    parameters: {{resistance: '50 ohm'}}
ports:
  - {{id: P1, reference_impedance: '50 ohm'}}
ground_net: 'gnd'
"""
    with pytest.raises(CircuitBuildError) as excinfo:
        parse_and_build(parser, builder, yaml_netlist)
    assert "uses undeclared ports" in str(excinfo.value)
    assert "'p1'" in str(excinfo.value) or "'p2'" in str(excinfo.value) # Check for the bad ports
    assert f"[{PORT_1}, {PORT_2}]" in str(excinfo.value) # Check for the declared ports