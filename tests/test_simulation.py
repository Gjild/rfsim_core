# tests/test_simulation.py
import pytest
import numpy as np
from rfsim_core import (
    NetlistParser,
    CircuitBuilder,
    # run_simulation, # Keep if you want dedicated single-point tests
    run_sweep,       # Main focus for Phase 4
    SimulationError,
    MnaInputError,
    SingularMatrixError,
    ureg, Quantity,
    ParsingError, SchemaValidationError # Added for testing invalid YAMLs
)

# Helper for comparing complex matrices
def assert_matrix_close(m1, m2, rtol=1e-5, atol=1e-8):
    """Asserts element-wise closeness for complex numpy arrays."""
    np.testing.assert_allclose(m1, m2, rtol=rtol, atol=atol)

# --- Fixtures ---
@pytest.fixture
def parser():
    return NetlistParser()

@pytest.fixture
def builder():
    return CircuitBuilder()

# --- Single Frequency Tests (using run_sweep with one frequency) ---
# These essentially replace the old run_simulation tests but use the new sweep mechanism

@pytest.fixture
def freq_1ghz_list():
    """Provides a simple frequency list for single-point tests via run_sweep."""
    return np.array([1e9]) # 1 GHz

def test_single_freq_resistor_series(parser, builder, freq_1ghz_list):
    """ Test Z = R, Y = 1/R (single point via run_sweep) """
    yaml_netlist = """
sweep: { type: list, points: ['1 GHz'] } # Define sweep for parsing
components:
  - type: Resistor
    id: R1
    ports: {p1: P1, p2: P2}
    parameters: {resistance: '100 ohm'}
ports:
  - {id: P1, reference_impedance: '50 ohm'}
  - {id: P2, reference_impedance: '50 ohm'}
ground_net: 'gnd'
"""
    circuit = parser.parse(yaml_netlist)
    built_circuit = builder.build_circuit(circuit)
    # Use run_sweep even for single frequency
    freqs_out, y_matrices = run_sweep(built_circuit, freq_1ghz_list)

    assert freqs_out.shape == (1,)
    assert y_matrices.shape == (1, 2, 2)
    y_matrix = y_matrices[0] # Extract the single result

    # Analytical calculation (as before)
    Z_expected = np.array([[37.5, 12.5], [12.5, 37.5]], dtype=complex)
    Y_expected = np.linalg.inv(Z_expected)

    assert_matrix_close(y_matrix, Y_expected)

def test_single_freq_resistor_shunt(parser, builder, freq_1ghz_list):
    """ Test Y = 1/R_shunt (single point via run_sweep)"""
    yaml_netlist = """
sweep: { type: list, points: ['1 GHz'] }
components:
  - type: Resistor
    id: R1
    ports: {p1: P1, p2: gnd}
    parameters: {resistance: '25 ohm'}
ports:
  - {id: P1, reference_impedance: '50 ohm'}
ground_net: 'gnd'
"""
    circuit = parser.parse(yaml_netlist)
    built_circuit = builder.build_circuit(circuit)
    freqs_out, y_matrices = run_sweep(built_circuit, freq_1ghz_list)

    assert freqs_out.shape == (1,)
    assert y_matrices.shape == (1, 1, 1)
    y_matrix = y_matrices[0]

    # Analytical calculation (as before)
    Z_expected = np.array([[50.0/3.0]], dtype=complex)
    Y_expected = np.linalg.inv(Z_expected) # [[0.06]]

    assert_matrix_close(y_matrix, Y_expected)


def test_single_freq_voltage_divider(parser, builder, freq_1ghz_list):
    """ R1 series, R2 shunt (single point via run_sweep) """
    yaml_netlist = """
sweep: { type: list, points: ['1 GHz'] }
components:
  - type: Resistor
    id: R1
    ports: {p1: P1, p2: mid}
    parameters: {resistance: '50 ohm'}
  - type: Resistor
    id: R2
    ports: {p1: mid, p2: gnd}
    parameters: {resistance: '50 ohm'}
ports:
  - {id: P1, reference_impedance: '50 ohm'}
ground_net: 'gnd'
"""
    circuit = parser.parse(yaml_netlist)
    built_circuit = builder.build_circuit(circuit)
    freqs_out, y_matrices = run_sweep(built_circuit, freq_1ghz_list)

    assert freqs_out.shape == (1,)
    assert y_matrices.shape == (1, 1, 1)
    y_matrix = y_matrices[0]

    # Analytical calculation (as before)
    Z_expected = np.array([[100.0/3.0]], dtype=complex)
    Y_expected = np.linalg.inv(Z_expected) # [[0.03]]

    assert_matrix_close(y_matrix, Y_expected)


def test_single_freq_lc_circuit(parser, builder, freq_1ghz_list):
    """ Simple series LC (single point via run_sweep) """
    yaml_netlist = """
sweep: { type: list, points: ['1 GHz'] }
components:
  - type: Inductor
    id: L1
    ports: {p1: P1, p2: mid}
    parameters: {inductance: '7.9577 nH'}
  - type: Capacitor
    id: C1
    ports: {p1: mid, p2: P2}
    parameters: {capacitance: '3.183 pF'}
ports:
  - {id: P1, reference_impedance: '50 ohm'}
  - {id: P2, reference_impedance: '50 ohm'}
ground_net: 'gnd'
"""
    circuit = parser.parse(yaml_netlist)
    built_circuit = builder.build_circuit(circuit)
    freqs_out, y_matrices = run_sweep(built_circuit, freq_1ghz_list)

    assert freqs_out.shape == (1,)
    assert y_matrices.shape == (1, 2, 2)
    y_matrix = y_matrices[0]

    # Analytical calculation at 1 GHz (as before)
    freq = 1e9
    ZL = 1j * 2*np.pi*freq * 7.9577e-9
    ZC = 1 / (1j * 2*np.pi*freq * 3.183e-12)
    Z_series = ZL + ZC
    Y_series = 1.0 / Z_series
    Y0 = 1.0 / 50.0
    Y_expected_direct = np.array([[Y0 + Y_series, -Y_series], [-Y_series, Y0 + Y_series]])

    assert_matrix_close(y_matrix, Y_expected_direct, rtol=1e-3, atol=1e-4)


# --- Sweep Specific Tests ---

def test_sweep_simple_resistor_shunt(parser, builder):
    """ Test shunt R across multiple frequencies """
    yaml_netlist = """
sweep: { type: list, points: ['1 MHz', '500 MHz', '1 GHz'] }
components:
  - type: Resistor
    id: R1
    ports: {p1: P1, p2: gnd}
    parameters: {resistance: '25 ohm'}
ports:
  - {id: P1, reference_impedance: '50 ohm'}
ground_net: 'gnd'
"""
    circuit = parser.parse(yaml_netlist)
    built_circuit = builder.build_circuit(circuit)
    # Use frequencies parsed from netlist
    freqs_hz, y_matrices = run_sweep(built_circuit, circuit.frequency_sweep_hz)

    assert len(freqs_hz) == 3
    assert y_matrices.shape == (3, 1, 1)
    assert np.allclose(freqs_hz, [1e6, 5e8, 1e9])

    # Analytical calculation (same for all freqs)
    Z_expected = np.array([[50.0/3.0]], dtype=complex)
    Y_expected = np.linalg.inv(Z_expected) # [[0.06]]

    # Check results for all frequencies
    for i in range(len(freqs_hz)):
        assert_matrix_close(y_matrices[i,:,:], Y_expected)


def test_sweep_lc_circuit_detailed(parser, builder):
    """ Test series LC across frequency, including resonance, checking multiple points """
    yaml_netlist = """
sweep: { type: linear, start: '0.8 GHz', stop: '1.2 GHz', num_points: 5 }
components:
  - type: Inductor
    id: L1
    ports: {p1: P1, p2: mid}
    parameters: {inductance: '7.9577 nH'} # Resonant freq = 1 GHz
  - type: Capacitor
    id: C1
    ports: {p1: mid, p2: P2}
    parameters: {capacitance: '3.183 pF'}
ports:
  - {id: P1, reference_impedance: '50 ohm'}
  - {id: P2, reference_impedance: '50 ohm'}
ground_net: 'gnd'
"""
    circuit = parser.parse(yaml_netlist)
    built_circuit = builder.build_circuit(circuit)
    freqs_hz, y_matrices = run_sweep(built_circuit, circuit.frequency_sweep_hz)

    assert len(freqs_hz) == 5
    assert y_matrices.shape == (5, 2, 2)
    assert np.isclose(freqs_hz[0], 0.8e9)
    assert np.isclose(freqs_hz[2], 1.0e9) # Resonance point
    assert np.isclose(freqs_hz[4], 1.2e9)

    Y0 = 1.0 / 50.0
    L_val = 7.9577e-9
    C_val = 3.183e-12

    # Check results for all frequencies
    for i, freq in enumerate(freqs_hz):
        if freq == 0: # Should not happen with this sweep range
            assert np.all(np.isnan(y_matrices[i]))
            continue

        ZL = 1j * 2*np.pi*freq * L_val
        ZC = 1 / (1j * 2*np.pi*freq * C_val)
        Z_series = ZL + ZC

        if abs(Z_series) < 1e-9: # Near resonance, use Z matrix check
            Z_expected = np.array([[25.0, 25.0], [25.0, 25.0]], dtype=complex)
            # Invert simulated Y to compare with expected Z
            try:
                Z_sim = np.linalg.inv(y_matrices[i])
                assert_matrix_close(Z_sim, Z_expected, rtol=1e-3)
            except np.linalg.LinAlgError:
                pytest.fail(f"Y matrix inversion failed at resonance (freq={freq} Hz)")
        else: # Away from resonance, compare Y matrices directly
            Y_series = 1.0 / Z_series
            Y_expected = np.array([[Y0 + Y_series, -Y_series], [-Y_series, Y0 + Y_series]])
            assert_matrix_close(y_matrices[i], Y_expected, rtol=1e-4)

def test_sweep_with_dc_point_handling(parser, builder):
    """ Test sweep including F=0, expecting NaN """
    yaml_netlist = """
sweep: { type: list, points: ['0 Hz', '1 MHz', '10 MHz'] }
components:
  - type: Resistor
    id: R1
    ports: {p1: P1, p2: gnd}
    parameters: {resistance: '25 ohm'}
ports:
  - {id: P1, reference_impedance: '50 ohm'}
ground_net: 'gnd'
"""
    circuit = parser.parse(yaml_netlist)
    built_circuit = builder.build_circuit(circuit)
    freqs_hz, y_matrices = run_sweep(built_circuit, circuit.frequency_sweep_hz)

    assert len(freqs_hz) == 3 # 0 Hz included
    assert y_matrices.shape == (3, 1, 1)
    assert freqs_hz[0] == 0.0

    # Check DC point - should be NaN as we skip MNA solve in run_sweep
    assert np.all(np.isnan(y_matrices[0,:,:]))

    # Check AC points (should be same result)
    Z_expected = np.array([[50.0/3.0]], dtype=complex)
    Y_expected = np.linalg.inv(Z_expected)
    assert_matrix_close(y_matrices[1,:,:], Y_expected)
    assert_matrix_close(y_matrices[2,:,:], Y_expected)

# --- Error Handling Tests ---

def test_sweep_invalid_frequency_input(parser, builder):
    """ Test run_sweep with invalid frequency inputs """
    # Use a minimal valid netlist first
    yaml_netlist = """
sweep: { type: list, points: ['1 GHz', '2 GHz'] }
components:
  - type: Resistor
    id: R1
    ports: {p1: P1, p2: mid}
    parameters: {resistance: '50 ohm'}
ports:
  - {id: P1, reference_impedance: '50 ohm'}
  - {id: P2, reference_impedance: '50 ohm'}
ground_net: 'gnd'
"""
    circuit = parser.parse(yaml_netlist)
    built_circuit = builder.build_circuit(circuit)

    # Test with empty array
    with pytest.raises(MnaInputError, match="Frequency sweep array cannot be empty"):
        run_sweep(built_circuit, np.array([]))

    # Test with 2D array
    with pytest.raises(MnaInputError, match="must be provided as a 1D NumPy array"):
        run_sweep(built_circuit, np.array([[1e6, 2e6]]))

    # Test with only F=0 (currently disallowed by run_sweep)
    # NOTE: This check might move/change when proper DC analysis is added
    with pytest.raises(MnaInputError, match="Cannot run AC sweep simulation with only F=0 Hz points."):
        run_sweep(built_circuit, np.array([0.0]))

def test_sweep_unconnected_port(parser, builder):
    """ Test sweep with a circuit likely to be singular """
    yaml_netlist = """
sweep: { type: list, points: ['1 GHz', '2 GHz'] }
components:
  - type: Resistor
    id: R1
    ports: {p1: P1, p2: mid}
    parameters: {resistance: '50 ohm'}
ports:
  - {id: P1, reference_impedance: '50 ohm'}
  - {id: P2, reference_impedance: '50 ohm'}
ground_net: 'gnd'
"""
    circuit = parser.parse(yaml_netlist)
    built_circuit = builder.build_circuit(circuit)

    # We expect the solver to fail (SingularMatrixError) inside run_sweep.
    # run_sweep currently catches this and returns NaN. Let's verify NaNs.
    freqs_hz, y_matrices = run_sweep(built_circuit, circuit.frequency_sweep_hz)

    assert len(freqs_hz) == 2
    assert y_matrices.shape == (2, 2, 2) # 2 ports defined
    # Expect NaN for both frequencies due to singularity
    assert np.all(np.isnan(y_matrices))


def test_circuit_not_built_error(parser):
    """ Test calling run_sweep with a circuit not processed by builder """
    yaml_netlist = """
sweep: { type: list, points: ['1 MHz'] }
components:
  - type: Resistor
    id: R1
    ports: # Use standard mapping
      p1: P1
      p2: gnd
    parameters: {resistance:'1k'}
ports: [{id: P1, reference_impedance: '50'}]
"""
    # Parse only, don't build
    circuit = parser.parse(yaml_netlist)
    assert not hasattr(circuit, 'sim_components') # Verify it's not built

    with pytest.raises(MnaInputError, match="Circuit object must be processed by CircuitBuilder first"):
        run_sweep(circuit, circuit.frequency_sweep_hz)


# Add more tests: Pi network, T network sweeps.