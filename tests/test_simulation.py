# tests/test_simulation.py
import pytest
import numpy as np
from rfsim_core import (
    NetlistParser,
    CircuitBuilder,
    run_simulation,
    SimulationError,
    MnaInputError,
    SingularMatrixError,
    ureg, Quantity
)

# Helper for comparing complex matrices
def assert_matrix_close(m1, m2, rtol=1e-5, atol=1e-8):
    np.testing.assert_allclose(m1, m2, rtol=rtol, atol=atol)

# --- Fixtures ---
@pytest.fixture
def parser():
    return NetlistParser()

@pytest.fixture
def builder():
    return CircuitBuilder()

# --- Test Cases ---

def test_simple_resistor_series(parser, builder):
    """ Test Z = R, Y = 1/R """
    yaml_netlist = """
components:
  - type: Resistor
    id: R1
    ports: {p1: P1, p2: P2}
    parameters: {resistance: '100 ohm'}
ports:
  - {id: P1, reference_impedance: '50 ohm'}
  - {id: P2, reference_impedance: '50 ohm'}
ground_net: 'gnd' # Ensure default ground is known if R connects elsewhere
"""
    freq = 1e9 # Frequency shouldn't matter for resistor

    circuit = parser.parse(yaml_netlist)
    built_circuit = builder.build_circuit(circuit)
    y_matrix = run_simulation(built_circuit, freq)

    # Analytical calculation
    # Z = [[Z11, Z12], [Z21, Z22]]
    # Z11 = V1/I1 | I2=0. I1 flows through R1 to P2 (open). Z11 -> infinite ideally?
    # Let's rethink: MNA solves with ports loaded by Z0.
    # The run_simulation calculates the Z matrix *relative to injected currents*
    # Z_matrix[i,j] = V_port_i / I_inj_j
    # For a series 100 ohm resistor between two 50 ohm ports:
    # Exciting P1 (I1=1, I2=0): V1 = 50 || (100 + 50) = 50 || 150 = 37.5 V (relative to gnd)
    #                           Current into R1 = (V1-V2)/100. V2 = V1 * 50 / (100+50) = V1 * 1/3
    #                           I1 = V1/50 + (V1-V2)/100 = V1/50 + (V1 - V1/3)/100 = V1/50 + (2/3 V1)/100 = V1/50 + V1/150 = (3V1 + V1)/150 = 4V1/150
    #                           V1 = I1 * 150 / 4 = 1 * 37.5 = 37.5 V
    #                           V2 = V1 / 3 = 37.5 / 3 = 12.5 V
    # Z11 = V1/I1 = 37.5 / 1 = 37.5
    # Z21 = V2/I1 = 12.5 / 1 = 12.5
    # By symmetry: Z22 = 37.5, Z12 = 12.5
    Z_expected = np.array([[37.5, 12.5], [12.5, 37.5]], dtype=complex)
    Y_expected = np.linalg.inv(Z_expected)

    # Compare Y matrices
    assert_matrix_close(y_matrix, Y_expected)


def test_simple_resistor_shunt(parser, builder):
    """ Test Y = 1/R_shunt """
    yaml_netlist = """
components:
  - type: Resistor
    id: R1
    ports: {p1: P1, p2: gnd}
    parameters: {resistance: '25 ohm'}
ports:
  - {id: P1, reference_impedance: '50 ohm'}
ground_net: 'gnd'
"""
    freq = 1e9 # Frequency shouldn't matter for resistor

    circuit = parser.parse(yaml_netlist)
    built_circuit = builder.build_circuit(circuit)
    y_matrix = run_simulation(built_circuit, freq)

    # Analytical calculation (1-port)
    # Y_circuit = 1/25 = 0.04 S
    # The simulation calculates the Y matrix assuming ports are loaded
    # Z11 = V1/I1. I1 injected at P1.
    # V1 = I1 * (Z0 || R_shunt) = 1 * (50 || 25) = 1 * (50*25 / (50+25)) = 1 * (1250 / 75) = 1 * 50/3
    # Z_expected = [[50/3]]
    Z_expected = np.array([[50.0/3.0]], dtype=complex)
    Y_expected = np.linalg.inv(Z_expected) # [[3/50]] = [[0.06]]

    # Compare Y matrices
    assert y_matrix.shape == (1, 1)
    assert_matrix_close(y_matrix, Y_expected)


def test_voltage_divider(parser, builder):
    """ R1 series, R2 shunt """
    yaml_netlist = """
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
    freq = 1e9

    circuit = parser.parse(yaml_netlist)
    built_circuit = builder.build_circuit(circuit)
    y_matrix = run_simulation(built_circuit, freq)

    # Analytical calculation (1-port)
    # Input impedance looking into P1 = R1 + R2 = 50 + 50 = 100 ohm
    # Z11 = V1/I1. I1 injected at P1. Port load Z0=50ohm.
    # V1 = I1 * (Z0 || (R1+R2)) = 1 * (50 || 100) = 1 * (50*100 / (50+100)) = 1 * (5000 / 150) = 1 * 100/3
    # Z_expected = [[100/3]]
    Z_expected = np.array([[100.0/3.0]], dtype=complex)
    Y_expected = np.linalg.inv(Z_expected) # [[3/100]] = [[0.03]]

    assert y_matrix.shape == (1, 1)
    assert_matrix_close(y_matrix, Y_expected)


def test_lc_circuit(parser, builder):
    """ Simple series LC """
    # Choose f=1GHz. L=7.9577nH -> ZL = j50. C=3.183pF -> ZC = -j50
    yaml_netlist = """
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
    freq = 1e9 # 1 GHz

    circuit = parser.parse(yaml_netlist)
    built_circuit = builder.build_circuit(circuit)
    y_matrix = run_simulation(built_circuit, freq)

    # Analytical calculation at 1 GHz
    # ZL = j * 2*pi*1e9 * 7.9577e-9 = j * 6.283 * 7.9577 ~= j50
    # ZC = 1 / (j * 2*pi*1e9 * 3.183e-12) = 1 / (j * 6.283 * 3.183e-3) = 1 / (j * 0.01999) ~= -j50
    # Total series impedance Z_series = ZL + ZC ~= j50 - j50 = 0 (Resonance)
    ZL = 1j * 2*np.pi*freq * 7.9577e-9
    ZC = 1 / (1j * 2*np.pi*freq * 3.183e-12)
    Z_series = ZL + ZC # Should be close to 0

    # Exciting P1 (I1=1, I2=0):
    # V1 = I1 * (Z0 || (Z_series + Z0)) = 1 * (50 || (Z_series + 50))
    # V2 = V1 * Z0 / (Z_series + Z0)
    Z0 = 50.0 + 0j
    Z11 = (Z0 * (Z_series + Z0)) / (Z0 + Z_series + Z0)
    Z21 = Z11 * Z0 / (Z_series + Z0)
    # By symmetry
    Z22 = Z11
    Z12 = Z21

    Z_expected = np.array([[Z11, Z12], [Z21, Z22]])
    # Since Z_series is near 0, Z11 ~= (50*50)/(50+50) = 25. Z21 ~= Z11*50/50 = Z11 = 25
    # Z_expected ~= [[25, 25], [25, 25]]
    # Y = inv(Z) -> Should be singular or near-singular if Z_series is exactly 0.
    # Check Y matrix directly might be better for resonant case. Y = [[Y11, Y12],[Y21,Y22]]
    # Y_series = 1/Z_series (very large)
    # Y11 = Y_load1 + Y_series = 1/50 + Y_series
    # Y12 = -Y_series
    # Y21 = -Y_series
    # Y22 = Y_load2 + Y_series = 1/50 + Y_series
    # Let's compare Z as inversion might fail numerically near resonance
    # We need the Z matrix computed by run_simulation
    # The run_simulation returns Y, let's invert expected Y.
    Y_series = 1.0 / Z_series
    Y0 = 1.0 / Z0
    Y_expected_direct = np.array([[Y0 + Y_series, -Y_series], [-Y_series, Y0 + Y_series]])

    # Compare Y matrix calculated by simulation with direct calculation
    assert_matrix_close(y_matrix, Y_expected_direct, rtol=1e-3, atol=1e-4) # Looser tolerance near resonance

def test_invalid_frequency(parser, builder):
    yaml_netlist = """
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
    with pytest.raises(MnaInputError, match="requires frequency > 0 Hz"):
        run_simulation(built_circuit, 0.0)
    with pytest.raises(MnaInputError, match="requires frequency > 0 Hz"):
        run_simulation(built_circuit, -1e9)


def test_unconnected_port_simulation(parser, builder):
    # A port must be connected to *something* for MNA to work.
    # The parser ensures ports connect to nets, semantic validation (Phase 6)
    # will ensure nets have >= 2 connections.
    # Let's test a case that *should* be singular if not handled properly.
    # One port connected to a resistor to ground, second port floating.
    yaml_netlist = """
components:
  - type: Resistor
    id: R1
    ports: {p1: P1, p2: gnd}
    parameters: {resistance: '50 ohm'}
ports:
  - {id: P1, reference_impedance: '50 ohm'}
  - {id: P2, reference_impedance: '50 ohm'} # P2 connects to nothing
ground_net: 'gnd'
"""
    # This circuit is problematic. P2 is defined as a port but not connected.
    # The MNA matrix will likely be singular because node P2 is floating.
    # Current MnaAssembler doesn't check for floating nodes yet (Phase 7).
    circuit = parser.parse(yaml_netlist)
    built_circuit = builder.build_circuit(circuit)
    # We expect the solver to potentially fail here due to singularity
    with pytest.raises(SingularMatrixError):
         Y = run_simulation(built_circuit, 1e9)
         print(Y)

# Add more tests: Pi network, T network, capacitor only, inductor only.