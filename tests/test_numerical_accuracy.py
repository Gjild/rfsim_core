# tests/test_numerical_accuracy.py
import pytest
import numpy as np
from src.rfsim_core import (
    CircuitBuilder,
    run_sweep,
    ureg,
    Quantity
)
from tests.conftest import create_and_build_circuit # Helper from conftest

class TestGoldenReferenceCircuits:

    def test_resistive_t_attenuator(self, circuit_builder_instance):
        """
        Tests a symmetric resistive T-attenuator.
        P1 -- R_series1 -- N1 -- R_series2 -- P2
                           |
                         R_shunt
                           |
                          GND
        R_series1 = 50 ohm, R_shunt = 25 ohm, R_series2 = 50 ohm.
        Expected Y-matrix (Siemens):
        Y11 = (R_shunt + R_series2) / D = (25+50)/5000 = 0.015
        Y22 = (R_series1 + R_shunt) / D = (50+25)/5000 = 0.015
        Y12 = Y21 = -R_shunt / D = -25/5000 = -0.005
        where D = R_series1*R_shunt + R_series1*R_series2 + R_shunt*R_series2
                = 50*25 + 50*50 + 25*50 = 1250 + 2500 + 1250 = 5000
        """
        components = [
            ("Rs1", "Resistor", {"resistance": "50 ohm"}, {"0": "P1", "1": "N1"}),
            ("Rsh", "Resistor", {"resistance": "25 ohm"}, {"0": "N1", "1": "gnd"}),
            ("Rs2", "Resistor", {"resistance": "50 ohm"}, {"0": "N1", "1": "P2"}),
        ]
        ext_ports = {"P1": "50 ohm", "P2": "50 ohm"}
        circuit_name = "T_Attenuator"
        
        # Create and build the circuit using the conftest helper
        sim_circuit = create_and_build_circuit(
            circuit_builder_instance,
            components_def=components,
            external_ports_def=ext_ports,
            circuit_name=circuit_name
        )

        # Define a single frequency for the sweep (DC or AC, result is the same for resistors)
        freq_array_hz = np.array([1e9]) # 1 GHz

        # Run the sweep
        returned_freqs, y_matrices, dc_results = run_sweep(sim_circuit, freq_array_hz)

        assert y_matrices.shape == (1, 2, 2) # 1 frequency, 2 ports, 2 ports

        # Expected Y-matrix values
        y11_expected = 0.015
        y12_expected = -0.005
        y21_expected = -0.005
        y22_expected = 0.015
        
        # Extract the Y-matrix from the simulation results (first frequency point)
        y_sim = y_matrices[0] # This is a 2x2 NumPy array

        # Perform assertions. Port order P1, P2 is fixed by MnaAssembler sorting port names.
        np.testing.assert_allclose(y_sim[0, 0], y11_expected, atol=1e-9, rtol=1e-6,
                                    err_msg="Y11 mismatch for T-attenuator")
        np.testing.assert_allclose(y_sim[0, 1], y12_expected, atol=1e-9, rtol=1e-6,
                                    err_msg="Y12 mismatch for T-attenuator")
        np.testing.assert_allclose(y_sim[1, 0], y21_expected, atol=1e-9, rtol=1e-6,
                                    err_msg="Y21 mismatch for T-attenuator")
        np.testing.assert_allclose(y_sim[1, 1], y22_expected, atol=1e-9, rtol=1e-6,
                                    err_msg="Y22 mismatch for T-attenuator")

        # Also check DC results if freq=0 was included, though for pure resistive it's the same
        if 0.0 in freq_array_hz:
            assert dc_results is not None
            y_dc_from_dc_analyzer = dc_results['Y_ports_dc'].to(ureg.siemens).magnitude
            np.testing.assert_allclose(y_dc_from_dc_analyzer[0, 0], y11_expected, atol=1e-9, rtol=1e-6)
            # ... and so on for other DC elements if needed for this specific test.
            # For this T-attenuator, the AC result at any freq and DC result should match.
            y_at_f0_from_sweep = y_matrices[np.where(returned_freqs == 0.0)[0][0]]
            np.testing.assert_allclose(y_at_f0_from_sweep, y_dc_from_dc_analyzer, atol=1e-9, rtol=1e-6)