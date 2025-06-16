# tests/test_dc_analyzer.py
import pytest
import numpy as np
from src.rfsim_core import (
    Circuit, CircuitBuilder, DCAnalyzer, ComponentError, DCAnalysisError,
    Resistor, Capacitor, Inductor,
    ureg, Quantity, COMPONENT_REGISTRY, ADMITTANCE_DIMENSIONALITY
)
from src.rfsim_core.components.base import DCBehaviorType
from tests.conftest import create_and_build_circuit # Assuming helper is in conftest.py

# Ensure components are registered for tests if not done automatically
if "Resistor" not in COMPONENT_REGISTRY: Resistor("dummyR", "Resistor", None, []) # Trigger registration
if "Capacitor" not in COMPONENT_REGISTRY: Capacitor("dummyC", "Capacitor", None, [])
if "Inductor" not in COMPONENT_REGISTRY: Inductor("dummyL", "Inductor", None, [])


class TestDCAnalyzer:

    def test_single_component_dc_behaviors(self, circuit_builder_instance):
        # Test Resistor DC behaviors
        r_params = [
            ("R_finite", "10 ohm", DCBehaviorType.ADMITTANCE, Quantity(0.1, 'S')),
            ("R_short", "0 ohm", DCBehaviorType.SHORT_CIRCUIT, None),
            ("R_open_inf", "inf ohm", DCBehaviorType.OPEN_CIRCUIT, None),
            ("R_open_nan", "nan ohm", DCBehaviorType.OPEN_CIRCUIT, None),
        ]
        for comp_id, r_val_str, expected_type, expected_adm in r_params:
            circuit = create_and_build_circuit(circuit_builder_instance,
                                               [(comp_id, "Resistor", {"resistance": r_val_str}, {"0": "n1", "1": "gnd"})],
                                               circuit_name=f"Test_{comp_id}")
            dc_analyzer = DCAnalyzer(circuit)
            # Accessing internal method for direct check of behavior collection (simplified test)
            behaviors = dc_analyzer._resolve_and_collect_dc_behaviors()
            assert behaviors[comp_id][0] == expected_type
            if expected_adm is not None:
                # OLD: assert behaviors[comp_id][1].check('[admittance]')
                # NEW: Check dimensionality against the canonical ADMITTANCE_DIMENSIONALITY
                assert behaviors[comp_id][1].dimensionality == ADMITTANCE_DIMENSIONALITY
                np.testing.assert_allclose(behaviors[comp_id][1].magnitude, expected_adm.magnitude)
                assert behaviors[comp_id][1].units == expected_adm.units
            else:
                assert behaviors[comp_id][1] is None

        # Test Inductor DC behaviors
        l_params = [
            ("L_finite", "1 nH", DCBehaviorType.SHORT_CIRCUIT, None),
            ("L_short", "0 H", DCBehaviorType.SHORT_CIRCUIT, None),
            ("L_open_inf", "inf H", DCBehaviorType.OPEN_CIRCUIT, None),
            ("L_open_nan", "nan H", DCBehaviorType.OPEN_CIRCUIT, None),
        ]
        for comp_id, l_val_str, expected_type, _ in l_params: # expected_adm is always None
            circuit = create_and_build_circuit(circuit_builder_instance,
                                               [(comp_id, "Inductor", {"inductance": l_val_str}, {"0": "n1", "1": "gnd"})],
                                               circuit_name=f"Test_{comp_id}")
            dc_analyzer = DCAnalyzer(circuit)
            behaviors = dc_analyzer._resolve_and_collect_dc_behaviors()
            assert behaviors[comp_id][0] == expected_type
            assert behaviors[comp_id][1] is None

        # Test Capacitor DC behaviors
        c_params = [
            ("C_open_finite", "1 pF", DCBehaviorType.OPEN_CIRCUIT, None),
            ("C_open_zero", "0 F", DCBehaviorType.OPEN_CIRCUIT, None),
            ("C_short_inf", "inf F", DCBehaviorType.SHORT_CIRCUIT, None),
            ("C_open_nan", "nan F", DCBehaviorType.OPEN_CIRCUIT, None),
        ]
        for comp_id, c_val_str, expected_type, _ in c_params: # expected_adm is always None
            circuit = create_and_build_circuit(circuit_builder_instance,
                                               [(comp_id, "Capacitor", {"capacitance": c_val_str}, {"0": "n1", "1": "gnd"})],
                                               circuit_name=f"Test_{comp_id}")
            dc_analyzer = DCAnalyzer(circuit)
            behaviors = dc_analyzer._resolve_and_collect_dc_behaviors()
            assert behaviors[comp_id][0] == expected_type
            assert behaviors[comp_id][1] is None

    def test_dc_parameter_error_handling(self, circuit_builder_instance):
        error_cases = [
            ("R_neg", "Resistor", {"resistance": "-10 ohm"}),
            ("R_complex", "Resistor", {"resistance": "10+1j ohm"}),
            ("L_neg", "Inductor", {"inductance": "-1 nH"}),
            ("L_complex", "Inductor", {"inductance": "1+1j nH"}),
            ("C_neg", "Capacitor", {"capacitance": "-1 pF"}),
            ("C_complex", "Capacitor", {"capacitance": "1+1j pF"}),
        ]
        for comp_id, comp_type, params in error_cases:
            circuit = create_and_build_circuit(circuit_builder_instance,
                                               [(comp_id, comp_type, params, {"0": "n1", "1": "gnd"})])
            dc_analyzer = DCAnalyzer(circuit)
            with pytest.raises(ComponentError): # Expect ComponentError from get_dc_behavior
                dc_analyzer.analyze()


    def test_supernode_identification(self, circuit_builder_instance):
        # Case 1: Simple short
        circuit1 = create_and_build_circuit(circuit_builder_instance,
                                           [("R0", "Resistor", {"resistance": "0 ohm"}, {"0": "p1", "1": "n1"})],
                                           ground_name="gnd") # gnd is isolated initially
        # Manually add gnd to R0's connections for this test to link n1 to gnd via another short
        circuit1.components["R0"].ports["1"].net = circuit1.get_or_create_net("gnd") # This is a bit hacky post-build
                                                                                 # Better: add another component
        circuit1_updated = create_and_build_circuit(circuit_builder_instance,
                                           [("R0", "Resistor", {"resistance": "0 ohm"}, {"0": "p1", "1": "n1"}),
                                            ("L0", "Inductor", {"inductance": "0 H"}, {"0": "n1", "1": "gnd"})],
                                           external_ports_def={"p1":"50 ohm"},
                                           ground_name="gnd")
        dc_analyzer1 = DCAnalyzer(circuit1_updated)
        results1 = dc_analyzer1.analyze()
        smap1 = results1['dc_supernode_mapping']
        gnd_rep1 = smap1.get("gnd", "gnd") # Should map to itself or the designated ground rep
        assert smap1.get("p1") == gnd_rep1
        assert smap1.get("n1") == gnd_rep1


        # Case 2: Multi-component short, lexicographical representative
        circuit2_comps = [
            ("R0a", "Resistor", {"resistance": "0 ohm"}, {"0": "portX", "1": "N1"}),
            ("L0b", "Inductor", {"inductance": "0 H"}, {"0": "N1", "1": "N2"}),
            ("Cinf", "Capacitor", {"capacitance": "inf F"}, {"0": "N2", "1": "portY"})
        ]
        circuit2 = create_and_build_circuit(circuit_builder_instance, circuit2_comps, ground_name="gnd_iso")
        dc_analyzer2 = DCAnalyzer(circuit2)
        results2 = dc_analyzer2.analyze()
        smap2 = results2['dc_supernode_mapping']
        # Expect N1, N2, portX, portY to map to the same representative, e.g., "N1"
        rep2 = smap2.get("N1")
        assert rep2 is not None
        assert smap2.get("N2") == rep2
        assert smap2.get("portX") == rep2
        assert smap2.get("portY") == rep2
        assert smap2.get("gnd_iso") == "gnd_iso" # Isolated ground

    def test_ydc_single_port_resistor(self, circuit_builder_instance):
        circuit = create_and_build_circuit(circuit_builder_instance,
                                           [("R1", "Resistor", {"resistance": "10 ohm"}, {"0": "p1", "1": "gnd"})],
                                           external_ports_def={"p1": "50 ohm"})
        dc_analyzer = DCAnalyzer(circuit)
        results = dc_analyzer.analyze()

        assert results['dc_port_names_ordered'] == ['p1']
        assert results['dc_port_mapping'] == {'p1': 0}
        Ydc = results['Y_ports_dc']
        assert isinstance(Ydc, Quantity)
        assert Ydc.dimensionality == ADMITTANCE_DIMENSIONALITY
        np.testing.assert_allclose(Ydc.to_base_units().magnitude, np.array([[0.1]]))

    def test_ydc_pi_network(self, circuit_builder_instance):
        components = [
            ("R1", "Resistor", {"resistance": "10 ohm"}, {"0": "p1", "1": "gnd"}), # G1 = 0.1 S
            ("R2", "Resistor", {"resistance": "20 ohm"}, {"0": "p2", "1": "gnd"}), # G2 = 0.05 S
            ("R3", "Resistor", {"resistance": "5 ohm"}, {"0": "p1", "1": "p2"})    # G3 = 0.2 S
        ]
        circuit = create_and_build_circuit(circuit_builder_instance, components,
                                           external_ports_def={"p1": "50 ohm", "p2": "50 ohm"})
        dc_analyzer = DCAnalyzer(circuit)
        results = dc_analyzer.analyze()

        # Expected Y: [[G1+G3, -G3], [-G3, G2+G3]]
        expected_Y_mag = np.array([
            [0.1 + 0.2, -0.2],
            [-0.2, 0.05 + 0.2]
        ])
        Ydc = results['Y_ports_dc']
        assert isinstance(Ydc, Quantity)
        np.testing.assert_allclose(Ydc.to(ureg.siemens).magnitude, expected_Y_mag, atol=1e-9)
        # Port order can vary based on sorting of port names
        if results['dc_port_names_ordered'] == ['p1', 'p2']:
            assert results['dc_port_mapping'] == {'p1': 0, 'p2': 1}
        elif results['dc_port_names_ordered'] == ['p2', 'p1']:
            assert results['dc_port_mapping'] == {'p1': 1, 'p2': 0}
            # If order is swapped, expected_Y_mag also needs swapping of rows/cols for comparison
            np.testing.assert_allclose(Ydc.to(ureg.siemens).magnitude, expected_Y_mag[[1,0]][:,[1,0]], atol=1e-9)
        else:
            pytest.fail("Unexpected dc_port_names_ordered")


    def test_dc_port_mapping_port_to_ground(self, circuit_builder_instance):
        circuit = create_and_build_circuit(circuit_builder_instance,
                                           [("R0", "Resistor", {"resistance": "0 ohm"}, {"0": "p1", "1": "gnd"})],
                                           external_ports_def={"p1": "50 ohm"})
        dc_analyzer = DCAnalyzer(circuit)
        results = dc_analyzer.analyze()
        assert results['dc_port_names_ordered'] == [] # p1 is shorted to ground, so no non-ground DC port
        assert results['dc_port_mapping'] == {'p1': None}
        assert results['Y_ports_dc'] is None or results['Y_ports_dc'].magnitude.shape == (0,0)

    def test_empty_circuit(self, circuit_builder_instance):
        circuit = create_and_build_circuit(circuit_builder_instance, [], circuit_name="Empty")
        dc_analyzer = DCAnalyzer(circuit)
        results = dc_analyzer.analyze()
        assert results['Y_ports_dc'] is None
        assert results['dc_port_names_ordered'] == []
        assert results['dc_supernode_mapping'] == {"gnd": "gnd"} # Assuming default ground name