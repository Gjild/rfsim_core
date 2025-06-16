# tests/test_topology_analyzer.py
import pytest
import numpy as np
from src.rfsim_core import (
    CircuitBuilder, TopologyAnalyzer, TopologyAnalysisError, ComponentError,
    Resistor, Capacitor, Inductor
)
from tests.conftest import create_and_build_circuit


class TestTopologyAnalyzer:

    def test_is_structurally_open_behavior(self, circuit_builder_instance):
        # Test Resistor structural open
        circuit_r_inf = create_and_build_circuit(circuit_builder_instance,
            [("Rinf", "Resistor", {"resistance": "inf ohm"}, {"0": "n1", "1": "n2"})])
        ta_r_inf = TopologyAnalyzer(circuit_r_inf)
        assert "Rinf" in ta_r_inf._resolve_and_identify_structurally_open_components()

        circuit_r_large = create_and_build_circuit(circuit_builder_instance,
            [("Rlarge", "Resistor", {"resistance": "1e12 ohm"}, {"0": "n1", "1": "n2"})])
        ta_r_large = TopologyAnalyzer(circuit_r_large)
        assert "Rlarge" not in ta_r_large._resolve_and_identify_structurally_open_components()

        # Test Capacitor structural open (C=0)
        circuit_c_zero = create_and_build_circuit(circuit_builder_instance,
            [("Czero", "Capacitor", {"capacitance": "0 F"}, {"0": "n1", "1": "n2"})])
        ta_c_zero = TopologyAnalyzer(circuit_c_zero)
        assert "Czero" in ta_c_zero._resolve_and_identify_structurally_open_components()

        circuit_c_small = create_and_build_circuit(circuit_builder_instance,
            [("Csmall", "Capacitor", {"capacitance": "1e-18 F"}, {"0": "n1", "1": "n2"})])
        ta_c_small = TopologyAnalyzer(circuit_c_small)
        assert "Csmall" not in ta_c_small._resolve_and_identify_structurally_open_components()
        
        # Test Inductor structural open (L=inf)
        circuit_l_inf = create_and_build_circuit(circuit_builder_instance,
            [("Linf", "Inductor", {"inductance": "inf H"}, {"0": "n1", "1": "n2"})])
        ta_l_inf = TopologyAnalyzer(circuit_l_inf)
        assert "Linf" in ta_l_inf._resolve_and_identify_structurally_open_components()

        circuit_l_large = create_and_build_circuit(circuit_builder_instance,
            [("Llarge", "Inductor", {"inductance": "1e12 H"}, {"0": "n1", "1": "n2"})])
        ta_l_large = TopologyAnalyzer(circuit_l_large)
        assert "Llarge" not in ta_l_large._resolve_and_identify_structurally_open_components()

        # Test non-constant parameter
        # Note: This requires ParameterManager to correctly identify non-constant
        circuit_r_expr = create_and_build_circuit(circuit_builder_instance,
            [("Rexpr", "Resistor", {"resistance": {"expression": "freq * 10", "dimension":"ohm"}}, {"0": "n1", "1": "n2"})])
        ta_r_expr = TopologyAnalyzer(circuit_r_expr)
        assert "Rexpr" not in ta_r_expr._resolve_and_identify_structurally_open_components()


    def test_active_nets_calculation(self, circuit_builder_instance): # circuit_builder_instance from conftest
        # Fully connected
        circuit1 = create_and_build_circuit(circuit_builder_instance,
            [("R1", "Resistor", {"resistance": "50 ohm"}, {"0": "p1", "1": "N1"}),
             ("R2", "Resistor", {"resistance": "50 ohm"}, {"0": "N1", "1": "gnd"})],
            external_ports_def={"p1": "50 ohm"})
        ta1 = TopologyAnalyzer(circuit1)
        assert ta1.get_active_nets() == {"p1", "N1", "gnd"}

        # Floating section
        circuit2 = create_and_build_circuit(circuit_builder_instance,
            [("R1", "Resistor", {"resistance": "50 ohm"}, {"0": "p1", "1": "gnd"}),
             ("Riso", "Resistor", {"resistance": "100 ohm"}, {"0": "N_iso1", "1": "N_iso2"})], # N_iso1, N_iso2 are floating
            external_ports_def={"p1": "50 ohm"})
        ta2 = TopologyAnalyzer(circuit2)
        assert ta2.get_active_nets() == {"p1", "gnd"}

        # Port structurally open
        circuit3 = create_and_build_circuit(circuit_builder_instance,
            [("Rinf", "Resistor", {"resistance": "inf ohm"}, {"0": "p1", "1": "N1"}),
             ("R2", "Resistor", {"resistance": "50 ohm"}, {"0": "N1", "1": "gnd"})],
            external_ports_def={"p1": "50 ohm"})
        ta3 = TopologyAnalyzer(circuit3)
        # p1 is an external port, thus a source for traversal. It is active.
        # N1 and gnd are connected to the ground source.
        # Rinf being open isolates p1 from N1 in the AC graph.
        # Active nets are union of {p1} (from p1 source) and {N1, gnd} (from gnd source).
        assert ta3.get_active_nets() == {"p1", "N1", "gnd"} # This assertion is correct based on spec

        # Ground structurally open from port
        circuit4 = create_and_build_circuit(circuit_builder_instance,
            [("R1", "Resistor", {"resistance": "50 ohm"}, {"0": "p1", "1": "N1"}),
             ("Rinf", "Resistor", {"resistance": "inf ohm"}, {"0": "N1", "1": "gnd"})], # Rinf connects N1 to gnd
            external_ports_def={"p1": "50 ohm"})
        ta4 = TopologyAnalyzer(circuit4)
        # Rinf is open, so p1-N1 is disconnected from gnd in the AC graph.
        # Active nets are union of {p1, N1} (from p1 source) and {gnd} (from gnd source).
        assert ta4.get_active_nets() == {"p1", "N1", "gnd"} # Corrected assertion

    def test_topology_analyzer_no_ports_no_ground_connection(self, circuit_builder_instance):
        circuit = create_and_build_circuit(circuit_builder_instance,
            [("R1", "Resistor", {"resistance": "50 ohm"}, {"0": "N1", "1": "N2"})],
            ground_name="gnd_unconn") # gnd_unconn is not connected to N1 or N2
        ta = TopologyAnalyzer(circuit)
        # Traversal sources are gnd_unconn (not in graph effectively) + empty external ports.
        # So, active_nets should be empty unless gnd_unconn itself is considered active.
        # Based on plan: "Traversal sources: circuit.ground_net_name (if it exists) + all net names from circuit.external_ports.keys()"
        # if gnd_unconn is added to graph but nothing connects to it -> only gnd_unconn
        # if graph only contains N1,N2 -> empty
        # Current TopologyAnalyzer adds all nets to graph, then traverses.
        # If gnd_unconn is a source and isolated, only it is active.
        # If N1/N2 cannot be reached from gnd_unconn, they are not active.
        assert ta.get_active_nets() == {"gnd_unconn"} # gnd_unconn is a source, but isolated.

    def test_topology_analyzer_only_ground_net(self, circuit_builder_instance):
        circuit = create_and_build_circuit(circuit_builder_instance, [], ground_name="gnd_only")
        ta = TopologyAnalyzer(circuit)
        assert ta.get_active_nets() == {"gnd_only"}