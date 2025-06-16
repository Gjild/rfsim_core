# tests/test_mna_assembler_filtering.py
import pytest
import numpy as np
from src.rfsim_core import (
    CircuitBuilder, MnaAssembler, MnaInputError, Resistor,
    ureg, Quantity
)
from tests.conftest import create_and_build_circuit

@pytest.fixture
def base_mna_circuit(circuit_builder_instance):
    components = [
        ("Rg1", "Resistor", {"resistance": "10 ohm"}, {"0": "gnd", "1": "n1"}),
        ("R12", "Resistor", {"resistance": "20 ohm"}, {"0": "n1", "1": "n2"}),
        ("R2g", "Resistor", {"resistance": "30 ohm"}, {"0": "n2", "1": "gnd"})
    ]
    # Define p1 on n1, p2 on n2
    ext_ports = {"n1": "50 ohm", "n2": "50 ohm"} # Port names are the same as net names here
    circuit = create_and_build_circuit(circuit_builder_instance, components, ext_ports, circuit_name="BaseMNA")
    return circuit

class TestMnaAssemblerFiltering:

    def test_mao_subset1_gnd_n1(self, base_mna_circuit):
        active_nets = {"gnd", "n1"}
        assembler = MnaAssembler(base_mna_circuit, active_nets_override=active_nets)

        assert assembler.node_count == 2
        # Ground should be 0 if present and active
        assert assembler.node_map.get("gnd") == 0
        assert assembler.node_map.get("n1") == 1
        assert "n2" not in assembler.node_map # n2 is not active

        assert assembler.port_names == ["n1"] # p1 (on n1) is active
        assert assembler.port_indices == [1]  # n1 is MNA index 1

        # Test assemble behavior (simplified check: number of non-zeros or specific values)
        # Only Rg1 and port n1's Z0 are relevant
        freq = 1e9
        Yn_full = assembler.assemble(freq)
        assert Yn_full.shape == (2, 2)
        # At n1 (idx 1): G_Rg1 + G_Z0_n1
        # G_Rg1 = 0.1, G_Z0_n1 = 1/50 = 0.02
        # Expected Y[1,1] = 0.1 + 0.02 (approx, if port impedance is stamped on diagonal)
        # This depends on how MnaAssembler handles port Z0 when active_nets are used.
        # The original MnaAssembler stamps 1/Z0 from port node to ground.
        # If ground (idx 0) is active, this is Y[1,1] += G_Z0, Y[0,0] += G_Z0, Y[0,1]-=G_Z0, Y[1,0]-=G_Z0
        # This part needs a very clear spec on Z0 stamping in filtered MNA.
        # Assuming standard port stamping:
        # Y[1,1] should contain Rg1 (0.1S) + Z0_n1 (0.02S).
        # Y[0,0] should contain Rg1 (0.1S) + Z0_n1 (0.02S).
        # Y[0,1] and Y[1,0] should contain -Rg1 (-0.1S) -Z0_n1 (-0.02S) - this is wrong.
        # Port Z0 is between port node and ground.
        # Rg1 is between n1 (idx 1) and gnd (idx 0)
        # Y[1,1] += 1/10 + 1/50 = 0.1 + 0.02 = 0.12
        # Y[0,0] += 1/10 + 1/50 = 0.12
        # Y[1,0] -= (1/10 + 1/50) = -0.12
        # Y[0,1] -= (1/10 + 1/50) = -0.12
        # The assembler stamps components, and external ports are typically handled in solver/Schur.
        # Let's assume MnaAssembler stamps component admittances.
        # Y[1,1] should have 1/10. Y[0,0] should have 1/10. Y[1,0] and Y[0,1] have -1/10.
        np.testing.assert_allclose(Yn_full[1,1], 0.1, atol=1e-9) # Contribution from Rg1
        np.testing.assert_allclose(Yn_full[0,0], 0.1, atol=1e-9) # Contribution from Rg1
        np.testing.assert_allclose(Yn_full[1,0], -0.1, atol=1e-9)
        np.testing.assert_allclose(Yn_full[0,1], -0.1, atol=1e-9)

    def test_mao_no_override(self, base_mna_circuit): # base_mna_circuit is a fixture
        # Assuming MnaAssembler is imported
        from src.rfsim_core import MnaAssembler

        assembler = MnaAssembler(base_mna_circuit, active_nets_override=None)
        assert assembler.node_count == 3 # gnd, n1, n2
        assert "gnd" in assembler.node_map and "n1" in assembler.node_map and "n2" in assembler.node_map
        assert set(assembler.port_names) == {"n1", "n2"} # Order might vary

        Yn_full = assembler.assemble(1e9) # freq_hz = 1 GHz
        assert Yn_full.shape == (3,3)
        
        # Verify some expected non-zero entries based on full circuit connectivity
        idx_n1 = assembler.node_map["n1"]
        idx_n2 = assembler.node_map["n2"]
        # idx_gnd = assembler.node_map["gnd"] # Not used in these specific asserts, but good to have

        # Y_n1,n1 should include 1/Rg1 + 1/R12
        np.testing.assert_allclose(Yn_full[idx_n1, idx_n1], 1/10 + 1/20, atol=1e-9)
        
        # Y_n1,n2 should include -1/R12
        np.testing.assert_allclose(Yn_full[idx_n1, idx_n2], -1/20, atol=1e-9)


    def test_mao_empty_override(self, base_mna_circuit):
        assembler = MnaAssembler(base_mna_circuit, active_nets_override=set())
        assert assembler.node_count == 0
        assert assembler.port_names == []
        Yn_full = assembler.assemble(1e9)
        assert Yn_full.shape == (0,0)
        assert Yn_full.nnz == 0