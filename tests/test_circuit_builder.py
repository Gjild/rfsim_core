import pytest
import numpy as np

from src.rfsim_core.parser import NetlistParser
from src.rfsim_core.circuit_builder import CircuitBuilder, CircuitBuildError
from src.rfsim_core.data_structures import Circuit, Component
from src.rfsim_core.parameters import ParameterManager, ParameterError, ParameterDefinitionError, CircularParameterDependencyError
from src.rfsim_core.components import Resistor, Capacitor, Inductor
from src.rfsim_core.units import ureg, Quantity, pint

# --- Fixtures for Parsed Circuit Objects ---
# Simulate the output of NetlistParser.parse()

@pytest.fixture
def parsed_circuit_constants() -> Circuit:
    """Parsed circuit object with only constant parameters."""
    circuit = Circuit(name="ConstantsOnly", ground_net_name="gnd")
    # Raw global params stored by parser
    setattr(circuit, 'raw_global_parameters', {"L_global": "1 nH"})
    # Raw component data
    circuit.add_component(Component(
        instance_id="R1", component_type="Resistor",
        parameters={"resistance": "50 ohm"}
    ))
    circuit.add_component(Component(
        instance_id="L1", component_type="Inductor",
        parameters={"inductance": "L_global"} # Reference global
    ))
    # Add minimal nets/ports for basic structure
    circuit.get_or_create_net("N1")
    circuit.get_or_create_net("gnd", is_ground=True)
    circuit.components["R1"].add_port(0).net = circuit.nets["N1"]
    circuit.components["R1"].add_port(1).net = circuit.nets["gnd"]
    circuit.components["L1"].add_port(0).net = circuit.nets["N1"]
    circuit.components["L1"].add_port(1).net = circuit.nets["gnd"]
    circuit.set_external_port("N1", "50 ohm")
    return circuit

@pytest.fixture
def parsed_circuit_global_expr() -> Circuit:
    """Parsed circuit with a global expression."""
    circuit = Circuit(name="GlobalExpr", ground_net_name="gnd")
    setattr(circuit, 'raw_global_parameters', {
        "R_base": "100 ohm",
        "gain": {
            "expression": "sqrt(R_base / 50)", # Use R_base
            "dimension": "dimensionless"
        }
    })
    circuit.add_component(Component(
        instance_id="R1", component_type="Resistor",
        parameters={"resistance": "R_base"}
    ))
    # Add minimal nets/ports
    circuit.get_or_create_net("N1")
    circuit.get_or_create_net("gnd", is_ground=True)
    circuit.components["R1"].add_port(0).net = circuit.nets["N1"]
    circuit.components["R1"].add_port(1).net = circuit.nets["gnd"]
    circuit.set_external_port("N1", "50 ohm")
    return circuit

@pytest.fixture
def parsed_circuit_instance_expr() -> Circuit:
    """Parsed circuit with an instance expression."""
    circuit = Circuit(name="InstanceExpr", ground_net_name="0")
    setattr(circuit, 'raw_global_parameters', {"R_load": "50 ohm"})
    circuit.add_component(Component(
        instance_id="R1", component_type="Resistor",
        parameters={
            "resistance": { # Instance specific expression
                "expression": "R_load * 2",
                "dimension": "ohm" # Present in YAML, ignored by builder
            }
        }
    ))
    circuit.add_component(Component(
        instance_id="R2", component_type="Resistor",
        parameters={"resistance": "R_load"} # Instance referencing global
    ))
    # Add minimal nets/ports
    circuit.get_or_create_net("N1")
    circuit.get_or_create_net("0", is_ground=True)
    circuit.components["R1"].add_port(0).net = circuit.nets["N1"]
    circuit.components["R1"].add_port(1).net = circuit.nets["0"]
    circuit.components["R2"].add_port(0).net = circuit.nets["N1"]
    circuit.components["R2"].add_port(1).net = circuit.nets["0"]
    circuit.set_external_port("N1", "50 ohm")
    return circuit


@pytest.fixture
def parsed_circuit_interdependent() -> Circuit:
    """Parsed circuit with inter-dependent instance parameters."""
    circuit = Circuit(name="InterDep", ground_net_name="gnd")
    setattr(circuit, 'raw_global_parameters', {"mult": "2"})
    circuit.add_component(Component(
        instance_id="R1", component_type="Resistor",
        parameters={"resistance": "100 ohm"}
    ))
    circuit.add_component(Component(
        instance_id="R2", component_type="Resistor",
        parameters={ # R2 depends on R1.resistance and _rfsim_global_.mult
            "resistance": {
                "expression": "R1.resistance * mult",
                "dimension": "ohm"
            }
        }
    ))
    # Add minimal nets/ports
    circuit.get_or_create_net("N1")
    circuit.get_or_create_net("gnd", is_ground=True)
    circuit.components["R1"].add_port(0).net = circuit.nets["N1"]
    circuit.components["R1"].add_port(1).net = circuit.nets["gnd"]
    circuit.components["R2"].add_port(0).net = circuit.nets["N1"]
    circuit.components["R2"].add_port(1).net = circuit.nets["gnd"]
    return circuit

@pytest.fixture
def parsed_circuit_circular_dep() -> Circuit:
    """Parsed circuit with a circular parameter dependency."""
    circuit = Circuit(name="Circular", ground_net_name="gnd")
    setattr(circuit, 'raw_global_parameters', {
        "p1": {"expression": "p2 * 2", "dimension": "volt"},
        "p2": {"expression": "p1 / 2", "dimension": "volt"}
    })
    circuit.add_component(Component( # Dummy component
        instance_id="D1", component_type="Resistor", parameters={"resistance": "1 ohm"}
    ))
    return circuit # No need for connections for parameter test

@pytest.fixture
def parsed_circuit_bad_ports() -> Circuit:
    """Parsed circuit with component using undeclared ports."""
    circuit = Circuit(name="BadPorts", ground_net_name="gnd")
    setattr(circuit, 'raw_global_parameters', {})
    circuit.add_component(Component(
        instance_id="R1", component_type="Resistor",
        parameters={"resistance": "50 ohm"}
        # Missing ports definition in raw Component data will be caught later
    ))
     # Manually add bad port connections to simulate parser output
    comp = circuit.components["R1"]
    net1 = circuit.get_or_create_net("N1")
    gnd = circuit.get_or_create_net("gnd", is_ground=True)
    # Try connecting a port ID that Resistor doesn't declare
    bad_port_id = 'bad_port'
    # Need to manually add to comp.ports dict for builder to see it
    from src.rfsim_core.data_structures import Port
    comp.ports[bad_port_id] = Port(component=comp, port_id=bad_port_id, net=net1)
    comp.ports[1] = Port(component=comp, port_id=1, net=gnd) # Valid port
    return circuit

# --- Test Class ---

class TestCircuitBuilder:

    def test_build_constants_only(self, parsed_circuit_constants):
        """Test building a circuit with only constant parameters."""
        builder = CircuitBuilder()
        sim_circuit = builder.build_circuit(parsed_circuit_constants)

        assert isinstance(sim_circuit, Circuit)
        assert sim_circuit.name == "ConstantsOnly"
        assert sim_circuit.parameter_manager is not None
        assert isinstance(sim_circuit.parameter_manager, ParameterManager)
        pm = sim_circuit.parameter_manager

        # Check parameter manager state
        assert "_rfsim_global_.L_global" in pm.get_all_internal_names()
        assert "R1.resistance" in pm.get_all_internal_names()
        assert "L1.inductance" in pm.get_all_internal_names()
        assert pm.is_constant("_rfsim_global_.L_global")
        assert pm.is_constant("R1.resistance")
        assert pm.is_constant("L1.inductance")
        assert pm.get_constant_value("_rfsim_global_.L_global") == ureg.Quantity("1 nH")
        assert pm.get_constant_value("R1.resistance") == ureg.Quantity("50 ohm")
        # L1 references L_global, it should also be treated as constant after PM build resolves this link conceptually
        assert pm.get_constant_value("L1.inductance") == ureg.Quantity("1 nH")
        assert pm.get_declared_dimension("L1.inductance") == "henry" # Henry

        # Check sim components
        assert "R1" in sim_circuit.sim_components
        assert "L1" in sim_circuit.sim_components
        assert isinstance(sim_circuit.sim_components["R1"], Resistor)
        assert isinstance(sim_circuit.sim_components["L1"], Inductor)

        # Check component parameter manager reference and internal names
        r1_sim = sim_circuit.sim_components["R1"]
        assert r1_sim.parameter_manager is pm # Should be the same object
        assert r1_sim.parameter_internal_names == ["R1.resistance"]

        l1_sim = sim_circuit.sim_components["L1"]
        assert l1_sim.parameter_manager is pm
        assert l1_sim.parameter_internal_names == ["L1.inductance"]

    def test_build_with_global_expr(self, parsed_circuit_global_expr):
        """Test building with global expressions."""
        builder = CircuitBuilder()
        sim_circuit = builder.build_circuit(parsed_circuit_global_expr)
        pm = sim_circuit.parameter_manager

        assert "_rfsim_global_.R_base" in pm.get_all_internal_names()
        assert "_rfsim_global_.gain" in pm.get_all_internal_names()
        assert "R1.resistance" in pm.get_all_internal_names()

        assert pm.is_constant("_rfsim_global_.R_base")
        assert not pm.is_constant("_rfsim_global_.gain")
        assert pm.is_constant("R1.resistance") # References a constant

        assert pm.get_dependencies("_rfsim_global_.gain") == {"_rfsim_global_.R_base"} # Sympy parsing might simplify unit away if not careful
        assert pm.get_declared_dimension("_rfsim_global_.gain") == "dimensionless" # dimensionless

        assert "R1" in sim_circuit.sim_components
        r1_sim = sim_circuit.sim_components["R1"]
        assert r1_sim.parameter_manager is pm
        assert r1_sim.parameter_internal_names == ["R1.resistance"]

    def test_build_with_instance_expr(self, parsed_circuit_instance_expr):
        """Test building with instance expressions."""
        builder = CircuitBuilder()
        sim_circuit = builder.build_circuit(parsed_circuit_instance_expr)
        pm = sim_circuit.parameter_manager

        assert "_rfsim_global_.R_load" in pm.get_all_internal_names()
        assert "R1.resistance" in pm.get_all_internal_names()
        assert "R2.resistance" in pm.get_all_internal_names()

        assert pm.is_constant("_rfsim_global_.R_load")
        assert not pm.is_constant("R1.resistance") # Defined by expr
        assert pm.is_constant("R2.resistance")    # References constant

        assert pm.get_dependencies("R1.resistance") == {"_rfsim_global_.R_load"}
        assert pm.get_declared_dimension("R1.resistance") == "ohm" # ohm

        assert "R1" in sim_circuit.sim_components
        assert "R2" in sim_circuit.sim_components
        r1_sim = sim_circuit.sim_components["R1"]
        assert r1_sim.parameter_manager is pm
        assert r1_sim.parameter_internal_names == ["R1.resistance"]
        r2_sim = sim_circuit.sim_components["R2"]
        assert r2_sim.parameter_manager is pm
        assert r2_sim.parameter_internal_names == ["R2.resistance"]


    def test_build_interdependent(self, parsed_circuit_interdependent):
        """Test building with interdependent instance parameters (no cycle)."""
        builder = CircuitBuilder()
        sim_circuit = builder.build_circuit(parsed_circuit_interdependent)
        pm = sim_circuit.parameter_manager

        assert "_rfsim_global_.mult" in pm.get_all_internal_names()
        assert "R1.resistance" in pm.get_all_internal_names()
        assert "R2.resistance" in pm.get_all_internal_names()

        assert pm.is_constant("R1.resistance")
        assert not pm.is_constant("R2.resistance")

        # Check dependency of R2.resistance
        # Should depend on R1.resistance and _rfsim_global_.mult
        assert pm.get_dependencies("R2.resistance") == {"R1.resistance", "_rfsim_global_.mult"}

        assert "R1" in sim_circuit.sim_components
        assert "R2" in sim_circuit.sim_components

    def test_build_error_missing_global_dimension(self):
        """Test build error if global expression lacks dimension in YAML."""
        circuit = Circuit(name="NoDim", ground_net_name="gnd")
        setattr(circuit, 'raw_global_parameters', {
            "gain": {"expression": "freq/1e9"} # Missing dimension
        })
        circuit.add_component(Component("D1", "Resistor", {"resistance":"1"})) # Dummy

        builder = CircuitBuilder()
        with pytest.raises(CircuitBuildError) as excinfo:
            builder.build_circuit(circuit)
        # Check underlying error type is ParameterDefinitionError
        assert isinstance(excinfo.value.__cause__, ParameterDefinitionError)
        assert "missing mandatory 'dimension' key" in str(excinfo.value)

    def test_build_error_invalid_global_constant_unit(self):
        """Test build error for global constant with invalid units."""
        circuit = Circuit(name="BadUnit", ground_net_name="gnd")
        setattr(circuit, 'raw_global_parameters', {"bad_val": "10 foobars"})
        circuit.add_component(Component("D1", "Resistor", {"resistance":"1"})) # Dummy

        builder = CircuitBuilder()
        with pytest.raises(CircuitBuildError) as excinfo:
            builder.build_circuit(circuit)
        assert isinstance(excinfo.value.__cause__, ParameterDefinitionError)
        assert "Error parsing global constant parameter 'bad_val'" in str(excinfo.value)
        assert "'foobars' is not defined" in str(excinfo.value) # From Pint

    #def test_build_error_undeclared_instance_param(self):
    #   """Test build error if instance provides an undeclared parameter."""
    #    circuit = Circuit(name="Undeclared", ground_net_name="gnd")
    #    setattr(circuit, 'raw_global_parameters', {})
    #    circuit.add_component(Component(
    #        instance_id="R1", component_type="Resistor",
    #        parameters={
    #            "resistance": "50 ohm",
    #            "color": "red" # Resistor does not declare 'color'
    #        }
    #    ))
    #    # Add minimal nets/ports
    #    circuit.get_or_create_net("N1")
    #    circuit.get_or_create_net("gnd", is_ground=True)
    #    circuit.components["R1"].add_port(0).net = circuit.nets["N1"]
    #    circuit.components["R1"].add_port(1).net = circuit.nets["gnd"]

    #    builder = CircuitBuilder()
    #    with pytest.raises(CircuitBuildError) as excinfo:
    #        builder.build_circuit(circuit)
    #    assert isinstance(excinfo.value.__cause__, ParameterDefinitionError)
    #    assert "provided parameter 'color' which is not declared" in str(excinfo.value)

    #def test_build_error_missing_required_instance_param(self):
    #    """Test build error if instance definition omits a required parameter."""
    #    circuit = Circuit(name="MissingParam", ground_net_name="gnd")
    #    setattr(circuit, 'raw_global_parameters', {})
    #    circuit.add_component(Component(
    #        instance_id="R1", component_type="Resistor",
    #        parameters={} # Missing 'resistance'
    #    ))
    #    # Add minimal nets/ports
    #    circuit.get_or_create_net("N1")
    #    circuit.get_or_create_net("gnd", is_ground=True)
    #    circuit.components["R1"].add_port(0).net = circuit.nets["N1"]
    #    circuit.components["R1"].add_port(1).net = circuit.nets["gnd"]

    #    builder = CircuitBuilder()
    #    with pytest.raises(CircuitBuildError) as excinfo:
    #        builder.build_circuit(circuit)
    #    assert isinstance(excinfo.value.__cause__, ParameterDefinitionError)
    #    assert "Required parameter 'resistance' missing for component instance 'R1'" in str(excinfo.value)

    def test_build_error_circular_dependency(self, parsed_circuit_circular_dep):
        """Test build error for circular parameter dependencies."""
        builder = CircuitBuilder()
        with pytest.raises(CircuitBuildError) as excinfo:
            builder.build_circuit(parsed_circuit_circular_dep)
        # Check that the underlying cause is the specific circular dependency error
        assert isinstance(excinfo.value.__cause__, CircularParameterDependencyError)
        assert "Circular dependency detected:" in str(excinfo.value)

    #def test_build_error_port_mismatch(self, parsed_circuit_bad_ports):
    #    """Test build error for component using undeclared ports."""
    #    builder = CircuitBuilder()
    #    with pytest.raises(CircuitBuildError) as excinfo:
    #        builder.build_circuit(parsed_circuit_bad_ports)
    #    # Error message comes from port validation section
    #    assert "uses undeclared ports: ['bad_port']" in str(excinfo.value)

    def test_build_empty_circuit(self):
        """Test building an empty circuit (no components/ports)."""
        circuit = Circuit(name="Empty", ground_net_name="gnd")
        setattr(circuit, 'raw_global_parameters', {})
        # No components added
        circuit.get_or_create_net("gnd", is_ground=True) # Ensure ground exists

        builder = CircuitBuilder()
        # Should build successfully, just creating an empty PM and no sim_components
        sim_circuit = builder.build_circuit(circuit)

        assert isinstance(sim_circuit, Circuit)
        assert sim_circuit.parameter_manager is not None
        assert len(sim_circuit.parameter_manager.get_all_internal_names()) == 0
        assert len(sim_circuit.sim_components) == 0