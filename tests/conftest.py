# tests/conftest.py
import pytest
import numpy as np
from src.rfsim_core import (
    Circuit, NetlistParser, CircuitBuilder, ParameterManager, ComponentError,
    Resistor, Capacitor, Inductor, COMPONENT_REGISTRY,
    DCAnalyzer, TopologyAnalyzer, MnaAssembler,
    DCAnalysisError, TopologyAnalysisError, MnaInputError, SimulationError, SemanticValidationError,
    ureg, Quantity
)
from src.rfsim_core.data_structures import Component as ComponentDataStructure
from src.rfsim_core.data_structures import Port as PortDataStructure
from src.rfsim_core.data_structures import Net as NetDataStructure
from src.rfsim_core.parameters import ParameterDefinition
from src.rfsim_core.components.base import DCBehaviorType
from src.rfsim_core.constants import LARGE_ADMITTANCE_SIEMENS

# Common fixture for CircuitBuilder
@pytest.fixture
def circuit_builder_instance():
    return CircuitBuilder()

# Helper function to create a simple circuit programmatically and build it
def create_and_build_circuit(
    circuit_builder: CircuitBuilder,
    components_def: list,  # List of tuples: (id, type, params_dict, connections_dict)
    external_ports_def: dict = None,  # Dict: {port_name: Z0_str}
    global_params_def: dict = None,
    circuit_name: str = "TestCircuit",
    ground_name: str = "gnd"
) -> Circuit:
    """
    Programmatically creates a Circuit object from definitions and runs CircuitBuilder.
    components_def: e.g., [("R1", "Resistor", {"resistance": "10 ohm"}, {"0": "n1", "1": "gnd"})]
    external_ports_def: e.g., {"P1": "50 ohm"} (P1 is assumed to be a net name)
    global_params_def: e.g., {"my_global_res": "1k"}
    """
    parsed_circuit = Circuit(name=circuit_name, ground_net_name=ground_name)

    if global_params_def:
        setattr(parsed_circuit, 'raw_global_parameters', global_params_def)

    # Create nets and components based on definitions
    for comp_id, comp_type_str, params, connections in components_def:
        comp_data = ComponentDataStructure(instance_id=comp_id, component_type=comp_type_str, parameters=params)
        for port_yaml_id, net_name_str in connections.items():
            # This mimics NetlistParser's net creation and port linking
            port_obj = comp_data.add_port(port_yaml_id)
            port_obj.original_yaml_net_name = net_name_str # For SemanticValidator if needed
            net_obj = parsed_circuit.get_or_create_net(net_name_str)
            port_obj.net = net_obj
            if comp_data not in net_obj.connected_components:
                 net_obj.connected_components.append(comp_data)
        parsed_circuit.add_component(comp_data)

    if external_ports_def:
        for port_name, z0_str in external_ports_def.items():
            parsed_circuit.set_external_port(port_name, z0_str)
            # The following line was removed as it's redundant and caused the TypeError:
            # parsed_circuit.get_or_create_net(port_name, is_external=True)


    # Build the circuit (ParameterManager creation, sim_components instantiation)
    # This relies on CircuitBuilder correctly handling parameter definitions
    # based on the structure of `params` in components_def
    sim_circuit = circuit_builder.build_circuit(parsed_circuit)
    return sim_circuit