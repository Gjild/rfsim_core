# src/rfsim_core/data_structures.py
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set

# Get a logger for this module
logger = logging.getLogger(__name__)

# Forward declaration for type hints within classes
class Component:
    pass

# Forward declaration placeholder
class ParameterManager: 
    pass

@dataclass
class Net:
    """Represents an electrical node (net) in the circuit."""
    name: str
    # Components connected to this net (populated during parsing)
    connected_components: List['Component'] = field(default_factory=list) # Use string hint
    is_ground: bool = False
    is_external: bool = False # Is this net an external port?
    index: Optional[int] = None # Node index for MNA (assigned later)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Net name must be a non-empty string.")
        logger.debug(f"Net created: {self.name}")

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, Net):
            return NotImplemented
        return self.name == other.name

@dataclass
class Port:
    """Represents a connection point on a component."""
    component: 'Component' # Use string hint
    port_id: str | int   # Identifier within the component (e.g., 'p1', 1)
    net: Optional[Net] = None # The net this port connects to (set during connection)

    def __post_init__(self):
         if not isinstance(self.port_id, (str, int)):
             raise ValueError(f"Port ID must be a string or integer for component {getattr(self.component, 'instance_id', 'UNKNOWN')}, got {type(self.port_id)}")
         logger.debug(f"Port created for component {getattr(self.component, 'instance_id', 'UNKNOWN')}: {self.port_id}")

@dataclass
class Component:
    """Base representation of a circuit component."""
    component_type: str
    instance_id: str
    # Raw parameters from netlist initially. Values are typically strings or numbers.
    # These will be processed later into Quantity objects where applicable.
    parameters: Dict[str, Any] = field(default_factory=dict)
    # Ports dict populated during parsing, mapping port ID to Port object
    ports: Dict[str | int, Port] = field(default_factory=dict)

    def __post_init__(self):
         if not isinstance(self.component_type, str) or not self.component_type:
             raise ValueError("Component type must be a non-empty string.")
         if not isinstance(self.instance_id, str) or not self.instance_id:
             raise ValueError("Component instance ID must be a non-empty string.")
         logger.debug(f"Component created: {self.instance_id} (Type: {self.component_type})")
         # Now create Port objects immediately based on parameters if needed?
         # No, ports are defined by the connections in the netlist, not params.

    def __hash__(self):
        return hash(self.instance_id)

    def __eq__(self, other):
        if not isinstance(other, Component):
            return NotImplemented
        return self.instance_id == other.instance_id

    def add_port(self, port_id: str | int) -> Port:
        """Creates and adds a port to this component."""
        if port_id in self.ports:
            raise ValueError(f"Duplicate port ID '{port_id}' for component '{self.instance_id}'")
        port = Port(component=self, port_id=port_id)
        self.ports[port_id] = port
        logger.debug(f"Added port {port_id} to component {self.instance_id}")
        return port

@dataclass
class Circuit:
    """Represents the overall circuit."""
    name: str = "UnnamedCircuit"
    components: Dict[str, Component] = field(default_factory=dict)
    nets: Dict[str, Net] = field(default_factory=dict)
    # Map external port name (which is also a net name) to the Net object
    external_ports: Dict[str, Net] = field(default_factory=dict)
    # Store the raw reference impedance string for each external port
    external_port_impedances: Dict[str, str] = field(default_factory=dict)
    # Global parameters processed by ParameterManager
    parameter_manager: Optional[ParameterManager] = None # Placeholder for manager instance
    ground_net_name: str = "gnd"

    def __post_init__(self):
        logger.info(f"Circuit object initialized: {self.name}")
        # Ensure ground net exists by default
        self.get_or_create_net(self.ground_net_name, is_ground=True)

    def add_component(self, component: Component):
        if component.instance_id in self.components:
            raise ValueError(f"Duplicate component instance ID: {component.instance_id}")
        self.components[component.instance_id] = component
        logger.debug(f"Added component to circuit: {component.instance_id}")

    def add_net(self, net: Net):
        if net.name in self.nets:
            logger.warning(f"Attempted to add duplicate net: {net.name}. Using existing.")
            # Ensure flags are consistent if re-adding (e.g. ground status)
            existing_net = self.nets[net.name]
            if net.is_ground and not existing_net.is_ground:
                existing_net.is_ground = True
                logger.info(f"Net '{net.name}' marked as ground.")
            return existing_net
        self.nets[net.name] = net
        logger.debug(f"Added net to circuit: {net.name}")
        return net

    def get_or_create_net(self, name: str, is_ground: bool = False) -> Net:
        """Gets a net if it exists, otherwise creates and adds it."""
        if name in self.nets:
            net = self.nets[name]
            # Ensure ground status is updated if specified later
            if is_ground and not net.is_ground:
                net.is_ground = True
                logger.info(f"Net '{name}' marked as ground.")
            return net
        else:
            # Determine ground status primarily if name matches ground_net_name
            is_gnd_flag = is_ground or (name == self.ground_net_name)
            new_net = Net(name=name, is_ground=is_gnd_flag)
            return self.add_net(new_net)

    def get_ground_net(self) -> Net:
        """Returns the designated ground net."""
        if self.ground_net_name not in self.nets:
            # This should ideally not happen if post_init works
            logger.warning(f"Ground net '{self.ground_net_name}' not found, creating.")
            return self.get_or_create_net(self.ground_net_name, is_ground=True)
        return self.nets[self.ground_net_name]

    def set_external_port(self, port_name: str, reference_impedance_str: str):
        """Designates a net as an external port and stores its impedance."""
        if not isinstance(port_name, str) or not port_name:
             raise ValueError("External port name must be a non-empty string.")
        if not isinstance(reference_impedance_str, str):
             # Allow numbers maybe? For now, stick to spec: string only
             raise ValueError(f"Reference impedance for port '{port_name}' must be a string (e.g., '50 ohm').")

        net = self.get_or_create_net(port_name)
        if net.is_external and self.external_port_impedances.get(port_name) != reference_impedance_str:
             logger.warning(f"Redefining external port '{port_name}'. Previous impedance: '{self.external_port_impedances.get(port_name)}'. New: '{reference_impedance_str}'.")

        net.is_external = True
        self.external_ports[port_name] = net
        self.external_port_impedances[port_name] = reference_impedance_str
        logger.debug(f"Net '{port_name}' set as external port with Z0='{reference_impedance_str}'.")