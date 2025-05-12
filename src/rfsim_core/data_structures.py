# --- src/rfsim_core/data_structures.py ---
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, TYPE_CHECKING

logger = logging.getLogger(__name__)

# Use TYPE_CHECKING to avoid circular imports for type hints
if TYPE_CHECKING:
    from .parameters import ParameterManager # Forward declaration
    # Add forward reference for ComponentBase if needed elsewhere
    from .components.base import ComponentBase

@dataclass
class Net:
    """Represents an electrical node (net) in the circuit."""
    name: str
    connected_components: List['Component'] = field(default_factory=list)
    is_ground: bool = False
    is_external: bool = False
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
    component: 'Component'
    port_id: str | int
    net: Optional[Net] = None
    original_yaml_net_name: Optional[str] = None # Added for SemanticValidator

    def __post_init__(self):
         if not isinstance(self.port_id, (str, int)):
             raise ValueError(f"Port ID must be a string or integer for component {getattr(self.component, 'instance_id', 'UNKNOWN')}, got {type(self.port_id)}")
         logger.debug(f"Port created for component {getattr(self.component, 'instance_id', 'UNKNOWN')}: {self.port_id}")

@dataclass
class Component:
    """Base representation of a circuit component (parsed data)."""
    component_type: str
    instance_id: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    ports: Dict[str | int, Port] = field(default_factory=dict)

    def __post_init__(self):
         if not isinstance(self.component_type, str) or not self.component_type:
             raise ValueError("Component type must be a non-empty string.")
         if not isinstance(self.instance_id, str) or not self.instance_id:
             raise ValueError("Component instance ID must be a non-empty string.")
         logger.debug(f"Component created: {self.instance_id} (Type: {self.component_type})")

    def __hash__(self):
        return hash(self.instance_id)

    def __eq__(self, other):
        if not isinstance(other, Component):
            return NotImplemented
        return self.instance_id == other.instance_id

    def add_port(self, port_id: str | int) -> 'Port':
        """Creates and adds a port to this component."""
        if port_id in self.ports:
            raise ValueError(f"Duplicate port ID '{port_id}' used in definition of component '{self.instance_id}'")
        port = Port(component=self, port_id=port_id)
        self.ports[port_id] = port
        logger.debug(f"Added port {port_id} to component {self.instance_id}")
        return port

@dataclass
class Circuit:
    """Represents the overall circuit structure and parsed data."""
    name: str = "UnnamedCircuit"
    components: Dict[str, Component] = field(default_factory=dict) # Raw component data
    nets: Dict[str, Net] = field(default_factory=dict)
    external_ports: Dict[str, Net] = field(default_factory=dict)
    external_port_impedances: Dict[str, str] = field(default_factory=dict)
    parameter_manager: Optional['ParameterManager'] = None # Use forward reference string
    ground_net_name: str = "gnd"

    def __post_init__(self):
        try:
            logger.info(f"Circuit object initialized: {self.name}")
            self.get_or_create_net(self.ground_net_name, is_ground=True)
        except Exception as e:
            raise

    def add_component(self, component: Component):
        if component.instance_id in self.components:
            raise ValueError(f"Duplicate component instance ID: {component.instance_id}")
        self.components[component.instance_id] = component
        logger.debug(f"Added component to circuit: {component.instance_id}")

    def add_net(self, net: Net):
        if net.name in self.nets:
            logger.warning(f"Attempted to add duplicate net: {net.name}. Using existing.")
            existing_net = self.nets[net.name]
            if net.is_ground and not existing_net.is_ground:
                existing_net.is_ground = True
                logger.info(f"Net '{net.name}' marked as ground.")
            return existing_net
        self.nets[net.name] = net
        logger.debug(f"Added net to circuit: {net.name}")
        return net

    def get_or_create_net(self, name: str, is_ground: bool = False) -> Net:
        if name in self.nets:
            net = self.nets[name]
            if is_ground and not net.is_ground:
                net.is_ground = True
                logger.info(f"Net '{name}' marked as ground.")
            return net
        else:
            is_gnd_flag = is_ground or (name == self.ground_net_name)
            new_net = Net(name=name, is_ground=is_gnd_flag)
            return self.add_net(new_net)

    def get_ground_net(self) -> Net:
        if self.ground_net_name not in self.nets:
            logger.warning(f"Ground net '{self.ground_net_name}' not found, creating.")
            return self.get_or_create_net(self.ground_net_name, is_ground=True)
        return self.nets[self.ground_net_name]

    def set_external_port(self, port_name: str, reference_impedance_str: str):
        if not isinstance(port_name, str) or not port_name:
             raise ValueError("External port name must be a non-empty string.")
        if not isinstance(reference_impedance_str, str):
             raise ValueError(f"Reference impedance for port '{port_name}' must be a string (e.g., '50 ohm').")

        net = self.get_or_create_net(port_name)
        if net.is_external and self.external_port_impedances.get(port_name) != reference_impedance_str:
             logger.warning(f"Redefining external port '{port_name}'. Previous impedance: '{self.external_port_impedances.get(port_name)}'. New: '{reference_impedance_str}'.")

        net.is_external = True
        self.external_ports[port_name] = net
        self.external_port_impedances[port_name] = reference_impedance_str
        logger.debug(f"Net '{port_name}' set as external port with Z0='{reference_impedance_str}'.")