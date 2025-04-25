import pytest
from pint import UnitRegistry

from rfsim_core import NetlistParser


# --- Test Fixtures ---
@pytest.fixture
def parser():
    """Provides a NetlistParser instance."""
    return NetlistParser()

@pytest.fixture
def valid_yaml_string():
    """Provides a basic valid YAML string."""
    return """
circuit_name: Test RLC Circuit
parameters:
  supply_voltage: '5 V'
  default_cap: '10 pF'
components:
  - type: Resistor
    id: R1
    ports: { p1: net1, p2: gnd }
    parameters:
      resistance: '1kohm' # Component params are strings for now
  - type: Capacitor
    id: C1
    ports: { p1: net1, p2: gnd }
    parameters:
      capacitance: default_cap # Raw string, resolution later
  - type: Inductor
    id: L1
    ports: { p1: net1, p2: 'out' } # Use quotes for net name 'out'
    parameters:
      inductance: '1 uH'
ports:
  - id: out
    reference_impedance: '50 ohm'
"""

@pytest.fixture
def yaml_missing_components():
    return """
ports:
  - id: out
    reference_impedance: '50 ohm'
"""

@pytest.fixture
def yaml_bad_schema_comp():
    return """
components:
  - type: Resistor
    # id is missing
    ports: { p1: n1, p2: gnd }
"""

@pytest.fixture
def yaml_bad_schema_port():
    return """
components:
  - type: Resistor
    id: R1
    ports: { p1: n1, p2: gnd }
ports:
  - id: p1
    # reference_impedance is missing
"""

@pytest.fixture
def yaml_bad_global_param_unit():
     return """
parameters:
  bad_param: '10 foobars'
components:
  - type: Resistor
    id: R1
    ports: { p1: n1, p2: gnd }
"""

@pytest.fixture
def yaml_duplicate_comp_id():
    return """
components:
  - type: Resistor
    id: R1
    ports: { p1: n1, p2: gnd }
  - type: Capacitor
    id: R1 # Duplicate ID
    ports: { p1: n1, p2: gnd }
"""

@pytest.fixture
def yaml_malformed():
    return "components: \n- type: R\n id: R1\n ports: {p1: n1}" # Indentation error