# tests/components/conftest.py
import pytest
import numpy as np
from typing import Dict, List

from rfsim_core.units import ureg, Quantity
from rfsim_core.components.base import ComponentBase, register_component, DCBehaviorType
from rfsim_core.components.capabilities import (
    IMnaContributor, IDcContributor, provides, ComponentCapability
)
from rfsim_core.parser import NetlistParser, ParsedLeafComponentData
from rfsim_core.circuit_builder import CircuitBuilder
from rfsim_core.simulation import run_sweep
from rfsim_core.components.exceptions import ComponentError
from rfsim_core.simulation.exceptions import SingleLevelSimulationFailure
from rfsim_core import SimulationRunError


def build_and_run_bad_component_test(component_yaml_str: str, tmp_path):
    """
    A generic test executor for bad component plugins. It creates a netlist,
    builds the circuit, and runs a simulation, expecting it to fail with a
    diagnosable error.

    Returns:
        The ComponentError that was the root cause of the simulation failure.
    """
    netlist_path = tmp_path / "bad_component_test.yaml"
    netlist_path.write_text(f"""
circuit_name: BadComponentTest
components:
{component_yaml_str}
ports:
  # --- DEFINITIVE FIX: Use the correct, mandated syntax for impedance literals. ---
  # The `reference_impedance` field expects a simple string literal that `pint` can
  # parse directly (e.g., "50 ohm"). The `Quantity(...)` syntax is reserved for
  # expressions in the `parameters` block, which are handled by the ParameterManager.
  # This change corrects the test to align with the framework's API contract.
  - {{id: p1, reference_impedance: "50 ohm"}}
""")

    parser = NetlistParser()
    builder = CircuitBuilder()

    # The build should succeed as these are runtime errors
    parsed_tree = parser.parse_to_circuit_tree(netlist_path)
    sim_circuit = builder.build_simulation_model(parsed_tree)

    freqs = np.array([1e9])

    # The simulation run must fail with a diagnosable error
    with pytest.raises(SimulationRunError) as excinfo:
        run_sweep(sim_circuit, freqs)

    # The top-level error wraps a SingleLevelSimulationFailure, which in turn wraps
    # the actual ComponentError. This logic correctly unwraps the exception chain
    # to find the root cause, which is what the tests need to inspect.
    cause = excinfo.value.__cause__
    assert isinstance(cause, (SingleLevelSimulationFailure, ComponentError)), \
        "Expected SimulationRunError to wrap a known simulation failure type"
    
    if isinstance(cause, SingleLevelSimulationFailure):
        return cause.original_error
    return cause


# --- Fixtures for Malicious/Incompetent Components ---
# These fixtures create minimal, valid ParsedLeafComponentData objects
# to satisfy the ComponentBase constructor for the mock components.

@pytest.fixture
def mock_ir_data(tmp_path):
    """Provides a minimal ParsedLeafComponentData for mock component initialization."""
    return ParsedLeafComponentData(
        instance_id="U1",
        component_type="Mock",
        raw_ports_dict={0: 'p1', 1: 'gnd'},
        raw_parameters_dict={},
        source_yaml_path=tmp_path
    )

# The following "bad" component definitions are INTENTIONALLY left as-is.
# Their purpose is to act as test vectors that violate the framework's contracts.
# The tests that use these components verify that the FRAMEWORK correctly
# detects and rejects these violations.

# 1. Component that returns a raw number instead of a Quantity
@register_component("BadComponentReturnsNumber")
class BadComponentReturnsNumber(ComponentBase):
    @provides(IMnaContributor)
    class MnaContributor:
        def get_mna_stamps(self, component, freq_hz_array, all_evaluated_params):
            # VIOLATION: Returns a raw NumPy array in the stamp tuple
            return [(np.ones((len(freq_hz_array), 2, 2)), [0, 1])]

    @classmethod
    def declare_parameters(cls) -> Dict[str, str]: return {}
    @classmethod
    def declare_ports(cls) -> List[str | int]: return [0, 1]
    def is_structurally_open(self, resolved_constant_params) -> bool: return False

# 2. Component that returns the wrong physical dimension
@register_component("BadComponentReturnsWrongDimension")
class BadComponentReturnsWrongDimension(ComponentBase):
    @provides(IMnaContributor)
    class MnaContributor:
        def get_mna_stamps(self, component, freq_hz_array, all_evaluated_params):
            # VIOLATION: Returns resistance instead of admittance
            stamp = np.ones((len(freq_hz_array), 2, 2))
            return [(Quantity(stamp, ureg.ohm), [0, 1])]

    @classmethod
    def declare_parameters(cls) -> Dict[str, str]: return {}
    @classmethod
    def declare_ports(cls) -> List[str | int]: return [0, 1]
    def is_structurally_open(self, resolved_constant_params) -> bool: return False

# 3. Component that returns a scalar instead of a vectorized array
@register_component("BadComponentReturnsWrongShape")
class BadComponentReturnsWrongShape(ComponentBase):
    @provides(IMnaContributor)
    class MnaContributor:
        def get_mna_stamps(self, component, freq_hz_array, all_evaluated_params):
            # VIOLATION: Returns a scalar quantity, not a vectorized one. This will cause an IndexError
            # when the simulation executive tries to slice it at the current frequency index.
            stamp = np.ones((2, 2))
            return [(Quantity(stamp, ureg.siemens), [0, 1])]

    @classmethod
    def declare_parameters(cls) -> Dict[str, str]: return {}
    @classmethod
    def declare_ports(cls) -> List[str | int]: return [0, 1]
    def is_structurally_open(self, resolved_constant_params) -> bool: return False

# 4. Component that just raises a random error
@register_component("BadComponentRaisesRandomError")
class BadComponentRaisesRandomError(ComponentBase):
    @provides(IMnaContributor)
    class MnaContributor:
        def get_mna_stamps(self, component, freq_hz_array, all_evaluated_params):
            # VIOLATION: Raises an unexpected, non-diagnosable error
            raise ValueError("Something went wrong inside the component!")

    @classmethod
    def declare_parameters(cls) -> Dict[str, str]: return {}
    @classmethod
    def declare_ports(cls) -> List[str | int]: return [0, 1]
    def is_structurally_open(self, resolved_constant_params) -> bool: return False

# 5. Component for testing inheritance
class TwoPortBase(ComponentBase):
    # This capability should be inherited by children
    @provides(IDcContributor)
    class DcContributor:
        def get_dc_behavior(self, component, all_dc_params):
            return (DCBehaviorType.OPEN_CIRCUIT, None)

    # These will be implemented by children
    @classmethod
    def declare_parameters(cls) -> Dict[str, str]: return {}
    @classmethod
    def declare_ports(cls) -> List[str | int]: return [0, 1]
    def is_structurally_open(self, resolved_constant_params) -> bool: return False

@register_component("InheritingComponent")
class InheritingComponent(TwoPortBase):
    # This component provides its own MNA capability
    @provides(IMnaContributor)
    class MnaContributor:
        def get_mna_stamps(self, component, freq_hz_array, all_evaluated_params):
            stamp = np.zeros((len(freq_hz_array), 2, 2))
            return [(Quantity(stamp, ureg.siemens), [0, 1])]

# Fixture to provide the helper function to tests
@pytest.fixture
def run_bad_comp_test_harness(tmp_path):
    """Provides a test harness that runs a simulation with a bad component and returns the resulting ComponentError."""
    return lambda yaml_str: build_and_run_bad_component_test(yaml_str, tmp_path)