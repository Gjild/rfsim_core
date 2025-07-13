# tests/components/conftest.py

"""
Definitive test configuration and fixtures for the RFSim Core component subsystem.

This file provides the foundational tools required for ruthlessly testing the
component system's contracts and the framework's resilience. It is the central
repository for shared testing logic and malicious component definitions, ensuring
that the main test file (`test_component_system.py`) remains clean, readable,
and focused on expressing test intent.

This file contains:

1.  **Test Harnesses:** Reusable functions (`build_and_run_*_test`) that
    encapsulate the boilerplate of creating a netlist, building a circuit,
    and running a simulation designed to fail in a specific, predictable way.
    This includes separate, targeted harnesses for AC and DC analysis failures,
    allowing for more focused and efficient testing of specific framework gateways.

2.  **Malicious Component Plugins:** A suite of custom component classes that are
    intentionally designed to violate the framework's API contracts. These are
    critical for verifying that the framework's "Validation Gateways" are robust,
    catch malformed data, and provide actionable diagnostics. They are the
    adversaries against which the framework's defenses are tested.

3.  **Helper Components:** Base classes and inheriting components used to verify
    the correctness of the capability discovery system's Method Resolution Order
    (MRO) traversal and inheritance logic.

4.  **Pytest Fixtures:** Standard pytest fixtures that provide the harnesses and
    other common test objects to the test functions, promoting code reuse and
    clarity according to pytest best practices.
"""

import pytest
import numpy as np
import pint
from typing import Dict, List, Optional

# --- Core Framework Imports ---
from rfsim_core.units import ureg, Quantity
from rfsim_core.parser import NetlistParser
from rfsim_core.circuit_builder import CircuitBuilder
from rfsim_core.simulation import run_sweep
from rfsim_core.cache import SimulationCache
from rfsim_core.errors import CircuitBuildError, SimulationRunError

# --- Component Subsystem Imports ---
from rfsim_core.components.base import ComponentBase, register_component, DCBehaviorType
from rfsim_core.components.capabilities import (
    IMnaContributor, IDcContributor, ITopologyContributor, provides, IConnectivityProvider
)
from rfsim_core.components.exceptions import ComponentError

# --- Analysis Subsystem Imports ---
from rfsim_core.analysis import DCAnalyzer
from rfsim_core.analysis.exceptions import DCAnalysisError

# --- Top-Level Exception Imports for Unpacking ---
from rfsim_core.simulation.exceptions import SingleLevelSimulationFailure


# =============================================================================
# == Test Harnesses
# =============================================================================

def build_and_run_bad_ac_component_test(component_yaml_str: str, tmp_path, extra_yaml: str = ""):
    """
    Test harness for components designed to fail during AC MNA stamp collection.

    This function orchestrates a full `run_sweep` simulation that is expected
    to fail with a `SimulationRunError`. It correctly unpacks the exception
    chain to return the root `ComponentError`, which is the object of interest
    for contract validation tests.

    Args:
        component_yaml_str: A string containing the 'components' block of a
                            YAML netlist defining the malicious component.
        tmp_path: The pytest `tmp_path` fixture for creating a temporary file.
        extra_yaml: Optional additional YAML content (e.g., a 'parameters' block).

    Returns:
        The root `ComponentError` that caused the simulation to fail.
    """
    netlist_path = tmp_path / "bad_ac_component_test.yaml"
    netlist_path.write_text(f"""
circuit_name: BadAcComponentTest
{extra_yaml}
components:
{component_yaml_str}
ports:
  - {{id: p1, reference_impedance: "50 ohm"}}
""")

    parser = NetlistParser()
    builder = CircuitBuilder()

    try:
        parsed_tree = parser.parse_to_circuit_tree(netlist_path)
        sim_circuit = builder.build_simulation_model(parsed_tree)
    except (CircuitBuildError, Exception) as e:
        pytest.fail(f"Circuit build failed unexpectedly for bad AC component test: {e}")

    freqs = np.array([1e9])

    with pytest.raises(SimulationRunError) as excinfo:
        run_sweep(sim_circuit, freqs)

    # The expected exception chain for AC failures is typically:
    #   SimulationRunError -> SingleLevelSimulationFailure -> ComponentError
    current_exc = excinfo.value
    while current_exc:
        if isinstance(current_exc, ComponentError):
            return current_exc
        current_exc = current_exc.__cause__
    
    pytest.fail(
        f"Expected to find a ComponentError in the exception chain, but it was not present. "
        f"Top-level exception was: {excinfo.value!r}"
    )


def build_and_run_bad_dc_component_test(component_yaml_str: str, tmp_path):
    """
    A targeted test harness for components designed to fail during DC analysis.

    This function does NOT run a full simulation sweep. Instead, it directly
    invokes the `DCAnalyzer` service on a built circuit. This provides a faster,
    more focused test for verifying the `IDcContributor` contract gateway. It
    expects a `DCAnalysisError` to be raised.

    Args:
        component_yaml_str: A string for the 'components' block of the YAML netlist.
        tmp_path: The pytest `tmp_path` fixture.

    Returns:
        The root `ComponentError` that caused the DC analysis to fail.
    """
    netlist_path = tmp_path / "bad_dc_component_test.yaml"
    netlist_path.write_text(f"""
circuit_name: BadDcComponentTest
components:
{component_yaml_str}
""")
    parser = NetlistParser()
    builder = CircuitBuilder()

    try:
        parsed_tree = parser.parse_to_circuit_tree(netlist_path)
        sim_circuit = builder.build_simulation_model(parsed_tree)
    except (CircuitBuildError, Exception) as e:
        pytest.fail(f"Circuit build failed unexpectedly for bad DC component test: {e}")

    dc_analyzer = DCAnalyzer(sim_circuit, cache=SimulationCache())

    with pytest.raises(DCAnalysisError) as excinfo:
        dc_analyzer.analyze()

    # ### REVISED ###
    # The assertion logic is corrected to handle the exception chain robustly, as
    # mandated by Critique 2 of the review. The expected chain is:
    #   DCAnalysisError -> ComponentError.
    # The `ComponentError` is the direct `__cause__` of the `DCAnalysisError`.
    cause = excinfo.value.__cause__
    assert isinstance(cause, ComponentError), \
        f"Expected DCAnalysisError to wrap a ComponentError, but got {type(cause).__name__}"

    return cause


# =============================================================================
# == Malicious Component Definitions
# =============================================================================
# These components are self-registering via the `@register_component` decorator.
# They are loaded automatically when pytest discovers this conftest.py file.
# They exist solely to provide malformed data to the framework's validation
# gateways.

# --- Components that Violate the IMnaContributor Contract (for AC tests) ---

@register_component("BadComponentReturnsNumber")
class BadComponentReturnsNumber(ComponentBase):
    @provides(IMnaContributor)
    class MnaContributor:
        def get_mna_stamps(self, c, f, p):
            # VIOLATION: Returns a raw numpy array, not a pint.Quantity.
            return (np.ones((len(f), 2, 2)), ['p1', 'p2'])
    @classmethod
    def declare_parameters(cls) -> Dict[str, str]: return {}
    @classmethod
    def declare_ports(cls) -> List[str]: return ['p1', 'p2']

@register_component("BadComponentReturnsWrongDimension")
class BadComponentReturnsWrongDimension(ComponentBase):
    @provides(IMnaContributor)
    class MnaContributor:
        def get_mna_stamps(self, c, f, p):
            # VIOLATION: Returns a Quantity with [impedance], not [admittance].
            stamp = np.ones((len(f), 2, 2))
            return (Quantity(stamp, ureg.ohm), ['p1', 'p2'])
    @classmethod
    def declare_parameters(cls) -> Dict[str, str]: return {}
    @classmethod
    def declare_ports(cls) -> List[str]: return ['p1', 'p2']

@register_component("BadComponentReturnsWrongShape")
class BadComponentReturnsWrongShape(ComponentBase):
    @provides(IMnaContributor)
    class MnaContributor:
        def get_mna_stamps(self, c, f, p):
            # VIOLATION: Returns a non-vectorized 2D array, not a 3D array (freqs, N, N).
            return (Quantity(np.ones((2, 2)), ureg.siemens), ['p1', 'p2'])
    @classmethod
    def declare_parameters(cls) -> Dict[str, str]: return {}
    @classmethod
    def declare_ports(cls) -> List[str]: return ['p1', 'p2']

@register_component("BadComponentRaisesRandomError")
class BadComponentRaisesRandomError(ComponentBase):
    @provides(IMnaContributor)
    class MnaContributor:
        def get_mna_stamps(self, c, f, p):
            # VIOLATION: Raises an unexpected, non-diagnosable error.
            raise ValueError("Something went wrong inside the component!")
    @classmethod
    def declare_parameters(cls) -> Dict[str, str]: return {}
    @classmethod
    def declare_ports(cls) -> List[str]: return ['p1', 'p2']

@register_component("BadComponentMismatchedPorts")
class BadComponentMismatchedPorts(ComponentBase):
    @provides(IMnaContributor)
    class MnaContributor:
        def get_mna_stamps(self, c, f, p):
            # VIOLATION: Returns a port name 'p_wrong' that is not in declare_ports.
            stamp = np.zeros((len(f), 2, 2), dtype=np.complex128)
            return (Quantity(stamp, ureg.siemens), ['p1', 'p_wrong']) 
    @classmethod
    def declare_parameters(cls) -> Dict[str, str]: return {}
    @classmethod
    def declare_ports(cls) -> List[str]: return ['p1', 'p2']


# --- Components that Violate the IDcContributor Contract (for DC tests) ---

@register_component("BadDcReturnsWrongPayloadType")
class BadDcReturnsWrongPayloadType(ComponentBase):
    @provides(IDcContributor)
    class DcContributor:
        def get_dc_behavior(self, component, all_dc_params):
            # VIOLATION: Returns a raw string for the payload.
            return (DCBehaviorType.ADMITTANCE, "not_a_quantity_or_tuple")
    @classmethod
    def declare_parameters(cls) -> Dict[str, str]: return {}
    @classmethod
    def declare_ports(cls) -> List[str]: return ['p1', 'p2']

@register_component("BadDcReturnsWrongScalarDimension")
class BadDcReturnsWrongScalarDimension(ComponentBase):
    @provides(IDcContributor)
    class DcContributor:
        def get_dc_behavior(self, component, all_dc_params):
            # VIOLATION: Returns a scalar Quantity with [voltage] dimension.
            return (DCBehaviorType.ADMITTANCE, Quantity(1.0, ureg.volt))
    @classmethod
    def declare_parameters(cls) -> Dict[str, str]: return {}
    @classmethod
    def declare_ports(cls) -> List[str]: return ['p1', 'p2']

@register_component("BadDcReturnsMalformedNPortTuple")
class BadDcReturnsMalformedNPortTuple(ComponentBase):
    @provides(IDcContributor)
    class DcContributor:
        def get_dc_behavior(self, component, all_dc_params):
            # VIOLATION: For N-port admittance, returns a tuple of length 1, not 2.
            return (DCBehaviorType.ADMITTANCE, (Quantity(np.ones((1,1)), ureg.siemens),))
    @classmethod
    def declare_parameters(cls) -> Dict[str, str]: return {}
    @classmethod
    def declare_ports(cls) -> List[str]: return ['p1']

@register_component("BadDcReturnsWrongScalarShape")
class BadDcReturnsWrongScalarShape(ComponentBase):
    @provides(IDcContributor)
    class DcContributor:
        def get_dc_behavior(self, component, all_dc_params):
            # VIOLATION: Returns a vector Quantity, not a scalar (ndim=0), for simple admittance.
            return (DCBehaviorType.ADMITTANCE, Quantity(np.array([1.0, 2.0]), ureg.siemens))
    @classmethod
    def declare_parameters(cls) -> Dict[str, str]: return {}
    @classmethod
    def declare_ports(cls) -> List[str]: return ['p1', 'p2']
    
@register_component("BadDcReturnsNPortTupleWithBadQuantity")
class BadDcReturnsNPortTupleWithBadQuantity(ComponentBase):
    @provides(IDcContributor)
    class DcContributor:
        def get_dc_behavior(self, component, all_dc_params):
            # VIOLATION: The first element of the N-port tuple is a raw ndarray, not a Quantity.
            return (DCBehaviorType.ADMITTANCE, (np.ones((2,2)), ['p1', 'p2']))
    @classmethod
    def declare_parameters(cls) -> Dict[str, str]: return {}
    @classmethod
    def declare_ports(cls) -> List[str]: return ['p1', 'p2']

@register_component("BadDcReturnsNPortTupleWithBadPortList")
class BadDcReturnsNPortTupleWithBadPortList(ComponentBase):
    @provides(IDcContributor)
    class DcContributor:
        def get_dc_behavior(self, component, all_dc_params):
            # VIOLATION: The second element of the N-port tuple is a dict, not a list of strings.
            return (DCBehaviorType.ADMITTANCE, (Quantity(np.ones((2,2)), ureg.siemens), {'p1': 0}))
    @classmethod
    def declare_parameters(cls) -> Dict[str, str]: return {}
    @classmethod
    def declare_ports(cls) -> List[str]: return ['p1', 'p2']
    
@register_component("BadDcReturnsNPortTupleWithWrongDimension")
class BadDcReturnsNPortTupleWithWrongDimension(ComponentBase):
    @provides(IDcContributor)
    class DcContributor:
        def get_dc_behavior(self, component, all_dc_params):
            # VIOLATION: The N-port admittance matrix has the wrong physical dimension.
            y_matrix = np.eye(2)
            return (DCBehaviorType.ADMITTANCE, (Quantity(y_matrix, ureg.ohm), ['p1', 'p2']))
    @classmethod
    def declare_parameters(cls) -> Dict[str, str]: return {}
    @classmethod
    def declare_ports(cls) -> List[str]: return ['p1', 'p2']

@register_component("BadDcReturnsNPortWithWrongShape")
class BadDcReturnsNPortWithWrongShape(ComponentBase):
    @provides(IDcContributor)
    class DcContributor:
        def get_dc_behavior(self, component, all_dc_params):
            # VIOLATION: The magnitude is a 1D vector, not a 2D matrix.
            return (DCBehaviorType.ADMITTANCE, (Quantity(np.ones(2), ureg.siemens), ['p1', 'p2']))
    @classmethod
    def declare_parameters(cls) -> Dict[str, str]: return {}
    @classmethod
    def declare_ports(cls) -> List[str]: return ['p1', 'p2']

# =============================================================================
# == Helper Components for Inheritance Testing
# =============================================================================

class TwoPortBase(ComponentBase):
    """A base class for testing capability inheritance."""
    # This capability should be inherited by children.
    @provides(IDcContributor)
    class DcContributor:
        def get_dc_behavior(self, c, p): return (DCBehaviorType.OPEN_CIRCUIT, None)

    # This capability is also inherited.
    @provides(ITopologyContributor)
    class TopologyContributor:
        def is_structurally_open(self, c, p): return False

    @classmethod
    def declare_parameters(cls) -> Dict[str, str]: return {}
    @classmethod
    def declare_ports(cls) -> List[str]: return ['p1', 'p2']


@register_component("InheritingComponent")
class InheritingComponent(TwoPortBase):
    """A component that inherits capabilities from TwoPortBase."""
    # This component provides its own MNA capability...
    @provides(IMnaContributor)
    class MnaContributor:
        def get_mna_stamps(self, c, f, p):
            return (Quantity(np.zeros((len(f), 2, 2)), ureg.siemens), ['p1', 'p2'])
    # ... and inherits its DC and Topology capabilities from TwoPortBase.

@register_component("ThreePortTee")
class ThreePortTee(ComponentBase):
    # This component models a simple T-junction where p3 is the common node.
    @provides(IConnectivityProvider)
    class ConnectivityProvider:
        def get_connectivity(self, component):
            return [('p1', 'p3'), ('p2', 'p3')] # Explicit connectivity

    # Other required methods...
    @classmethod
    def declare_parameters(cls): return {}
    @classmethod
    def declare_ports(cls): return ['p1', 'p2', 'p3']

@register_component("BadThreePortDefaultConnectivity")
class BadThreePortDefaultConnectivity(ComponentBase):
    # VIOLATION: Has 3 ports but does NOT provide a custom IConnectivityProvider.
    # It will inherit the default one from ComponentBase.
    @classmethod
    def declare_parameters(cls): return {}
    @classmethod
    def declare_ports(cls): return ['p1', 'p2', 'p3']

# =============================================================================
# == Pytest Fixtures
# =============================================================================

@pytest.fixture
def run_bad_ac_test_harness(tmp_path):
    """Provides the test harness for AC simulation failures to test functions."""
    return lambda yaml_str, extra_yaml="": build_and_run_bad_ac_component_test(yaml_str, tmp_path, extra_yaml)

@pytest.fixture
def run_bad_dc_test_harness(tmp_path):
    """Provides the test harness for DC analysis failures to test functions."""
    return lambda yaml_str: build_and_run_bad_dc_component_test(yaml_str, tmp_path)