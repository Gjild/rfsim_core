# src/rfsim_core/components/subcircuit.py

"""
Provides the concrete implementation for the hierarchical `SubcircuitInstance` component.
"""

import logging
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
import networkx as nx

import numpy as np

# --- Foundational Imports ---
from ..units import ureg, Quantity
from ..parameters import ParameterManager
from ..data_structures import Circuit
from ..errors import FrameworkLogicError

# --- Core Component Model Imports ---
from .base import ComponentBase, register_component
from .exceptions import ComponentError
from .base_enums import DCBehaviorType

# --- Capability System Imports ---
from .capabilities import IMnaContributor, IDcContributor, IConnectivityProvider, DcAdmittancePayload, provides

# --- Type Imports for Explicit Contracts ---
# Use TYPE_CHECKING to import result objects only for type analysis,
# preventing circular imports at runtime.
if TYPE_CHECKING:
    from ..analysis.results import DCAnalysisResults, TopologyAnalysisResults
    StampInfo = Tuple[Quantity, List[str]]


logger = logging.getLogger(__name__)


@register_component("Subcircuit")
class SubcircuitInstance(ComponentBase):
    """
    Represents a hierarchical subcircuit instance within a larger circuit.

    Acts as a proxy for a nested `Circuit` object, presenting pre-computed, cached
    results to the parent circuit's analysis engines via its capability implementations.
    This component is a pure data proxy and holds no simulation logic itself.
    """

    @provides(IMnaContributor)
    class MnaContributor:
        """
        Provides the pre-computed, cached MNA stamp for the subcircuit.
        This capability trustingly assumes the simulation executive has populated the
        `cached_y_parameters_ac` attribute on the component instance.
        """
        def get_mna_stamps(
            self,
            component: "SubcircuitInstance",
            freq_hz_array: np.ndarray,
            all_evaluated_params: Dict[str, Quantity],
        ) -> "StampInfo":
            """
            Returns the N-port Y-matrix stamp from the pre-computed cache as a single tuple.
            """
            if component.cached_y_parameters_ac is None:
                raise FrameworkLogicError(
                    f"Pre-condition failed: MNA stamping was requested for subcircuit "
                    f"'{component.fqn}' before its AC simulation results were cached."
                )

            num_freqs_cache, _, _ = component.cached_y_parameters_ac.shape
            num_freqs_sweep = len(freq_hz_array)

            if num_freqs_cache != num_freqs_sweep:
                raise ComponentError(
                    component_fqn=component.fqn,
                    details=(
                        f"Mismatched frequency count. The cached data contains {num_freqs_cache} points, "
                        f"but current sweep requires {num_freqs_sweep} points. This indicates a critical "
                        "cache consistency failure in the simulation executive."
                    )
                )

            admittance_matrix_qty = Quantity(
                component.cached_y_parameters_ac, component.ureg.siemens
            )
            # MANDATED CHANGE: Return the tuple directly, not inside a list.
            return (admittance_matrix_qty, component.sub_circuit_external_port_names_ordered)

    @provides(IDcContributor)
    class DcContributor:
        """
        Provides the pre-computed, cached DC behavior for the subcircuit.
        This capability trustingly assumes the simulation executive has populated the
        `cached_dc_analysis_results` attribute on the component instance.
        """
        def get_dc_behavior(
            self,
            component: "SubcircuitInstance",
            all_dc_params: Dict[str, Quantity],
        ) -> Tuple[DCBehaviorType, Optional[DcAdmittancePayload]]:
            """
            Returns the DC behavior of the subcircuit based on its cached results.
            This implementation is now purified and component-agnostic.
            """
            if component.cached_dc_analysis_results is None:
                raise FrameworkLogicError(
                    f"Pre-condition failed: DC behavior was requested for subcircuit "
                    f"'{component.fqn}' before its DC analysis results were cached."
                )

            y_ports_dc_qty = component.cached_dc_analysis_results.y_ports_dc

            if y_ports_dc_qty is not None:
                port_order = component.cached_dc_analysis_results.dc_port_names_ordered
                return (DCBehaviorType.ADMITTANCE, (y_ports_dc_qty, port_order))
            else:
                return (DCBehaviorType.OPEN_CIRCUIT, None)

    @provides(IConnectivityProvider)
    class ConnectivityProvider:
        """
        Provides the subcircuit's effective external port-to-port connectivity.
        This capability trustingly assumes the simulation executive has populated the
        `cached_topology_results` attribute on the component instance.
        """
        def get_connectivity(self, component: "SubcircuitInstance") -> List[Tuple[str, str]]:
            """
            Provides the subcircuit's external port connectivity by reading it from
            the cached, formal TopologyAnalysisResults object. This now includes
            paths to the subcircuit's internal ground.
            """
            if component.cached_topology_results is None:
                raise FrameworkLogicError(
                    f"Pre-condition failed: Connectivity was requested for subcircuit "
                    f"'{component.fqn}' before its topology results were cached. "
                    "This indicates a logic error in the SimulationEngine's orchestration."
                )

            # The formal result object is the explicit contract.
            results = component.cached_topology_results
            sub_ac_graph = results.ac_graph
            sub_ground_name = component.sub_circuit_object.ground_net_name
            sub_ext_ports = list(component.sub_circuit_object.external_ports.keys())
            
            # This is the connectivity reported by the sub-analysis
            connectivity = results.external_port_connectivity

            # NEW LOGIC: Check which external ports have a path to the subcircuit's internal ground
            if sub_ground_name in sub_ac_graph:
                for port in sub_ext_ports:
                    if port in sub_ac_graph and nx.has_path(sub_ac_graph, port, sub_ground_name):
                        # Use the subcircuit's ground name as a special token.
                        # The parent's TopologyAnalyzer will know how to interpret this.
                        connectivity.append((port, sub_ground_name))
            
            return connectivity

    def __init__(
        self,
        instance_id: str,
        parameter_manager: ParameterManager,
        sub_circuit_object_ref: Circuit,
        sub_circuit_external_port_names_ordered: List[str],
        parent_hierarchical_id: str,
        port_net_map: Dict[str, str],
        raw_parameter_overrides: Dict,
    ):
        super().__init__(
            instance_id=instance_id,
            component_type_str="Subcircuit",
            parameter_manager=parameter_manager,
            parent_hierarchical_id=parent_hierarchical_id,
            port_net_map=port_net_map,
        )

        self.sub_circuit_object: Circuit = sub_circuit_object_ref
        self.sub_circuit_external_port_names_ordered: List[
            str
        ] = sub_circuit_external_port_names_ordered

        # These attributes are populated by the simulation executive before being used
        # by this component's capabilities. They are initialized to None.
        self.cached_y_parameters_ac: Optional[np.ndarray] = None
        self.cached_dc_analysis_results: Optional["DCAnalysisResults"] = None
        self.cached_topology_results: Optional["TopologyAnalysisResults"] = None
        self.raw_parameter_overrides = raw_parameter_overrides

        logger.debug(
            f"SubcircuitInstance '{self.fqn}' initialized, referencing definition "
            f"from '{self.sub_circuit_object.source_file_path}'."
        )

    @classmethod
    def declare_parameters(cls) -> Dict[str, str]:
        """Subcircuits do not declare parameters themselves; they are containers."""
        return {}

    @classmethod
    def declare_ports(cls) -> List[str]:
        """
        Subcircuit ports are dynamically determined by its definition file and are not
        declared statically on the component type itself. This method returns an
        empty list to satisfy the abstract base class contract.
        """
        return []