# src/rfsim_core/components/subcircuit.py

import logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

from .base import (
    ComponentBase,
    register_component,
    ComponentError,
    StampInfo,
    DCBehaviorType,
)
from ..units import ureg, Quantity
from ..parameters import ParameterManager
from ..data_structures import Circuit
from ..parser.raw_data import ParsedSubcircuitData

logger = logging.getLogger(__name__)


@register_component("Subcircuit")
class SubcircuitInstance(ComponentBase):
    """
    Represents a hierarchical subcircuit instance within a larger circuit.

    This component acts as a proxy for a complete, nested `Circuit` object. It does
    not have its own intrinsic physical behavior; instead, its behavior is defined
    by the pre-computed N-port Y-parameters of its underlying circuit definition.

    The simulation executive is responsible for recursively simulating the subcircuit's
    definition and populating the `cached_y_parameters_ac` and
    `cached_dc_analysis_results` attributes before this instance's stamping
    methods are called. This component's primary role is to present those cached
    results to the parent circuit's MNA assembler and DC analyzer.
    """

    def __init__(
        self,
        instance_id: str,
        parameter_manager: ParameterManager,
        sub_circuit_object_ref: Circuit,
        sub_circuit_external_port_names_ordered: List[str],
        parent_hierarchical_id: str,
        raw_ir_data: ParsedSubcircuitData,
    ):
        """
        Initializes the SubcircuitInstance.

        Args:
            instance_id: The unique ID of this subcircuit instance (e.g., 'X1').
            parameter_manager: The single, global `ParameterManager` for the entire simulation.
            sub_circuit_object_ref: A reference to the fully built `Circuit` object
                                    representing the subcircuit's definition.
            sub_circuit_external_port_names_ordered: The ordered list of the subcircuit
                                                     definition's external port names.
                                                     This order defines the N-port
                                                     Y-matrix indexing.
            parent_hierarchical_id: The fully qualified hierarchical ID of the parent
                                    circuit that contains this instance (e.g., 'top').
            raw_ir_data: The non-negotiable link to the original parsed IR data for
                         this specific instance, holding the raw port mappings and
                         parameter overrides.
        """
        super().__init__(
            instance_id=instance_id,
            parameter_manager=parameter_manager,
            parent_hierarchical_id=parent_hierarchical_id,
            raw_ir_data=raw_ir_data,
        )

        if not isinstance(raw_ir_data, ParsedSubcircuitData):
            raise TypeError(
                f"SubcircuitInstance '{instance_id}' must receive a "
                "ParsedSubcircuitData object for its raw instance data."
            )

        self.sub_circuit_object: Circuit = sub_circuit_object_ref
        self.sub_circuit_external_port_names_ordered: List[
            str
        ] = sub_circuit_external_port_names_ordered

        # These attributes are populated by the simulation executive during the
        # recursive caching pass. They are None until then.
        self.cached_y_parameters_ac: Optional[np.ndarray] = None
        self.cached_dc_analysis_results: Optional[Dict[str, Any]] = None

        logger.debug(
            f"SubcircuitInstance '{self.fqn}' initialized, referencing definition "
            f"from '{self.sub_circuit_object.source_file_path}'."
        )

    @classmethod
    def declare_parameters(cls) -> Dict[str, str]:
        """Subcircuits do not declare parameters; they are containers."""
        return {}

    @classmethod
    def declare_ports(cls) -> List[str | int]:
        """Subcircuit ports are dynamically defined by their definition file."""
        return []

    @classmethod
    def declare_connectivity(cls) -> List[Tuple[str | int, str | int]]:
        """Subcircuit connectivity is complex and handled by recursive analysis."""
        return []

    def get_mna_stamps(
        self,
        freq_hz_array: np.ndarray,
        all_evaluated_params: Dict[str, Quantity],
    ) -> List[StampInfo]:
        """
        Returns the N-port Y-matrix stamp from the pre-computed cache.

        This method fulfills the component's contract by providing its cached
        N-port admittance matrix, ready for stamping into the parent's MNA system.
        """
        assert (
            self.cached_y_parameters_ac is not None
        ), f"FATAL: Subcircuit '{self.fqn}' AC Y-parameters cache was not populated before stamping. This indicates a failure in the simulation executive."

        num_freqs, num_ports, _ = self.cached_y_parameters_ac.shape
        assert num_freqs == len(
            freq_hz_array
        ), f"Subcircuit '{self.fqn}' cache frequency count ({num_freqs}) mismatches sweep count ({len(freq_hz_array)})."
        assert num_ports == len(
            self.sub_circuit_external_port_names_ordered
        ), f"Subcircuit '{self.fqn}' cache port count mismatch."

        admittance_matrix_qty = Quantity(
            self.cached_y_parameters_ac, self._ureg.siemens
        )
        return [(admittance_matrix_qty, self.sub_circuit_external_port_names_ordered)]

    def get_dc_behavior(
        self, all_dc_params: Dict[str, Quantity]
    ) -> Tuple[DCBehaviorType, Optional[Quantity]]:
        """
        Returns the DC behavior of the subcircuit based on its cached results.

        This method's behavior is determined entirely by its pre-computed DC analysis.
        It will **NEVER** return `DCBehaviorType.SHORT_CIRCUIT`. A subcircuit that behaves
        as a DC short across some or all of its ports is correctly and completely
        represented by its N-port DC admittance matrix. Therefore, this method will
        always return `DCBehaviorType.ADMITTANCE` when a valid DC Y-matrix is available.

        This explicit contract is critical for the correctness of the `DCAnalyzer`.

        Args:
            all_dc_params: This input is ignored, as behavior is determined by the cache.

        Returns:
            A tuple containing:
            - `DCBehaviorType.ADMITTANCE` and the cached DC Y-matrix as a Quantity, if available.
            - `DCBehaviorType.OPEN_CIRCUIT` and `None`, if no DC Y-matrix is available in the cache.
        """
        assert (
            self.cached_dc_analysis_results is not None
        ), f"FATAL: Subcircuit '{self.fqn}' DC analysis results cache was not populated before its DC behavior was requested."

        y_ports_dc_qty = self.cached_dc_analysis_results.get("Y_ports_dc")

        if isinstance(y_ports_dc_qty, Quantity) and y_ports_dc_qty.check(
            self._ureg.siemens
        ):
            return (DCBehaviorType.ADMITTANCE, y_ports_dc_qty)

        if y_ports_dc_qty is not None:
            logger.error(
                f"Subcircuit '{self.fqn}': Cached 'Y_ports_dc' is invalid. "
                f"Expected Admittance Quantity, got {type(y_ports_dc_qty)}."
            )

        return (DCBehaviorType.OPEN_CIRCUIT, None)

    def is_structurally_open(
        self, resolved_constant_params: Dict[str, Quantity]
    ) -> bool:
        """
        A subcircuit instance itself is never a simple structural open.

        Its internal topology is complex and is handled by the recursive `TopologyAnalyzer`.
        This method must return `False` to prevent the `TopologyAnalyzer` from
        incorrectly pruning the entire subcircuit instance from the AC analysis graph.
        """
        return False