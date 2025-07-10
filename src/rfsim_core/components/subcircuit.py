# src/rfsim_core/components/subcircuit.py

"""
Provides the concrete implementation for the hierarchical `SubcircuitInstance` component.

**Architectural Refactoring (Definitive):**

This module has been fully refactored and hardened to align with the project's definitive
architectural mandates. The changes are as follows:

1.  **Elimination of Fragile Asserts:** All `assert` statements used for runtime state
    validation have been **REPLACED** with explicit `if/raise ComponentError` blocks. This
    is a non-negotiable architectural requirement. It ensures that critical integrity checks
    (e.g., verifying that a cache is populated) are not discardable by interpreter flags
    (like `-O`) and that any failure generates a rich, actionable diagnostic report via
    the `DiagnosableError` system, not a raw `AssertionError`.

2.  **Cohesion of Implementation:** All logic related to a specific analysis domain
    (e.g., MNA, DC) is encapsulated within a dedicated, stateless, nested class
    inside its parent component class (e.g., `Resistor.MnaContributor`). This makes
    the component's total functionality self-contained and highly discoverable.

3.  **Declarative Registration:** The `@provides` decorator is used on these
    nested classes to declaratively register them as implementers of a specific
    capability protocol (e.g., `IMnaContributor`). This automates discovery and
    improves the developer experience for plugin authors.

4.  **Explicit Contracts:** The component now uses and relies on the formal, type-safe
    `DCAnalysisResults` data contract for its DC cache, eliminating the use of fragile,
    untyped dictionaries for this purpose.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional, TYPE_CHECKING

import numpy as np

# --- Foundational Imports ---
from ..units import ureg, Quantity
from ..parameters import ParameterManager
from ..data_structures import Circuit
from ..parser.raw_data import ParsedSubcircuitData

# --- Core Component Model Imports (Refactored) ---
from .base import (
    ComponentBase,
    register_component,
    StampInfo,
)
from .exceptions import ComponentError
from .base_enums import DCBehaviorType

# --- Capability System Imports (The Heart of the New Architecture) ---
from .capabilities import IMnaContributor, IDcContributor, provides

# --- Type Imports for Explicit Contracts ---
# Use TYPE_CHECKING to import DCAnalysisResults only for type analysis,
# preventing a circular import at runtime with analysis.results.
if TYPE_CHECKING:
    # This import points to the new, formal result object in the 'analysis' package.
    from ..analysis.results import DCAnalysisResults


logger = logging.getLogger(__name__)


@register_component("Subcircuit")
class SubcircuitInstance(ComponentBase):
    """
    Represents a hierarchical subcircuit instance within a larger circuit.
    
    Acts as a proxy for a nested `Circuit` object, presenting pre-computed, cached
    results to the parent circuit's analysis engines via its capability implementations.
    """

    # --- Capability Implementations are Nested Here ---

    @provides(IMnaContributor)
    class MnaContributor:
        """
        Provides the pre-computed, cached MNA stamp for the subcircuit.
        """

        def get_mna_stamps(
            self,
            component: "SubcircuitInstance",
            freq_hz_array: np.ndarray,
            # This argument is part of the IMnaContributor protocol signature
            # but is intentionally ignored here because a SubcircuitInstance's
            # behavior is fully determined by its pre-computed cache.
            all_evaluated_params: Dict[str, Quantity],
        ) -> List[StampInfo]:
            """
            Returns the N-port Y-matrix stamp from the pre-computed cache.
            """
            # --- MANDATORY REMEDIATION: Replace fragile asserts with robust, diagnosable checks. ---
            if component.cached_y_parameters_ac is None:
                raise ComponentError(
                    component_fqn=component.fqn,
                    details=(
                        "FATAL: AC Y-parameters cache was not populated before stamping. "
                        "This indicates a critical failure in the simulation executive."
                    )
                )

            num_freqs_cache, _, _ = component.cached_y_parameters_ac.shape
            num_freqs_sweep = len(freq_hz_array)

            if num_freqs_cache != num_freqs_sweep:
                raise ComponentError(
                    component_fqn=component.fqn,
                    details=(
                        f"FATAL: Mismatched frequency count. Cache contains {num_freqs_cache} points, "
                        f"but current sweep has {num_freqs_sweep} points. This indicates a critical "
                        "failure in the simulation executive or caching logic."
                    )
                )
            # --- END OF REMEDIATION ---

            admittance_matrix_qty = Quantity(
                component.cached_y_parameters_ac, component.ureg.siemens
            )
            return [(admittance_matrix_qty, component.sub_circuit_external_port_names_ordered)]

    @provides(IDcContributor)
    class DcContributor:
        """
        Provides the pre-computed, cached DC behavior for the subcircuit.
        """

        def get_dc_behavior(
            self,
            component: "SubcircuitInstance",
            # This argument is part of the IDcContributor protocol but is ignored here.
            all_dc_params: Dict[str, Quantity],
        ) -> Tuple[DCBehaviorType, Optional[Quantity]]:
            """
            Returns the DC behavior of the subcircuit based on its cached results.

            This method's behavior is determined entirely by its pre-computed DC analysis.
            It will **NEVER** return `DCBehaviorType.SHORT_CIRCUIT`. A subcircuit that behaves
            as a DC short across some or all of its ports is correctly and completely
            represented by its N-port DC admittance matrix. Therefore, this method will
            always return `DCBehaviorType.ADMITTANCE` when a valid DC Y-matrix is available.
            This explicit contract is critical for the correctness of the `DCAnalyzer`.

            The returned DC Y-matrix MAY BE SINGULAR. It is the explicit responsibility
            of the consuming `DCAnalyzer` to handle this possibility robustly.
            """
            # --- MANDATORY REMEDIATION: Replace fragile assert with robust, diagnosable check. ---
            if component.cached_dc_analysis_results is None:
                raise ComponentError(
                    component_fqn=component.fqn,
                    details=(
                        "FATAL: DC analysis results cache was not populated before its DC behavior was requested. "
                        "This indicates a critical failure in the simulation executive."
                    )
                )
            # --- END OF REMEDIATION ---

            # This uses the new, explicit dataclass contract.
            y_ports_dc_qty = component.cached_dc_analysis_results.y_ports_dc

            if isinstance(y_ports_dc_qty, Quantity) and y_ports_dc_qty.check(
                component.ureg.siemens
            ):
                return (DCBehaviorType.ADMITTANCE, y_ports_dc_qty)

            return (DCBehaviorType.OPEN_CIRCUIT, None)

    # --- Component's Own Declarations and __init__ Follow ---

    def __init__(
        self,
        instance_id: str,
        parameter_manager: ParameterManager,
        sub_circuit_object_ref: Circuit,
        sub_circuit_external_port_names_ordered: List[str],
        parent_hierarchical_id: str,
        raw_ir_data: ParsedSubcircuitData,
    ):
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

        self.cached_y_parameters_ac: Optional[np.ndarray] = None
        
        # This attribute now holds a dedicated dataclass, not a raw dictionary,
        # creating a robust, type-safe contract with the simulation executive.
        self.cached_dc_analysis_results: Optional["DCAnalysisResults"] = None

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

    def is_structurally_open(
        self, resolved_constant_params: Dict[str, Quantity]
    ) -> bool:
        """
        A subcircuit instance itself is never a simple structural open.
        Its internal topology is handled by a recursive analysis.
        """
        return False