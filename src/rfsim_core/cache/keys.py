# src/rfsim_core/cache/keys.py
"""
Centralizes the logic for generating robust, correct, and consistent cache keys.

This module fulfills a critical architectural role by abstracting away the complex
details of what makes a simulation result unique and therefore cacheable. By
centralizing this logic, we prevent subtle bugs that would arise from different parts
of the system creating slightly different keys for the same conceptual result.

Each function here implements a non-negotiable algorithm for its key, ensuring that
all relevant inputs that could change the result are captured.
"""
import numpy as np
from typing import Tuple

from ..components.subcircuit import SubcircuitInstance
from ..data_structures import Circuit
from ..parameters import ParameterManager, ParameterError


def create_subcircuit_sim_key(
    sub_inst: SubcircuitInstance, freq_array_hz: np.ndarray, global_pm: ParameterManager
) -> Tuple:
    """
    Creates the definitive cache key for a single-level subcircuit simulation result.
    This logic is MOVED from Phase 9's `simulation/execution.py` and is now centralized.
    The key is sensitive to the subcircuit's definition, its local parameter overrides,
    any external parameters it depends on, and the frequency sweep.
    """
    # The key is a tuple, making it hashable. It starts with a namespace string
    # to prevent collisions between different types of cached results.
    key_namespace = "subcircuit_sim"

    # 1. The absolute path to the subcircuit's definition file.
    def_path_str = str(sub_inst.sub_circuit_object.source_file_path)

    # 2. A canonical representation of the parameter overrides from the instance's YAML.
    # Sorting ensures that the key is identical regardless of the order in the YAML file.
    overrides = sub_inst.raw_parameter_overrides
    canonical_overrides = tuple(sorted(
        (k, str(sorted(v.items()) if isinstance(v, dict) else v))
        for k, v in overrides.items()
    ))

    # 3. A canonical representation of all external parameters the subcircuit depends on.
    fqns_in_sub = {p.fqn for p in sub_inst.sub_circuit_object.parameter_manager.get_all_fqn_definitions()}
    const_ext, freq_ext = global_pm.get_external_dependencies_of_scope(fqns_in_sub)

    # For constant dependencies, the key includes their exact FQN and value.
    const_vals = tuple(sorted((fqn, f"{global_pm.get_constant_value(fqn):~P}") for fqn in const_ext))

    # For frequency-dependent dependencies, the key includes their full definition.
    freq_defs = tuple(sorted(
        (p.fqn, p.raw_value_or_expression_str, p.declared_dimension_str)
        for fqn in freq_ext if (p := global_pm.get_parameter_definition(fqn))
    ))

    ext_context = (const_vals, freq_defs)

    # 4. A canonical representation of the frequency array.
    freqs = tuple(np.sort(np.unique(freq_array_hz)))

    return (key_namespace, def_path_str, canonical_overrides, ext_context, freqs)


def create_topology_key(circuit: Circuit) -> Tuple:
    """
    Creates the definitive cache key for a topology analysis result.
    This logic is MOVED from Phase 9's `TopologyAnalyzer` and is now centralized.
    The key is sensitive to the circuit's definition path and the exact value of all
    constant parameters within its scope, as these can define structural opens (e.g., C=0).

    MANDATORY SAFETY FIX: The `try/except` block that previously existed around
    `get_constant_value` has been REMOVED. A failure to resolve a constant parameter
    during key generation MUST be a hard error, not silently ignored. This upholds the
    "Correctness by Construction" mandate by preventing the generation of an incorrect
    cache key that could lead to a false cache hit.
    """
    source_path = str(circuit.source_file_path)
    const_params = []
    for p_def in circuit.parameter_manager.get_all_fqn_definitions():
        # Check if the parameter is a constant and belongs to the current circuit's scope.
        if p_def.owner_fqn.startswith(circuit.hierarchical_id) and circuit.parameter_manager.is_constant(p_def.fqn):
            # This call will now raise an exception if it fails, which is the correct behavior.
            val = circuit.parameter_manager.get_constant_value(p_def.fqn)
            # Use Pint's pretty format `~P` for a canonical string representation.
            const_params.append((p_def.fqn, f"{val:~P}"))

    return ("topology", source_path, tuple(sorted(const_params)))


def create_dc_analysis_key(circuit: Circuit) -> Tuple:
    """
    Creates the definitive cache key for a DC analysis result.

    RIGOROUS JUSTIFICATION: A DC analysis is dependent on the precise numerical value
    of *every* constant parameter that can influence DC behavior (e.g., resistance,
    or the DC representation of any component). A key based only on which components
    are shorts/opens is dangerously insufficient. The most robust and correct cache key
    must therefore be sensitive to changes in *any* of these constant values.

    The existing `create_topology_key` function already generates a key based on a
    canonical representation of all constant parameter FQNs and their exact values
    within a circuit's scope. By leveraging this existing, robust key generation
    logic and adding a unique 'dc_analysis' namespace prefix, we ensure correctness,
    prevent key collisions with topology results, and adhere to the DRY (Don't Repeat
    Yourself) principle.
    """
    # The key is composed of a unique namespace and the result of the topology key logic.
    return ("dc_analysis",) + create_topology_key(circuit)[1:]