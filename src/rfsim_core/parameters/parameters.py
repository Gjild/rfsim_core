# src/rfsim_core/parameters/parameters.py

"""
Manages all circuit parameters, their dependencies, and their evaluation with
rigorous, dimensionally-aware arithmetic.

**Architectural Overview (Post-Phase 10 Refactoring):**

This module represents the definitive architecture for parameter handling in RFSim Core.
Its design has been refactored to a single, unified pipeline that is intrinsically
unit-safe and predictable. It is governed by the following non-negotiable principles:

1.  **A Single, Consistent Language:** All parameter expressions are interpreted as
    standard Python code. Any literal with units MUST be explicitly constructed
    using the provided `Quantity` function (e.g., "Quantity('10 nH')"). This
    eliminates all syntactic ambiguity of the previous "Two-Language" architecture.

2.  **Intrinsic Safety via `eval()`:** The core of the system is Python's `eval()`
    function, operating within a carefully constructed, unit-aware scope populated
    with `pint.Quantity` objects. This delegates all dimensional analysis and
    arithmetic to the `pint` library itself, making silently incorrect,
    dimensionally-incompatible calculations impossible by construction.

3.  **Build-Time Validation via Graph-Based Dependency Resolution:**
    The `ParameterManager.build()` process now uses a correct and robust two-stage
    approach for constant parameters:
    a) First, it builds a dependency graph of all constant-valued parameters by
       statically analyzing their expression strings.
    b) Second, it topologically sorts this graph to get a guaranteed-correct
       evaluation order, then evaluates each constant in sequence.
    This provides immediate, build-time validation for a large class of user inputs
    and pre-computes results for a significant performance gain, while correctly
    handling all valid dependency chains.
"""

import logging
import re
from typing import Dict, Any, Set, Optional, List, Callable, Tuple, ChainMap
from dataclasses import dataclass
from pathlib import Path

import pint
import networkx as nx
import numpy as np

from ..units import ureg, Quantity
from .exceptions import (
    ParameterError,
    ParameterDefinitionError,
    ParameterScopeError,
    ParameterSyntaxError,
    CircularParameterDependencyError,
    ParameterEvaluationError
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParameterDefinition:
    """The explicit contract object representing a single parameter's definition."""
    owner_fqn: str
    base_name: str
    raw_value_or_expression_str: str
    source_yaml_path: Path
    declared_dimension_str: str
    is_sweepable: bool = False

    @property
    def fqn(self) -> str:
        """The canonical, fully qualified name (FQN) of the parameter."""
        return f"{self.owner_fqn}.{self.base_name}"


class ParameterManager:
    """
    Manages all circuit parameters, their dependencies, and their evaluation with
    rigorous, dimensionally-aware arithmetic.
    """
    GLOBAL_SCOPE_PREFIX = "_rfsim_global_"
    RESERVED_KEYWORDS = {'freq'}

    _EVAL_GLOBALS = {
        "ureg": ureg,
        "Quantity": Quantity,
        "np": np,
        "pi": np.pi,
    }

    _IDENTIFIER_REGEX = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')

    def __init__(self):
        self._ureg = ureg
        self._parameter_context_map: Dict[str, Dict[str, Any]] = {}
        self._dependency_graph = nx.DiGraph()
        self._build_complete = False
        self._parsed_constants: Dict[str, Quantity] = {}
        self._evaluation_order: List[str] = []
        self._scope_maps: Dict[str, ChainMap] = {}
        logger.info("ParameterManager initialized (empty).")

    def build(self, all_definitions: List[ParameterDefinition], scope_maps: Dict[str, ChainMap]):
        if self._build_complete:
            logger.warning("ParameterManager.build() called again. Rebuilding.")
            self._clear_build_state()
        if not all_definitions:
            self._build_complete = True
            return

        self._scope_maps = scope_maps
        try:
            self._create_context_map_from_definitions(all_definitions)
            constant_candidates = {
                fqn for fqn, ctx in self._parameter_context_map.items()
                if "freq" not in ctx['definition'].raw_value_or_expression_str
            }
            self._build_constant_dependency_graph(constant_candidates)
            self._check_circular_dependencies()
            self._evaluate_and_cache_constants()
            self._compute_and_cache_constant_flags()
            self._compute_evaluation_order()
        except ParameterError as e:
            self._clear_build_state()
            raise
        except Exception as e:
            self._clear_build_state()
            raise ParameterError(f"Unexpected error during build: {e}") from e

        self._build_complete = True
        logger.info(f"ParameterManager build complete. Defined parameters: {len(self._parameter_context_map)}.")

    def evaluate_all(self, freq_hz: np.ndarray) -> Dict[str, Quantity]:
        self._check_build_complete()
        results: Dict[str, Quantity] = {}
        for fqn, const_qty in self._parsed_constants.items():
            mag = const_qty.magnitude
            if not isinstance(mag, np.ndarray) or mag.shape != freq_hz.shape:
                broadcasted_mag = np.full_like(freq_hz, mag, dtype=np.result_type(mag, float))
                results[fqn] = Quantity(broadcasted_mag, const_qty.units)
            else:
                results[fqn] = const_qty

        dynamic_fqns = [fqn for fqn in self._evaluation_order if fqn not in self._parsed_constants]

        base_eval_scope = results.copy()
        base_eval_scope['freq'] = Quantity(freq_hz, self._ureg.hertz)
        base_eval_scope.update(self._EVAL_GLOBALS)

        for fqn in dynamic_fqns:
            if fqn == 'freq': continue
            
            definition = self._parameter_context_map[fqn]['definition']
            
            param_lexical_scope = self._scope_maps.get(fqn)
            eval_locals = base_eval_scope.copy()
            if param_lexical_scope:
                for base_name, resolved_fqn in param_lexical_scope.items():
                    if resolved_fqn in base_eval_scope:
                        eval_locals[base_name] = base_eval_scope[resolved_fqn]
            
            try:
                result_val = eval(definition.raw_value_or_expression_str, self._EVAL_GLOBALS, eval_locals)
                
                if not isinstance(result_val, Quantity):
                    result_val = Quantity(result_val, definition.declared_dimension_str)
                if not isinstance(result_val.magnitude, np.ndarray):
                    result_val = Quantity(np.full_like(freq_hz, result_val.magnitude), result_val.units)

                with np.errstate(invalid='ignore', divide='ignore'):
                    numerical_errors_mask = np.isnan(result_val.magnitude) | np.isinf(result_val.magnitude)

                if np.any(numerical_errors_mask):
                    error_indices = np.argwhere(numerical_errors_mask).flatten()
                    fail_val = result_val.magnitude[error_indices[0]]
                    details = f"Expression resulted in a non-finite value ({fail_val})."
                    raise ParameterEvaluationError(fqn=fqn, details=details, error_indices=error_indices, frequencies=freq_hz, input_values=eval_locals)

                # --- START OF CORRECTED DIMENSIONALITY LOGIC (DYNAMIC EVAL) ---
                if definition.declared_dimension_str == "dimensionless":
                    if not result_val.dimensionless:
                        raise pint.DimensionalityError(
                            result_val.units, "dimensionless",
                            extra_msg=f" for parameter '{fqn}' which was declared dimensionless."
                        )
                    final_qty = result_val
                else:
                    # Parameter was declared with a dimension.
                    expected_unit = self._ureg.Unit(definition.declared_dimension_str)
                    if not result_val.is_compatible_with(expected_unit):
                        raise pint.DimensionalityError(
                            result_val.units, expected_unit,
                            extra_msg=f" for expression '{definition.raw_value_or_expression_str}'"
                        )
                    # Always convert to the declared dimension for consistency.
                    final_qty = result_val.to(expected_unit)
                # --- END OF CORRECTED DIMENSIONALITY LOGIC ---
                
                results[fqn] = final_qty
                base_eval_scope[fqn] = final_qty

            except Exception as e:
                if isinstance(e, ParameterEvaluationError): raise
                if isinstance(e, ParameterError): raise
                raise ParameterEvaluationError(fqn=fqn, details=str(e), error_indices=None, frequencies=freq_hz, input_values=eval_locals) from e

        return results

    def _clear_build_state(self):
        self._parameter_context_map = {}
        self._dependency_graph.clear()
        self._parsed_constants = {}
        self._evaluation_order = []
        self._scope_maps = {}
        self._build_complete = False

    def _create_context_map_from_definitions(self, all_definitions: List[ParameterDefinition]):
        for definition in all_definitions:
            fqn = definition.fqn
            if fqn in self._parameter_context_map:
                raise ParameterDefinitionError(fqn=fqn, user_input=definition.raw_value_or_expression_str, source_yaml_path=definition.source_yaml_path, details=f"Duplicate parameter FQN '{fqn}' detected.")
            self._parameter_context_map[fqn] = {'definition': definition, 'dependencies': set()}

    def _build_constant_dependency_graph(self, constant_candidates: Set[str]):
        logger.debug("Building dependency graph for %d constant candidates...", len(constant_candidates))
        self._dependency_graph.add_nodes_from(constant_candidates)

        for fqn in constant_candidates:
            definition = self._parameter_context_map[fqn]['definition']
            expression = definition.raw_value_or_expression_str
            scope = self._scope_maps.get(fqn)
            if not scope: continue

            potential_deps = self._IDENTIFIER_REGEX.findall(expression)
            for dep_base_name in potential_deps:
                if dep_base_name in self._EVAL_GLOBALS or dep_base_name in self.RESERVED_KEYWORDS:
                    continue
                if dep_base_name in scope:
                    resolved_dep_fqn = scope[dep_base_name]
                    if resolved_dep_fqn in constant_candidates:
                        self._dependency_graph.add_edge(resolved_dep_fqn, fqn)

    def _check_circular_dependencies(self):
        try:
            cycles = list(nx.simple_cycles(self._dependency_graph))
            if cycles:
                raise CircularParameterDependencyError(cycle=cycles[0])
        except nx.NetworkXError as e:
            raise ParameterError(f"NetworkX error during cycle check: {e}") from e

    def _evaluate_and_cache_constants(self):
        try:
            constant_evaluation_order = list(nx.topological_sort(self._dependency_graph))
        except nx.NetworkXUnfeasible:
            raise CircularParameterDependencyError(cycle=[])

        logger.debug("Eagerly evaluating %d constants in dependency order...", len(constant_evaluation_order))
        self._parsed_constants = {}
        base_eval_scope = self._EVAL_GLOBALS.copy()

        for fqn in constant_evaluation_order:
            definition = self._parameter_context_map[fqn]['definition']
            
            param_lexical_scope = self._scope_maps.get(fqn)
            eval_locals = base_eval_scope.copy()
            if param_lexical_scope:
                for base_name, resolved_fqn in param_lexical_scope.items():
                    if resolved_fqn in self._parsed_constants:
                        eval_locals[base_name] = self._parsed_constants[resolved_fqn]

            try:
                result_val = eval(definition.raw_value_or_expression_str, self._EVAL_GLOBALS, eval_locals)

                if not isinstance(result_val, Quantity):
                    result_val = Quantity(result_val, definition.declared_dimension_str)

                # --- START OF FIX #2: NON-NEGOTIABLE CHECK FOR NON-FINITE CONSTANTS ---
                mag = result_val.magnitude
                # This check is robust for scalars (float, int, np.number) and numpy arrays.
                if isinstance(mag, (float, int, np.number, np.ndarray)):
                    # Suppress "invalid value encountered in isfinite" warnings from the check itself
                    with np.errstate(invalid='ignore'):
                        if not np.all(np.isfinite(mag)):
                            # Find the first non-finite value for a clear error message.
                            fail_val = mag
                            if isinstance(mag, np.ndarray):
                                non_finite_mask = ~np.isfinite(mag)
                                fail_val = mag[non_finite_mask][0] if np.any(non_finite_mask) else mag
                            
                            details = f"Constant expression resulted in a non-finite value ({fail_val})."
                            # This is an unrecoverable build error.
                            raise ValueError(details)
                # --- END OF FIX #2 ---

                # --- START OF FIX #1: CORRECT DIMENSIONALITY LOGIC (CONSTANT EVAL) ---
                if definition.declared_dimension_str == "dimensionless":
                    if not result_val.dimensionless:
                        raise pint.DimensionalityError(
                            result_val.units, "dimensionless",
                            extra_msg=f" for parameter '{fqn}' which was declared dimensionless."
                        )
                    final_qty = result_val
                else:
                    # Parameter was declared with a dimension.
                    expected_unit = self._ureg.Unit(definition.declared_dimension_str)
                    if not result_val.is_compatible_with(expected_unit):
                        raise pint.DimensionalityError(
                            result_val.units, expected_unit,
                            extra_msg=f" for expression '{definition.raw_value_or_expression_str}'"
                        )
                    # Always convert to the declared dimension for consistency.
                    final_qty = result_val.to(expected_unit)
                # --- END OF FIX #1 ---

                self._parsed_constants[fqn] = final_qty

            except Exception as e:
                if isinstance(e, ParameterError): raise
                raise ParameterEvaluationError(fqn=fqn, details=str(e), error_indices=None, frequencies=None, input_values=eval_locals) from e

    def _compute_and_cache_constant_flags(self):
        logger.debug("Computing and caching is_constant flags for all parameters...")
        for fqn, context_info in self._parameter_context_map.items():
            context_info['is_constant'] = fqn in self._parsed_constants

    def _compute_evaluation_order(self):
        constant_order = list(nx.topological_sort(self._dependency_graph))
        all_fqns = set(self._parameter_context_map.keys())
        dynamic_params = sorted(list(all_fqns - set(constant_order) - {'freq'}))
        self._evaluation_order = constant_order + dynamic_params

    def _check_build_complete(self):
        if not self._build_complete:
            raise ParameterError("ParameterManager has not been built. Call the 'build()' method first.")

    def get_all_fqn_definitions(self) -> List[ParameterDefinition]:
        self._check_build_complete()
        return [ctx['definition'] for ctx in self._parameter_context_map.values()]

    def get_parameter_definition(self, fqn: str) -> ParameterDefinition:
        self._check_build_complete()
        try:
            return self._parameter_context_map[fqn]['definition']
        except KeyError:
            raise ParameterScopeError(owner_fqn="<lookup>", unresolved_symbol=fqn, user_input="", source_yaml_path=Path(), resolution_path_details=f"FQN '{fqn}' not found in parameter map.")

    def is_constant(self, fqn: str) -> bool:
        self._check_build_complete()
        if fqn == 'freq': return False
        try:
            return self._parameter_context_map[fqn]['is_constant']
        except KeyError:
            raise ParameterError(f"FQN '{fqn}' not found.")

    def get_constant_value(self, fqn: str) -> Quantity:
        self._check_build_complete()
        if not self.is_constant(fqn):
            raise ParameterError(f"Parameter '{fqn}' is not a constant value (it depends on 'freq').")
        try:
            return self._parsed_constants[fqn]
        except KeyError:
            raise ParameterError(f"Internal Error: Constant '{fqn}' was not found in the pre-evaluation cache.") from None