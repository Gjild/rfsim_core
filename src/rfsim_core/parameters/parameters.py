# src/rfsim_core/parameters/parameters.py

"""
Manages all circuit parameters, their dependencies, and their evaluation with
rigorous, dimensionally-aware arithmetic.

**Architectural Overview (Post-Refactoring):**

This module represents the definitive architecture for parameter handling in RFSim Core.
Its design is governed by three non-negotiable principles:

1.  **Correctness by Construction via Lifecycle Management:**
    The `ParameterManager` has a strict two-stage lifecycle: an "incomplete" build phase
    and a "complete" post-build phase. A guard, `_check_build_complete()`, ensures that
    public API methods cannot be called on an incompletely built instance, preventing
    a large class of bugs. The build process itself is self-contained and does not
    rely on its own public interface.

2.  **Computation Over Inquisition for Performance and Simplicity:**
    During the build process, expensive-to-compute state, such as whether a parameter
    is constant, is computed exactly once and cached (`_compute_and_cache_constant_flags`).
    Subsequent queries for this state are transformed into trivial, O(1) dictionary
    lookups. This is faster and architecturally cleaner than re-computing the state
    on every query.

3.  **Intrinsic Unit Safety via Pint-Native Evaluation:**
    The evaluation pipeline is engineered to be intrinsically unit-safe. The core
    evaluation is performed using Python's `eval()` on the original expression strings,
    where the evaluation scope contains full `pint.Quantity` objects. This delegates
    all arithmetic and function calls directly to the `pint` library, guaranteeing that
    any dimensionally incompatible operations (e.g., adding ohms and farads) will
    raise a `pint.DimensionalityError`, thus preventing silently incorrect numerical results.
    The use of `sympy` is now strictly limited to parsing and dependency analysis, not
    numerical evaluation.
"""

import logging
from typing import Dict, Any, Set, Optional, List, Callable, Tuple, ChainMap
from dataclasses import dataclass
from pathlib import Path

import pint
import sympy
from sympy.core.function import UndefinedFunction
from sympy import (
    Symbol, Integer, Float, Rational, Add, Mul, Pow, I, Function,
    Abs, sqrt, log, exp,
    sin, cos, tan, asin, acos, atan, atan2,
    re, im, arg, conjugate, Expr, pi, diff
)
from sympy.core.relational import Relational
from sympy.logic.boolalg import BooleanFunction
from sympy import Derivative, Integral, Lambda, Piecewise

import networkx as nx
import numpy as np

from ..units import ureg, Quantity
from .preprocessor import ExpressionPreprocessor
from .exceptions import (
    ParameterError, ParameterDefinitionError, ParameterScopeError,
    ParameterSyntaxError, CircularParameterDependencyError, ParameterEvaluationError
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

    def log10(x):
        return log(x, 10)

    # SymPy functions are used for parsing and dependency checking only.
    ALLOWED_SYMPY_FUNCTIONS = {
        Abs, sqrt, log, exp, log10,
        sin, cos, tan, asin, acos, atan, atan2,
        re, im, arg, conjugate,
    }
    ALLOWED_SYMPY_SYMBOLS = {sympy.pi, sympy.E, sympy.I}
    RESERVED_KEYWORDS = {'freq'}

    _PARSE_GLOBALS = {
        "Symbol": Symbol, "Integer": Integer, "Float": Float, "Rational": Rational,
        "Add": Add, "Mul": Mul, "Pow": Pow, "I": I, "Function": Function,
        "pi": sympy.pi, "E": sympy.E,
        **{func.__name__: func for func in ALLOWED_SYMPY_FUNCTIONS},
        "diff": diff,
    }
    
    # Define the execution scope for `eval` with pint-aware functions.
    _EVAL_GLOBALS = {
        "ureg": ureg,
        "Quantity": Quantity,
        "np": np,
        # Make numpy functions available, which Pint wraps for unit safety.
        "sqrt": np.sqrt, "log": np.log, "log10": np.log10, "exp": np.exp,
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan, "atan2": np.arctan2,
        "abs": np.abs, "re": np.real, "im": np.imag, "conjugate": np.conjugate,
        "pi": np.pi,
        # Allow direct access to units like 'ohm', 'pF', etc.
        **{unit: ureg[unit] for unit in ureg}
    }


    def __init__(self):
        self._ureg = ureg
        self._parameter_context_map: Dict[str, Dict[str, Any]] = {}
        self._dependency_graph = nx.DiGraph()
        self._build_complete = False
        self._parsed_constants: Dict[str, Quantity] = {}
        self._evaluation_order: List[str] = []
        logger.info("ParameterManager initialized (empty).")

    def build(self, all_definitions: List[ParameterDefinition], scope_maps: Dict[str, ChainMap]):
        """
        Executes the full, multi-stage build process for the ParameterManager.
        This method is NOT re-entrant. It populates the internal state and prepares
        the manager for evaluation queries.
        """
        if self._build_complete:
            logger.warning("ParameterManager.build() called again. Rebuilding.")
            self._clear_build_state()
        if not all_definitions:
            self._build_complete = True
            return

        try:
            self._create_context_map_from_definitions(all_definitions)
            self._parse_all_values_and_find_deps(scope_maps)
            self._build_dependency_graph()
            self._check_circular_dependencies()
            self._compute_and_cache_constant_flags()
            self._compute_evaluation_order()
            self._evaluate_and_cache_all_constants()
        except ParameterError as e:
            self._clear_build_state()
            raise
        except Exception as e:
            self._clear_build_state()
            raise ParameterError(f"Unexpected error during build: {e}") from e

        self._build_complete = True
        logger.info(f"ParameterManager build complete. Defined parameters: {len(self._parameter_context_map)}.")

    def evaluate_all(self, freq_hz: np.ndarray) -> Dict[str, Quantity]:
        """
        Evaluates all parameters for a given frequency array using dimensionally-aware arithmetic.
        """
        self._check_build_complete()
        results: Dict[str, Quantity] = {}
        freq_qty = Quantity(freq_hz, self._ureg.hertz)

        for fqn in self._evaluation_order:
            if fqn == 'freq':
                continue
            context_info = self._parameter_context_map[fqn]
            definition = context_info['definition']

            if context_info.get('is_constant', False):
                const_qty = self._parsed_constants[fqn]
                mag = const_qty.magnitude
                if not isinstance(mag, np.ndarray) or mag.shape != freq_hz.shape:
                    broadcasted_mag = np.full_like(freq_hz, mag, dtype=np.result_type(mag, float))
                    results[fqn] = Quantity(broadcasted_mag, const_qty.units)
                else:
                    results[fqn] = const_qty
                continue

            # --- START OF REVISED EVALUATION AND ERROR HANDLING BLOCK ---
            eval_locals = {}
            try:
                dependencies = context_info['dependencies']
                eval_locals = {
                    dep_name: freq_qty if dep_name == 'freq' else results[fqn_val]
                    for dep_name, fqn_val in dependencies.items()
                }

                # Step 1: Perform the core numerical evaluation.
                result_val = eval(definition.raw_value_or_expression_str, self._EVAL_GLOBALS, eval_locals)

                if not isinstance(result_val, Quantity):
                    # If the result is a bare number, assign the declared dimension.
                    result_val = Quantity(result_val, definition.declared_dimension_str)

                # Ensure the result is a NumPy array for vectorization consistency.
                if not isinstance(result_val.magnitude, np.ndarray):
                    result_val = Quantity(np.full_like(freq_hz, result_val.magnitude, dtype=float), result_val.units)

                # Step 2: Introspect for numerical errors (NaN/Inf) before unit conversion.
                # This allows us to find the exact failure indices for errors like division by zero.
                with np.errstate(invalid='ignore'): # Suppress warnings about NaN/Inf comparisons
                    numerical_errors_mask = np.isnan(result_val.magnitude) | np.isinf(result_val.magnitude)
                
                if np.any(numerical_errors_mask):
                    error_indices = np.argwhere(numerical_errors_mask).flatten()
                    first_fail_idx = error_indices[0]
                    fail_val = result_val.magnitude[first_fail_idx]
                    details = f"Expression resulted in a non-finite value ({fail_val})."
                    raise ParameterEvaluationError(
                        fqn=fqn, details=details,
                        error_indices=error_indices,
                        frequencies=freq_hz,
                        input_values=eval_locals
                    )

                # Step 3: If numerically valid, proceed with unit compatibility check and conversion.
                if definition.declared_dimension_str != "dimensionless":
                    if not result_val.is_compatible_with(definition.declared_dimension_str):
                        # This is a holistic dimensionality error. It's correct to not have specific indices.
                        raise pint.DimensionalityError(
                            result_val.units,
                            self._ureg.Unit(definition.declared_dimension_str),
                            extra_msg=f" for expression '{definition.raw_value_or_expression_str}'"
                        )
                    results[fqn] = result_val.to(definition.declared_dimension_str)
                else:
                    results[fqn] = result_val

            except Exception as e:
                # Step 4: Catch all errors and wrap them in the diagnosable exception.
                # If the error is already a ParameterEvaluationError (from the numerical check), re-raise it.
                if isinstance(e, ParameterEvaluationError):
                    raise
                
                # For other errors (NameError, pint.DimensionalityError, etc.), wrap them.
                # The `error_indices` will be None because these are typically holistic failures.
                raise ParameterEvaluationError(
                    fqn=fqn, details=str(e),
                    error_indices=None,
                    frequencies=freq_hz,
                    input_values=eval_locals
                ) from e
        
        return results

    def _clear_build_state(self):
        """Resets the manager to its initial, unbuilt state."""
        self._parameter_context_map = {}
        self._dependency_graph = nx.DiGraph()
        self._parsed_constants = {}
        self._evaluation_order = []
        self._build_complete = False

    def _create_context_map_from_definitions(self, all_definitions: List[ParameterDefinition]):
        """Initializes the internal context map from the flat list of definitions."""
        for definition in all_definitions:
            fqn = definition.fqn
            if fqn in self._parameter_context_map:
                raise ParameterDefinitionError(fqn=fqn, user_input=definition.raw_value_or_expression_str, source_yaml_path=definition.source_yaml_path, details=f"Duplicate parameter FQN '{fqn}' detected.")
            self._parameter_context_map[fqn] = {'definition': definition, 'dependencies': {}, 'sympy_expr': None}

    def _parse_all_values_and_find_deps(self, scope_maps: Dict[str, ChainMap]):
        """
        Uses SymPy and the preprocessor ONLY to find dependencies, not for evaluation logic.
        """
        preprocessor = ExpressionPreprocessor()
        for fqn, context_info in self._parameter_context_map.items():
            definition = context_info['definition']
            raw_value = definition.raw_value_or_expression_str

            # Check if it's a literal value (either with units or a plain number)
            is_literal = False
            if self._try_parse_literal_quantity_string(raw_value) is not None:
                is_literal = True
            else:
                try:
                    float(raw_value)
                    is_literal = True
                except ValueError:
                    is_literal = False

            if is_literal:
                continue

            # If not a literal, it must be an expression and must have a scope map.
            scope = scope_maps.get(fqn)
            if scope is None: raise ParameterError(f"Internal Error: No scope map provided for parameter expression '{fqn}'.")
            
            sympy_expr = preprocessor.preprocess(definition, scope, self.RESERVED_KEYWORDS)
            self._validate_expression_subset(fqn, sympy_expr)
            context_info['sympy_expr'] = sympy_expr
            
            dep_fqns = {str(s) for s in sympy_expr.free_symbols if s not in self.ALLOWED_SYMPY_SYMBOLS}
            dep_name_to_fqn_map = {}
            for dep_fqn in dep_fqns:
                if dep_fqn == 'freq':
                    dep_name_to_fqn_map['freq'] = 'freq'
                    continue
                
                found = False
                for m in scope.maps:
                    for key, val in m.items():
                        if val == dep_fqn:
                            dep_name_to_fqn_map[key] = dep_fqn
                            found = True
                            break
                    if found:
                        break
            context_info['dependencies'] = dep_name_to_fqn_map

    def _build_dependency_graph(self):
        """Constructs the NetworkX dependency graph from the parsed dependencies."""
        self._dependency_graph.add_nodes_from(self._parameter_context_map.keys())
        
        freq_is_dependency = False
        for ctx in self._parameter_context_map.values():
            expr = ctx.get('sympy_expr')
            if expr is not None and 'freq' in {str(s) for s in expr.free_symbols}:
                freq_is_dependency = True
                break
                
        if freq_is_dependency:
            self._dependency_graph.add_node('freq')

        for fqn, ctx_info in self._parameter_context_map.items():
            if (expr := ctx_info.get('sympy_expr')) is not None:
                dep_fqns = {str(s) for s in expr.free_symbols if s not in self.ALLOWED_SYMPY_SYMBOLS}
                for dep_fqn in dep_fqns:
                    if not self._dependency_graph.has_node(dep_fqn):
                        raise ParameterScopeError(
                            owner_fqn=fqn, unresolved_symbol=dep_fqn,
                            user_input=ctx_info['definition'].raw_value_or_expression_str,
                            source_yaml_path=ctx_info['definition'].source_yaml_path,
                            resolution_path_details=f"Dependency '{dep_fqn}' was not found in the dependency graph."
                        )
                    self._dependency_graph.add_edge(dep_fqn, fqn)

    def _check_circular_dependencies(self):
        """Raises a diagnosable error if any circular dependencies are found."""
        try:
            cycles = list(nx.simple_cycles(self._dependency_graph))
            if cycles:
                raise CircularParameterDependencyError(cycle=cycles[0])
        except nx.NetworkXError as e:
            raise ParameterError(f"NetworkX error during cycle check: {e}") from e

    def _compute_and_cache_constant_flags(self):
        """
        Computes the is_constant status for every parameter once and caches
        the boolean result directly in the internal context map.
        """
        logger.debug("Computing and caching is_constant flags for all parameters...")
        for fqn, context_info in self._parameter_context_map.items():
            is_const = 'freq' not in nx.ancestors(self._dependency_graph, fqn)
            context_info['is_constant'] = is_const

    def _compute_evaluation_order(self):
        """Calculates the topological sort of the dependency graph for evaluation."""
        self._evaluation_order = list(nx.topological_sort(self._dependency_graph))

    def _evaluate_and_cache_all_constants(self):
        """Pre-evaluates and caches all constant parameters using the new, safe method."""
        logger.debug("Pre-evaluating and caching all constant parameter values...")
        constant_results: Dict[str, Quantity] = {}

        for fqn in self._evaluation_order:
            if fqn == 'freq':
                continue

            if self._parameter_context_map[fqn].get('is_constant', False):
                ctx = self._parameter_context_map[fqn]
                definition = ctx['definition']

                try:
                    if ctx.get('sympy_expr') is not None:
                        dependencies = ctx['dependencies']
                        eval_locals = {dep_name: constant_results[fqn_val] for dep_name, fqn_val in dependencies.items()}
                        const_val = eval(definition.raw_value_or_expression_str, self._EVAL_GLOBALS, eval_locals)
                    else: # It's a literal value.
                        raw_val_str = definition.raw_value_or_expression_str
                        try:
                            const_val = float(raw_val_str)
                        except ValueError:
                            const_val = self._ureg.Quantity(raw_val_str)

                    # If the evaluated result is a bare number, assign the declared dimension.
                    if not isinstance(const_val, Quantity):
                        const_val = Quantity(const_val, definition.declared_dimension_str)

                    # Now, check for compatibility and convert.
                    if definition.declared_dimension_str != "dimensionless":
                        if not const_val.is_compatible_with(definition.declared_dimension_str):
                             raise pint.DimensionalityError(const_val.units, self._ureg.Unit(definition.declared_dimension_str), extra_msg=f" for expression '{definition.raw_value_or_expression_str}'")
                        constant_results[fqn] = const_val.to(definition.declared_dimension_str)
                    else:
                        constant_results[fqn] = const_val
                except Exception as e:
                    if isinstance(e, ParameterError): raise
                    raise ParameterEvaluationError(fqn=fqn, details=str(e), error_indices=None, frequencies=None, input_values={}) from e
        
        self._parsed_constants = constant_results
        logger.debug(f"Cached {len(self._parsed_constants)} constant parameter values.")

    def _validate_expression_subset(self, fqn: str, sympy_expr: Expr):
        """Ensures expressions only use the allowed subset of SymPy functions and constructs."""
        expr_str = self._parameter_context_map[fqn]['definition'].raw_value_or_expression_str
        definition = self._parameter_context_map[fqn]['definition']
        for node in sympy.preorder_traversal(sympy_expr):
            if isinstance(node, (Relational, BooleanFunction, Derivative, Integral, Lambda, Piecewise)):
                raise ParameterSyntaxError(owner_fqn=fqn, user_input=expr_str, source_yaml_path=definition.source_yaml_path, details=f"Disallowed operation type '{type(node).__name__}'.")
            if isinstance(node, Function) and not isinstance(node, UndefinedFunction) and type(node) not in self.ALLOWED_SYMPY_FUNCTIONS:
                 raise ParameterSyntaxError(owner_fqn=fqn, user_input=expr_str, source_yaml_path=definition.source_yaml_path, details=f"Disallowed function '{type(node).__name__}'.")

    def _try_parse_literal_quantity_string(self, value_str: str) -> Optional[Quantity]:
        """A helper to safely parse a string that might be a pint Quantity literal."""
        try:
            # Check if it's just a number, which should not be parsed as a Quantity here.
            float(value_str)
            return None
        except ValueError:
            # It's not a simple float, so try parsing it as a full quantity string.
            try:
                return self._ureg.Quantity(value_str)
            except (pint.UndefinedUnitError, pint.DimensionalityError, TypeError, ValueError):
                return None

    def _check_build_complete(self):
        """Guard method to ensure the manager is not used before it's fully built."""
        if not self._build_complete:
            raise ParameterError("ParameterManager has not been built. Call the 'build()' method first.")

    # --- Public API Methods (Post-Build) ---

    def get_all_fqn_definitions(self) -> List[ParameterDefinition]:
        """Returns a list of all parameter definitions known to the manager."""
        self._check_build_complete()
        return [ctx['definition'] for ctx in self._parameter_context_map.values()]

    def get_parameter_definition(self, fqn: str) -> ParameterDefinition:
        """Retrieves the definition object for a single parameter by its FQN."""
        self._check_build_complete()
        try:
            return self._parameter_context_map[fqn]['definition']
        except KeyError:
            raise ParameterScopeError(owner_fqn="<lookup>", unresolved_symbol=fqn, user_input="", source_yaml_path=Path(), resolution_path_details=f"FQN '{fqn}' not found in parameter map.")

    def is_constant(self, fqn: str) -> bool:
        """
        Public, safe method to check if a parameter is constant (i.e., does not depend on 'freq').
        """
        self._check_build_complete()
        if fqn == 'freq': 
            return False
        return self._parameter_context_map[fqn]['is_constant']

    def get_constant_value(self, fqn: str) -> Quantity:
        """Retrieves the pre-evaluated value of a constant parameter."""
        self._check_build_complete()
        if not self.is_constant(fqn):
            raise ParameterError(f"Parameter '{fqn}' is not a constant value (it depends on 'freq').")
        try:
            return self._parsed_constants[fqn]
        except KeyError:
            raise ParameterError(f"Internal Error: Constant '{fqn}' was not found in the pre-evaluation cache.") from None

    def get_external_dependencies_of_scope(self, fqns_in_scope: Set[str]) -> Tuple[Set[str], Set[str]]:
        """Identifies all external parameter dependencies for a given set of FQNs (a scope)."""
        self._check_build_complete()
        all_dependencies = set()
        for internal_fqn in fqns_in_scope:
            if internal_fqn in self._dependency_graph:
                all_dependencies.update(nx.ancestors(self._dependency_graph, internal_fqn))
        
        external_deps_fqns = {dep for dep in all_dependencies if dep not in fqns_in_scope and dep != 'freq'}
        const_ext_deps = {fqn for fqn in external_deps_fqns if self.is_constant(fqn)}
        freq_ext_deps = external_deps_fqns - const_ext_deps
        return const_ext_deps, freq_ext_deps