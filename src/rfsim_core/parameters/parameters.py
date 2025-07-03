# src/rfsim_core/parameters/parameters.py
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
    re, im, arg, conjugate, Expr, pi
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

    **Architectural Mandate:**
    The evaluation pipeline has been re-engineered to be **Correct by Construction**
    with respect to unit handling. `sympy.lambdify` has been removed. Evaluation
    is now performed using Python's `eval()` on the original expression strings, where
    the evaluation scope contains full `pint.Quantity` objects. This delegates
    all arithmetic and function calls to Pint, guaranteeing that any dimensionally
    incompatible operations (e.g., adding ohms and farads) will raise a
    `pint.DimensionalityError`, thus preventing silently incorrect results.
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
        **{func.__name__: func for func in ALLOWED_SYMPY_FUNCTIONS}
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
            if fqn == 'freq': continue
            context_info = self._parameter_context_map[fqn]
            definition = context_info['definition']

            if self.is_constant(fqn):
                const_qty = self._parsed_constants[fqn]
                mag = const_qty.magnitude
                if not isinstance(mag, np.ndarray) or mag.shape != freq_hz.shape:
                    broadcasted_mag = np.full_like(freq_hz, mag, dtype=np.result_type(mag, float))
                    results[fqn] = Quantity(broadcasted_mag, const_qty.units)
                else:
                    results[fqn] = const_qty
                continue
            
            try:
                # 1. Get dependencies and their already-computed Quantity values.
                dependencies = context_info['dependencies']
                eval_locals = {dep_name: freq_qty if dep_name == 'freq' else results[fqn_val]
                               for dep_name, fqn_val in dependencies.items()}

                # 2. Evaluate the original string using Pint-aware scope.
                result_qty = eval(definition.raw_value_or_expression_str, self._EVAL_GLOBALS, eval_locals)

                # 3. Ensure the result is a Quantity and broadcast if necessary.
                if not isinstance(result_qty, Quantity):
                    result_qty = Quantity(result_qty)
                if not isinstance(result_qty.magnitude, np.ndarray):
                    result_qty = Quantity(np.full_like(freq_hz, result_qty.magnitude, dtype=float), result_qty.units)

                # 4. Check final dimensionality against declaration.
                if not result_qty.is_compatible_with(definition.declared_dimension_str):
                    raise pint.DimensionalityError(
                        result_qty.units,
                        self._ureg.Unit(definition.declared_dimension_str),
                        extra_msg=f" for expression '{definition.raw_value_or_expression_str}'"
                    )
                
                results[fqn] = result_qty.to(definition.declared_dimension_str)

            except Exception as e:
                input_values_for_error = {name: val for name, val in eval_locals.items()}
                raise ParameterEvaluationError(
                    fqn=fqn, details=str(e),
                    error_indices=None,
                    frequencies=freq_hz,
                    input_values=input_values_for_error
                ) from e
        
        return results

    def _clear_build_state(self):
        self._parameter_context_map = {}
        self._dependency_graph = nx.DiGraph()
        self._parsed_constants = {}
        self._evaluation_order = []
        self._build_complete = False

    def _create_context_map_from_definitions(self, all_definitions: List[ParameterDefinition]):
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
            
            if self._try_parse_literal_quantity_string(raw_value) is not None:
                continue

            scope = scope_maps.get(fqn)
            if scope is None: raise ParameterError(f"Internal Error: No scope map provided for parameter '{fqn}'.")
            
            sympy_expr = preprocessor.preprocess(definition, scope, self.RESERVED_KEYWORDS)
            self._validate_expression_subset(fqn, sympy_expr)
            context_info['sympy_expr'] = sympy_expr
            
            # This is a critical change. We must store a map of the original identifiers used
            # in the expression (e.g., 'gain_in') to their resolved FQNs for the `eval` scope.
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
        self._dependency_graph.add_nodes_from(self._parameter_context_map.keys())
        if any('freq' in {str(s) for s in ctx.get('sympy_expr', sympy.S.EmptySet).free_symbols} for ctx in self._parameter_context_map.values()):
            self._dependency_graph.add_node('freq')

        for fqn, ctx_info in self._parameter_context_map.items():
            if (expr := ctx_info.get('sympy_expr')) is not None:
                dep_fqns = {str(s) for s in expr.free_symbols if s not in self.ALLOWED_SYMPY_SYMBOLS}
                for dep_fqn in dep_fqns:
                    if not self._dependency_graph.has_node(dep_fqn):
                        raise ParameterScopeError(owner_fqn=fqn, unresolved_symbol=dep_fqn, user_input=ctx_info['definition'].raw_value_or_expression_str, source_yaml_path=ctx_info['definition'].source_yaml_path, resolution_path_details="Dependency was not found in graph.")
                    self._dependency_graph.add_edge(dep_fqn, fqn)

    def _check_circular_dependencies(self):
        try:
            cycles = list(nx.simple_cycles(self._dependency_graph))
            if cycles:
                raise CircularParameterDependencyError(cycle=cycles[0])
        except nx.NetworkXError as e:
            raise ParameterError(f"NetworkX error during cycle check: {e}") from e

    def _compute_evaluation_order(self):
        self._evaluation_order = list(nx.topological_sort(self._dependency_graph))

    def _evaluate_and_cache_all_constants(self):
        """Pre-evaluates and caches all constant parameters using the new, safe method."""
        logger.debug("Pre-evaluating and caching all constant parameters...")
        constant_results: Dict[str, Quantity] = {}

        for fqn in self._evaluation_order:
            if self.is_constant(fqn):
                ctx = self._parameter_context_map[fqn]
                definition = ctx['definition']
                
                if ctx.get('sympy_expr') is not None:
                    dependencies = ctx['dependencies']
                    eval_locals = {dep_name: constant_results[fqn_val] for dep_name, fqn_val in dependencies.items()}
                    
                    try:
                        const_val = eval(definition.raw_value_or_expression_str, self._EVAL_GLOBALS, eval_locals)
                        if not isinstance(const_val, Quantity):
                            const_val = Quantity(const_val)
                        constant_results[fqn] = const_val.to(definition.declared_dimension_str)
                    except Exception as e:
                        raise ParameterEvaluationError(fqn=fqn, details=str(e), error_indices=None, frequencies=None, input_values={}) from e
                else:
                    literal_val = self._try_parse_literal_quantity_string(definition.raw_value_or_expression_str)
                    if literal_val is None:
                        try:
                            literal_val = Quantity(float(definition.raw_value_or_expression_str), definition.declared_dimension_str)
                        except ValueError as e:
                             raise ParameterDefinitionError(fqn, definition.raw_value_or_expression_str, definition.source_yaml_path, f"Could not parse as number: {e}") from e
                    
                    if not literal_val.is_compatible_with(definition.declared_dimension_str):
                         raise ParameterDefinitionError(fqn, definition.raw_value_or_expression_str, definition.source_yaml_path, "Incompatible units.")

                    constant_results[fqn] = literal_val
        
        self._parsed_constants = constant_results
        logger.debug(f"Cached {len(self._parsed_constants)} constant parameter values.")

    def _validate_expression_subset(self, fqn: str, sympy_expr: Expr):
        expr_str = self._parameter_context_map[fqn]['definition'].raw_value_or_expression_str
        definition = self._parameter_context_map[fqn]['definition']
        for node in sympy.preorder_traversal(sympy_expr):
            if isinstance(node, (Relational, BooleanFunction, Derivative, Integral, Lambda, Piecewise)):
                raise ParameterSyntaxError(owner_fqn=fqn, user_input=expr_str, source_yaml_path=definition.source_yaml_path, details=f"Disallowed operation type '{type(node).__name__}'.")
            if isinstance(node, UndefinedFunction) and node.func not in self.ALLOWED_SYMPY_FUNCTIONS:
                 raise ParameterSyntaxError(owner_fqn=fqn, user_input=expr_str, source_yaml_path=definition.source_yaml_path, details=f"Disallowed function '{node.func.__name__}'.")

    def _try_parse_literal_quantity_string(self, value_str: str) -> Optional[Quantity]:
        try: return self._ureg.Quantity(value_str)
        except (pint.UndefinedUnitError, pint.DimensionalityError, TypeError, ValueError): return None

    def _check_build_complete(self):
        if not self._build_complete:
            raise ParameterError("ParameterManager has not been built.")

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
        return 'freq' not in nx.ancestors(self._dependency_graph, fqn)

    def get_constant_value(self, fqn: str) -> Quantity:
        self._check_build_complete()
        if not self.is_constant(fqn):
            raise ParameterError(f"Parameter '{fqn}' is not a constant value (it depends on 'freq').")
        try:
            return self._parsed_constants[fqn]
        except KeyError:
            raise ParameterError(f"Internal Error: Constant '{fqn}' was not found in the pre-evaluation cache.") from None

    def get_external_dependencies_of_scope(self, fqns_in_scope: Set[str]) -> Tuple[Set[str], Set[str]]:
        self._check_build_complete()
        all_dependencies = set()
        for internal_fqn in fqns_in_scope:
            if internal_fqn in self._dependency_graph:
                all_dependencies.update(nx.ancestors(self._dependency_graph, internal_fqn))
        
        external_deps_fqns = {dep for dep in all_dependencies if dep not in fqns_in_scope and dep != 'freq'}
        const_ext_deps = {fqn for fqn in external_deps_fqns if self.is_constant(fqn)}
        freq_ext_deps = external_deps_fqns - const_ext_deps
        return const_ext_deps, freq_ext_deps