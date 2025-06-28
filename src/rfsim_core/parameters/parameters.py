# src/rfsim_core/parameters/parameters.py
import logging
from typing import Dict, Any, Set, Optional, List, Callable, Tuple, ChainMap
from dataclasses import dataclass
from pathlib import Path

import pint
import sympy
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
    """
    The explicit contract object representing a single parameter's definition.
    It contains all necessary context for the ParameterManager's build process.
    """
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
    Manages all circuit parameters, their dependencies, and their evaluation.

    The manager is built once from a complete list of `ParameterDefinition` objects.
    Its build process performs a one-time, definitive topological sort of the
    dependency graph, producing a static evaluation order.

    The primary evaluation method, `evaluate_all`, leverages this static order to
    efficiently compute all parameter values for a given frequency sweep in a single,
    non-recursive, vectorized pass, returning a dictionary of vectorized `pint.Quantity` objects.
    """
    GLOBAL_SCOPE_PREFIX = "_rfsim_global_"

    def log10(x):
        return log(x, 10)

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

    def __init__(self):
        self._ureg = ureg
        self._parameter_context_map: Dict[str, Dict[str, Any]] = {}
        self._dependency_graph = nx.DiGraph()
        self._build_complete = False
        self._parsed_constants: Dict[str, Quantity] = {}
        self._compiled_functions: Dict[str, Callable] = {}
        self._evaluation_order: List[str] = []
        logger.info("ParameterManager initialized (empty).")

    def build(self, all_definitions: List[ParameterDefinition], scope_maps: Dict[str, ChainMap]):
        if self._build_complete:
            logger.warning("ParameterManager.build() called again. Rebuilding.")
            self._clear_build_state()

        if not all_definitions:
            logger.info("No parameter definitions provided. Build complete (empty).")
            self._build_complete = True
            return

        logger.info("Building ParameterManager context and dependency graph...")
        try:
            self._create_context_map_from_definitions(all_definitions)
            self._parse_all_values_and_find_deps(scope_maps)
            self._build_dependency_graph()
            self._check_circular_dependencies()
            self._compute_evaluation_order()
            self._validate_and_compile_all_expressions()
            self._evaluate_and_cache_all_constants()
        except ParameterError as e:
            logger.error(f"ParameterManager build failed: {e}", exc_info=False)
            self._clear_build_state()
            raise
        except Exception as e:
            logger.error(f"Unexpected error during ParameterManager build: {e}", exc_info=True)
            self._clear_build_state()
            raise ParameterError(f"Unexpected error during build: {e}") from e

        self._build_complete = True
        logger.info(f"ParameterManager build complete. Defined parameters: {len(self._parameter_context_map)}.")

    def evaluate_all(self, freq_hz: np.ndarray) -> Dict[str, Quantity]:
        """
        Evaluates all parameters for a given frequency array in a single, non-recursive pass.
        This is the primary method for obtaining parameter values for simulation.
        """
        self._check_build_complete()
        results: Dict[str, Quantity] = {}

        for fqn in self._evaluation_order:
            if fqn == 'freq': continue
            context_info = self._parameter_context_map[fqn]

            if self.is_constant(fqn):
                const_qty = self._parsed_constants[fqn]
                mag = const_qty.magnitude
                if not isinstance(mag, np.ndarray) or mag.shape != freq_hz.shape:
                    broadcasted_mag = np.full_like(freq_hz, mag, dtype=np.result_type(mag, float))
                    results[fqn] = Quantity(broadcasted_mag, const_qty.units)
                else:
                    results[fqn] = const_qty
                continue

            compiled_func = self._compiled_functions[fqn]
            sympy_expr = context_info['sympy_expr']
            
            arg_fqns = sorted([str(s) for s in sympy_expr.free_symbols if s not in self.ALLOWED_SYMPY_SYMBOLS], key=str)
            arg_values = [
                freq_hz if arg == 'freq' else results[arg].magnitude
                for arg in arg_fqns
            ]

            try:
                numerical_result = compiled_func(*arg_values)
                if not isinstance(numerical_result, np.ndarray):
                    numerical_result = np.full_like(freq_hz, numerical_result, dtype=float)
                
                non_finite_indices = np.where(~np.isfinite(numerical_result))[0]
                if non_finite_indices.size > 0:
                    raise ValueError(f"Result contains non-finite values (NaN or inf).")

                results[fqn] = Quantity(numerical_result, context_info['definition'].declared_dimension_str)

            except Exception as e:
                bad_indices = locals().get('non_finite_indices', None)
                input_values_for_error = {name: val for name, val in zip(arg_fqns, arg_values)}
                raise ParameterEvaluationError(
                    fqn=fqn, details=str(e),
                    error_indices=bad_indices,
                    frequencies=freq_hz,
                    input_values=input_values_for_error
                ) from e
        
        return results

    def get_external_dependencies_of_scope(self, fqns_in_scope: Set[str]) -> Tuple[Set[str], Set[str]]:
        """
        Analyzes the dependency graph to find all external parameters a given scope depends on.
        
        Args:
            fqns_in_scope: A set of FQNs that define the boundary of the scope.

        Returns:
            A tuple containing two sets:
            - The FQNs of all CONSTANT external parameters the scope depends on.
            - The FQNs of all FREQUENCY-DEPENDENT external parameters the scope depends on.
        """
        self._check_build_complete()
        all_dependencies = set()
        for internal_fqn in fqns_in_scope:
            if internal_fqn in self._dependency_graph:
                # Find all parameters that internal_fqn depends on (ancestors in graph).
                all_dependencies.update(nx.ancestors(self._dependency_graph, internal_fqn))
        
        external_deps_fqns = {dep for dep in all_dependencies if dep not in fqns_in_scope and dep != 'freq'}
        
        const_ext_deps = {fqn for fqn in external_deps_fqns if self.is_constant(fqn)}
        freq_ext_deps = external_deps_fqns - const_ext_deps
        
        return const_ext_deps, freq_ext_deps

    def _clear_build_state(self):
        self._parameter_context_map = {}
        self._dependency_graph = nx.DiGraph()
        self._parsed_constants = {}
        self._compiled_functions = {}
        self._evaluation_order = []
        self._build_complete = False

    def _create_context_map_from_definitions(self, all_definitions: List[ParameterDefinition]):
        for definition in all_definitions:
            fqn = definition.fqn
            if fqn in self._parameter_context_map:
                raise ParameterDefinitionError(fqn=fqn, user_input=definition.raw_value_or_expression_str, source_yaml_path=definition.source_yaml_path, details=f"Duplicate parameter FQN '{fqn}' detected.")
            self._parameter_context_map[fqn] = {'definition': definition, 'dependencies': set(), 'sympy_expr': None}

    def _parse_all_values_and_find_deps(self, scope_maps: Dict[str, ChainMap]):
        preprocessor = ExpressionPreprocessor()
        for fqn, context_info in self._parameter_context_map.items():
            definition = context_info['definition']
            raw_value = definition.raw_value_or_expression_str
            
            try:
                if self._try_parse_literal_quantity_string(raw_value) is not None:
                    continue
            except (pint.UndefinedUnitError, pint.DimensionalityError, TypeError, ValueError):
                pass

            scope = scope_maps.get(fqn)
            if scope is None: raise ParameterError(f"Internal Error: No scope map provided for parameter '{fqn}'.")
            
            sympy_expr = preprocessor.preprocess(definition, scope, self.RESERVED_KEYWORDS)
            context_info['sympy_expr'] = sympy_expr
            dependencies = {str(s) for s in sympy_expr.free_symbols if s not in self.ALLOWED_SYMPY_SYMBOLS}
            context_info['dependencies'] = dependencies

    def _build_dependency_graph(self):
        self._dependency_graph.add_nodes_from(self._parameter_context_map.keys())
        if any('freq' in ctx['dependencies'] for ctx in self._parameter_context_map.values()):
            self._dependency_graph.add_node('freq')

        for fqn, ctx_info in self._parameter_context_map.items():
            for dep_fqn in ctx_info['dependencies']:
                if not self._dependency_graph.has_node(dep_fqn):
                    raise ParameterScopeError(owner_fqn=fqn, unresolved_symbol=dep_fqn, user_input=ctx_info['definition'].raw_value_or_expression_str, source_yaml_path=ctx_info['definition'].source_yaml_path, resolution_path_details="Dependency was not found in graph.")
                self._dependency_graph.add_edge(dep_fqn, fqn) # Edge from dependency TO dependent

    def _check_circular_dependencies(self):
        try:
            cycles = list(nx.simple_cycles(self._dependency_graph))
            if cycles:
                raise CircularParameterDependencyError(cycle=cycles[0])
        except nx.NetworkXError as e:
            raise ParameterError(f"NetworkX error during cycle check: {e}") from e

    def _compute_evaluation_order(self):
        # Topological sort gives an ordering where each node comes before all nodes it points to.
        # Since our edges go from dependency -> dependent, this order is correct for evaluation.
        self._evaluation_order = list(nx.topological_sort(self._dependency_graph))

    def _validate_and_compile_all_expressions(self):
        for fqn, ctx in self._parameter_context_map.items():
            if (sympy_expr := ctx.get('sympy_expr')) is not None:
                self._validate_expression_subset(fqn, sympy_expr)
                free_symbols = sorted(list(sympy_expr.free_symbols), key=str)
                ctx['compiled_func'] = sympy.lambdify(free_symbols, sympy_expr, modules=['numpy'], cse=True)
                self._compiled_functions[fqn] = ctx['compiled_func']

    def _evaluate_and_cache_all_constants(self):
        logger.debug("Pre-evaluating and caching all constant parameters...")
        constant_results: Dict[str, Quantity] = {}
        for fqn in self._evaluation_order:
            if self.is_constant(fqn):
                ctx = self._parameter_context_map[fqn]
                if (compiled_func := ctx.get('compiled_func')) is not None:
                    arg_fqns = sorted([str(s) for s in ctx['sympy_expr'].free_symbols if s not in self.ALLOWED_SYMPY_SYMBOLS], key=str)
                    arg_values = [constant_results[arg].magnitude for arg in arg_fqns]
                    numerical_result = compiled_func(*arg_values)
                    constant_results[fqn] = Quantity(numerical_result, ctx['definition'].declared_dimension_str)
                else:
                    literal_val = self._try_parse_literal_quantity_string(ctx['definition'].raw_value_or_expression_str)
                    constant_results[fqn] = literal_val
        self._parsed_constants = constant_results
        logger.debug(f"Cached {len(self._parsed_constants)} constant parameter values.")

    def _validate_expression_subset(self, fqn: str, sympy_expr: Expr):
        expr_str = self._parameter_context_map[fqn]['definition'].raw_value_or_expression_str
        definition = self._parameter_context_map[fqn]['definition']
        for node in sympy.preorder_traversal(sympy_expr):
            if isinstance(node, (Relational, BooleanFunction, Derivative, Integral, Lambda, Piecewise)):
                raise ParameterSyntaxError(owner_fqn=fqn, user_input=expr_str, source_yaml_path=definition.source_yaml_path, details=f"Disallowed operation type '{type(node).__name__}'.")
            if node.is_Function and node.func not in self.ALLOWED_SYMPY_FUNCTIONS:
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
        # A parameter is constant if 'freq' is not one of its ancestors in the dependency graph.
        return 'freq' not in nx.ancestors(self._dependency_graph, fqn)

    def get_constant_value(self, fqn: str) -> Quantity:
        self._check_build_complete()
        if not self.is_constant(fqn):
            raise ParameterError(f"Parameter '{fqn}' is not a constant value (it depends on 'freq').")
        try:
            return self._parsed_constants[fqn]
        except KeyError:
            raise ParameterError(f"Internal Error: Constant '{fqn}' was not found in the pre-evaluation cache.") from None