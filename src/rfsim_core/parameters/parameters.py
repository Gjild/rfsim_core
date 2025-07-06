# src/rfsim_core/parameters/parameters.py

"""
Manages all circuit parameters, their dependencies, and their evaluation with
rigorous, dimensionally-aware arithmetic.

**Architectural Revision (Post-Phase 10):**
This module has been significantly hardened to enforce the "Correctness by Construction"
and "Actionable Diagnostics" mandates. The `ParameterManager.build()` method now
executes a rigorous, multi-stage validation pipeline that ensures all parameter
expressions are syntactically and semantically valid *before* any numerical
evaluation is attempted. This catches errors at the earliest possible stage and
provides superior diagnostic reports.
"""

import logging
import ast
from typing import Dict, Any, Set, List, Tuple, ChainMap
from dataclasses import dataclass
from pathlib import Path

import pint
import networkx as nx
import numpy as np

from ..units import ureg, Quantity
from .dependency_parser import ASTDependencyExtractor
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

    @property
    def fqn(self) -> str:
        """The canonical, fully qualified name (FQN) of the parameter."""
        return f"{self.owner_fqn}.{self.base_name}"


class _ExpressionEvaluator:
    """
    A private helper service that robustly evaluates a parameter expression string
    by first transforming its AST to use safe placeholder variables.
    """
    _EVAL_GLOBALS = {
        "ureg": ureg,
        "Quantity": Quantity,
        "np": np,
        "pi": np.pi,
        #"__builtins__": {"abs": abs, "min": min, "max": max} # A minimal, safe set
    }

    class _AstTransformer(ast.NodeTransformer):
        """
        Transforms an expression's AST by replacing dependencies with safe
        placeholder variables and building the corresponding evaluation scope.
        """
        def __init__(self, scope: ChainMap, evaluated_dependencies: Dict[str, Quantity]):
            self.scope = scope
            self.evaluated_dependencies = evaluated_dependencies
            self.eval_locals: Dict[str, Quantity] = {}
            self._placeholder_counter = 0

        def _get_placeholder(self) -> str:
            name = f"_rfsim_var_{self._placeholder_counter}"
            self._placeholder_counter += 1
            return name

        def _reconstruct_attribute_chain(self, node: ast.Attribute) -> str:
            """Same robust reconstruction logic as the dependency extractor."""
            parts = []
            curr = node
            while isinstance(curr, ast.Attribute):
                parts.append(curr.attr)
                curr = curr.value
            if isinstance(curr, ast.Name):
                parts.append(curr.id)
                return ".".join(reversed(parts))
            return ""

        def visit_Name(self, node: ast.Name) -> ast.Name:
            """
            Handles both special runtime values (like 'freq') and lexically-scoped
            parameter identifiers (like 'gain').
            """
            identifier_str = node.id

            if identifier_str == 'freq' and 'freq' in self.evaluated_dependencies:
                placeholder = self._get_placeholder()
                self.eval_locals[placeholder] = self.evaluated_dependencies['freq']
                return ast.Name(id=placeholder, ctx=ast.Load())

            if identifier_str in self.scope:
                resolved_fqn = self.scope[identifier_str]
                if resolved_fqn in self.evaluated_dependencies:
                    placeholder = self._get_placeholder()
                    self.eval_locals[placeholder] = self.evaluated_dependencies[resolved_fqn]
                    return ast.Name(id=placeholder, ctx=ast.Load())

            return node

        def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
            """Handles hierarchical identifiers like 'sub1.R_load.resistance'."""
            full_chain = self._reconstruct_attribute_chain(node)
            if full_chain and full_chain in self.scope:
                resolved_fqn = self.scope[full_chain]
                if resolved_fqn in self.evaluated_dependencies:
                    placeholder = self._get_placeholder()
                    self.eval_locals[placeholder] = self.evaluated_dependencies[resolved_fqn]
                    return ast.Name(id=placeholder, ctx=ast.Load())

            return self.generic_visit(node)

    def evaluate(
        self,
        definition: ParameterDefinition,
        scope: ChainMap,
        evaluated_dependencies: Dict[str, Quantity]
    ) -> Quantity:
        """
        Transforms and evaluates a parameter expression.
        """
        try:
            tree = ast.parse(definition.raw_value_or_expression_str, mode='eval')
        except SyntaxError as e:
            raise ParameterSyntaxError(
                owner_fqn=definition.fqn,
                user_input=definition.raw_value_or_expression_str,
                source_yaml_path=definition.source_yaml_path,
                details=f"Invalid Python syntax: {e}"
            ) from e

        transformer = self._AstTransformer(scope, evaluated_dependencies)
        transformed_tree = transformer.visit(tree)
        safe_expr_str = ast.unparse(transformed_tree)
        eval_locals = transformer.eval_locals

        return eval(safe_expr_str, self._EVAL_GLOBALS, eval_locals)


class ParameterManager:
    """
    Manages all circuit parameters, their dependencies, and their evaluation.
    """
    RESERVED_KEYWORDS = {'freq'}

    def __init__(self):
        self._ureg = ureg
        self._parameter_context_map: Dict[str, Dict[str, Any]] = {}
        self._dependency_graph = nx.DiGraph()
        self._build_complete = False
        self._parsed_constants: Dict[str, Quantity] = {}
        self._evaluation_order: List[str] = []
        self._scope_maps: Dict[str, ChainMap] = {}
        self._dependency_extractor = ASTDependencyExtractor()
        self._evaluator = _ExpressionEvaluator()
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
            # Step 1: Create the basic context map from definitions.
            self._create_context_map_from_definitions(all_definitions)

            # Step 2: Call the new, robust validation and graph-building method.
            # This performs all necessary build-time checks for syntax and semantics.
            self._validate_and_build_dependency_graph()

            # Step 3: Check for cycles in the now-validated graph of constants.
            self._check_circular_dependencies()

            # Step 4: Evaluate constants using the validated graph.
            self._evaluate_and_cache_constants()

            # Step 5 & 6 (No change): Compute final metadata.
            self._compute_and_cache_constant_flags()
            self._compute_evaluation_order()

        except ParameterError as e:
            self._clear_build_state()
            raise
        except Exception as e:
            self._clear_build_state()
            raise ParameterError(f"Unexpected error during ParameterManager build: {e}") from e

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

        for fqn in dynamic_fqns:
            if fqn == 'freq': continue

            definition = self._parameter_context_map[fqn]['definition']
            lexical_scope = self._scope_maps.get(fqn, ChainMap())

            try:
                result_val = self._evaluator.evaluate(definition, lexical_scope, base_eval_scope)

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
                    raise ParameterEvaluationError(fqn=fqn, details=details, error_indices=error_indices, frequencies=freq_hz, input_values=base_eval_scope)

                if definition.declared_dimension_str == "dimensionless":
                    if not result_val.dimensionless:
                        raise pint.DimensionalityError(result_val.units, "dimensionless", extra_msg=f" for parameter '{fqn}'")
                    final_qty = result_val
                else:
                    expected_unit = self._ureg.Unit(definition.declared_dimension_str)
                    if not result_val.is_compatible_with(expected_unit):
                        raise pint.DimensionalityError(result_val.units, expected_unit, extra_msg=f" for expression '{definition.raw_value_or_expression_str}'")
                    final_qty = result_val.to(expected_unit)

                results[fqn] = final_qty
                base_eval_scope[fqn] = final_qty

            except Exception as e:
                if isinstance(e, (ParameterEvaluationError, ParameterError)): raise
                raise ParameterEvaluationError(fqn=fqn, details=str(e), error_indices=None, frequencies=freq_hz, input_values=base_eval_scope) from e

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
            self._parameter_context_map[fqn] = {'definition': definition}

    def _validate_and_build_dependency_graph(self) -> Set[str]:
        """
        The definitive build-time validation and dependency analysis pipeline.

        This method iterates through ALL parameter definitions and performs rigorous checks:
        1.  Validates that the expression is syntactically correct Python.
        2.  Extracts all dependencies from the expression's AST.
        3.  For each dependency, verifies that it is either a known global, the special 'freq'
            keyword, or a symbol that can be resolved to another parameter's FQN.
        4.  Builds the dependency graph for all non-dynamic parameters.
        5.  Raises a specific, diagnosable error for any validation failure.
        """
        logger.debug("Starting comprehensive validation and dependency graph build...")
        self._dependency_graph = nx.DiGraph()
        self._dependency_graph.add_nodes_from(self._parameter_context_map.keys())
        dynamic_params: Set[str] = set()
        known_globals = set(self._evaluator._EVAL_GLOBALS.keys())

        for fqn, context in self._parameter_context_map.items():
            definition = context['definition']
            expression = definition.raw_value_or_expression_str
            scope = self._scope_maps.get(fqn, ChainMap())

            try:
                dependencies = self._dependency_extractor.get_dependencies(expression)
            except SyntaxError as e:
                raise ParameterSyntaxError(
                    owner_fqn=fqn,
                    user_input=expression,
                    source_yaml_path=definition.source_yaml_path,
                    details=f"Invalid Python syntax: {e}"
                ) from e

            is_dynamic = False
            for dep_name in dependencies:
                # A dependency is valid if:
                # 1. Its base (the part before the first '.') is a known global (e.g., 'np' in 'np.log').
                # 2. It is an exact match for a known global (e.g., 'pi').
                # 3. It is the special 'freq' keyword.
                # 4. It is resolvable in the current lexical scope.
                is_known_global_or_child = (dep_name in known_globals or dep_name.split('.')[0] in known_globals)

                if is_known_global_or_child:
                    continue

                if dep_name == 'freq':
                    is_dynamic = True
                    continue

                if dep_name not in scope:
                    raise ParameterScopeError(
                        owner_fqn=fqn,
                        unresolved_symbol=dep_name,
                        user_input=expression,
                        source_yaml_path=definition.source_yaml_path,
                        resolution_path_details=(
                            f"The symbol '{dep_name}' is not a known parameter in the current scope, "
                            "a recognized global (like 'np', 'pi'), or the 'freq' keyword."
                        )
                    )

                resolved_dep_fqn = scope[dep_name]
                self._dependency_graph.add_edge(resolved_dep_fqn, fqn)

            if is_dynamic:
                dynamic_params.add(fqn)

        # Prune dynamic parameters from the graph. The graph will now only contain
        # connections between parameters that are candidates for constant evaluation.
        self._dependency_graph.remove_nodes_from(dynamic_params)
        logger.debug(f"Validation and graph build complete. Found {len(dynamic_params)} dynamic parameters.")

    def _check_circular_dependencies(self):
        try:
            cycles = list(nx.simple_cycles(self._dependency_graph))
            if cycles:
                raise CircularParameterDependencyError(cycle=cycles[0])
        except nx.NetworkXError as e:
            raise ParameterError(f"NetworkX error during cycle check: {e}") from e

    def _evaluate_and_cache_constants(self):
        try:
            # The graph now only contains constant parameters, so this sort is correct.
            constant_evaluation_order = list(nx.topological_sort(self._dependency_graph))
        except nx.NetworkXUnfeasible:
            # This path is still a valid check for cycles among constants.
            raise CircularParameterDependencyError(cycle=[])

        logger.debug("Eagerly evaluating %d constants in dependency order...", len(constant_evaluation_order))
        self._parsed_constants = {}

        for fqn in constant_evaluation_order:
            definition = self._parameter_context_map[fqn]['definition']
            lexical_scope = self._scope_maps.get(fqn, ChainMap())

            try:
                result_val = self._evaluator.evaluate(definition, lexical_scope, self._parsed_constants)

                if not isinstance(result_val, Quantity):
                    result_val = Quantity(result_val, definition.declared_dimension_str)

                mag = result_val.magnitude
                if isinstance(mag, (float, int, np.number, np.ndarray)):
                    with np.errstate(invalid='ignore'):
                        if not np.all(np.isfinite(mag)):
                            fail_val = mag[~np.isfinite(mag)][0] if isinstance(mag, np.ndarray) and np.any(~np.isfinite(mag)) else mag
                            details = f"Constant expression resulted in a non-finite value ({fail_val})."
                            raise ValueError(details)

                if definition.declared_dimension_str == "dimensionless":
                    if not result_val.dimensionless:
                        raise pint.DimensionalityError(result_val.units, "dimensionless", extra_msg=f" for parameter '{fqn}'")
                    final_qty = result_val
                else:
                    expected_unit = self._ureg.Unit(definition.declared_dimension_str)
                    if not result_val.is_compatible_with(expected_unit):
                        raise pint.DimensionalityError(result_val.units, expected_unit, extra_msg=f" for expression '{definition.raw_value_or_expression_str}'")
                    final_qty = result_val.to(expected_unit)

                self._parsed_constants[fqn] = final_qty

            except Exception as e:
                if isinstance(e, ParameterError): raise
                raise ParameterEvaluationError(fqn=fqn, details=str(e), error_indices=None, frequencies=None, input_values=self._parsed_constants) from e

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

    def get_external_dependencies_of_scope(self, scope_fqns: Set[str]) -> Tuple[Set[str], Set[str]]:
        self._check_build_complete()
        const_deps = set()
        freq_deps = set()

        # This dependency graph only contains constant parameters now.
        constant_graph = self._dependency_graph
        all_deps = set()

        for fqn in scope_fqns:
            # Find dependencies for constant parameters via the graph
            if fqn in constant_graph:
                all_deps.update(nx.ancestors(constant_graph, fqn))
            # Find dependencies for dynamic parameters by parsing their expression
            elif not self.is_constant(fqn):
                expression = self.get_parameter_definition(fqn).raw_value_or_expression_str
                scope = self._scope_maps.get(fqn, ChainMap())
                deps_from_expr = self._dependency_extractor.get_dependencies(expression)
                for d in deps_from_expr:
                    if d in scope:
                        all_deps.add(scope[d])

        for dep_fqn in all_deps:
            if dep_fqn not in scope_fqns:
                if self.is_constant(dep_fqn):
                    const_deps.add(dep_fqn)
                else:
                    freq_deps.add(dep_fqn)
        return const_deps, freq_deps