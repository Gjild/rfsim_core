# src/rfsim_core/parameters/preprocessor.py
import ast
import logging
from collections import ChainMap
from pathlib import Path
from typing import Set, Dict, TYPE_CHECKING

import sympy

from . import exceptions as param_exc

# Use TYPE_CHECKING to import types for static analysis without creating
# a circular dependency at runtime.
if TYPE_CHECKING:
    from parameters import ParameterDefinition

logger = logging.getLogger(__name__)


class _IdentifierResolver(ast.NodeVisitor):
    """
    AST visitor to resolve all identifiers in an expression to their FQNs.
    It carries full context to generate rich, diagnosable errors.
    """
    def __init__(
        self,
        owner_fqn: str,
        raw_expr: str,
        source_path: Path,
        scope: ChainMap,
        reserved_keywords: Set[str]
    ):
        self.owner_fqn = owner_fqn
        self.raw_expr = raw_expr
        self.source_path = source_path
        self.scope = scope
        self.reserved_keywords = reserved_keywords
        self.symbol_map: Dict[str, str] = {}

    def _resolve_attribute_chain(self, node: ast.Attribute) -> str:
        """Recursively builds a dot-separated string from an Attribute chain."""
        parts = []
        curr = node
        while isinstance(curr, ast.Attribute):
            parts.append(curr.attr)
            curr = curr.value
        if isinstance(curr, ast.Name):
            parts.append(curr.id)
            return ".".join(reversed(parts))
        else:
            source_segment = ast.get_source_segment(node) or "a complex expression"
            raise param_exc.ParameterSyntaxError(
                owner_fqn=self.owner_fqn,
                user_input=self.raw_expr,
                source_yaml_path=self.source_path,
                details=f"Unsupported expression structure: attribute access on a non-identifier part: '{source_segment}'."
            )

    def _generate_resolution_path_details(self, symbol: str) -> str:
        """Creates the detailed diagnostic string for a failed lookup."""
        lines = ["Resolution Path Searched:"]
        for i, mapping in enumerate(self.scope.maps):
            scope_name = f"Scope {i}"  # Generic name
            if i == 0:
                scope_name = f"Local Scope for '{self.owner_fqn}'"
            elif i == 1:
                # This is a heuristic; assumes the first non-local scope is the parent.
                scope_name = "Parent Scope"
            
            # Check if symbol is in the current map without triggering the ChainMap lookup
            if symbol in mapping:
                 lines.append(f"- {scope_name}: '{symbol}' FOUND.")
                 break
            else:
                 lines.append(f"- {scope_name}: '{symbol}' not found.")
        return "\n".join(lines)

    def visit_Attribute(self, node: ast.Attribute):
        """Handles qualified identifiers like 'amp1.R1.resistance'."""
        fqn_candidate = self._resolve_attribute_chain(node)
        if fqn_candidate in self.scope:
            resolved_fqn = self.scope[fqn_candidate]
            if fqn_candidate not in self.symbol_map:
                self.symbol_map[fqn_candidate] = resolved_fqn
        else:
            raise param_exc.ParameterScopeError(
                owner_fqn=self.owner_fqn,
                unresolved_symbol=fqn_candidate,
                user_input=self.raw_expr,
                source_yaml_path=self.source_path,
                resolution_path_details=self._generate_resolution_path_details(fqn_candidate)
            )
        # We do NOT call generic_visit, as we have consumed the entire attribute chain.

    def visit_Name(self, node: ast.Name):
        """
        Handles standalone identifiers like 'gain'. This is only called for names
        not part of an attribute chain, as visit_Attribute consumes those.
        """
        name = node.id
        if name in self.reserved_keywords:
            if name not in self.symbol_map:
                self.symbol_map[name] = name
            return

        if name in self.scope:
            if name not in self.symbol_map:
                self.symbol_map[name] = self.scope[name]
        else:
            # Inline import is a pragmatic choice to break the circular dependency
            # between ParameterManager and ExpressionPreprocessor at runtime.
            from parameters import ParameterManager
            if name not in ParameterManager._PARSE_GLOBALS:
                raise param_exc.ParameterScopeError(
                    owner_fqn=self.owner_fqn,
                    unresolved_symbol=name,
                    user_input=self.raw_expr,
                    source_yaml_path=self.source_path,
                    resolution_path_details=self._generate_resolution_path_details(name)
                )

class ExpressionPreprocessor:
    """
    Implements the architectural mandate for robust, FQN-based expression handling.
    It parses an expression string, resolves all identifiers to their canonical
    Fully Qualified Names (FQNs), and produces a SymPy expression where all free
    symbols are guaranteed to be these FQNs or reserved keywords.
    """

    def preprocess(
        self,
        definition: 'ParameterDefinition',
        scope: ChainMap,
        reserved_keywords: Set[str]
    ) -> sympy.Expr:
        """
        Transforms a string expression into a SymPy expression with fully-qualified symbols.

        Args:
            definition: The full ParameterDefinition object, providing all necessary context.
            scope: The lexical scope (ChainMap) for resolving identifiers.
            reserved_keywords: Keywords like 'freq' that are not parameters.

        Returns:
            A `sympy.Expr` object where all free symbols are FQNs or reserved keywords.
        """
        try:
            tree = ast.parse(definition.raw_value_or_expression_str, mode='eval')
        except SyntaxError as e:
            raise param_exc.ParameterSyntaxError(
                owner_fqn=definition.fqn,
                user_input=definition.raw_value_or_expression_str,
                source_yaml_path=definition.source_yaml_path,
                details=f"Invalid Python syntax: {e}"
            ) from e

        resolver = _IdentifierResolver(
            owner_fqn=definition.fqn,
            raw_expr=definition.raw_value_or_expression_str,
            source_path=definition.source_yaml_path,
            scope=scope,
            reserved_keywords=reserved_keywords
        )
        resolver.visit(tree.body)

        local_sympy_dict = {
            orig_name: sympy.Symbol(fqn)
            for orig_name, fqn in resolver.symbol_map.items()
        }

        try:
            # Inline import is a pragmatic choice to break circular dependency.
            from parameters import ParameterManager
            sympy_expr = sympy.parse_expr(
                definition.raw_value_or_expression_str,
                local_dict=local_sympy_dict,
                global_dict=ParameterManager._PARSE_GLOBALS,
                evaluate=False
            )
            return sympy_expr
        except Exception as e:
            raise param_exc.ParameterSyntaxError(
                owner_fqn=definition.fqn,
                user_input=definition.raw_value_or_expression_str,
                source_yaml_path=definition.source_yaml_path,
                details=f"Error creating SymPy expression: {e}"
            ) from e