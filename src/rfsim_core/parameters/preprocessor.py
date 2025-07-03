# src/rfsim_core/parameters/preprocessor.py
import ast
import logging
from collections import ChainMap
from pathlib import Path
from typing import Set, Dict, TYPE_CHECKING, Tuple

import sympy

from . import exceptions as param_exc

if TYPE_CHECKING:
    from .parameters import ParameterDefinition


logger = logging.getLogger(__name__)


class _AstTransformer(ast.NodeTransformer):
    """
    An AST NodeTransformer that resolves all identifiers to their FQNs and replaces
    them with unique, safe placeholder variables. This is the core of the robust
    pre-processing logic for SymPy parsing.
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
        self.symbol_map: Dict[str, sympy.Symbol] = {}
        self._placeholder_counter = 0

    def _get_placeholder(self) -> str:
        """Generates a unique, safe placeholder variable name."""
        name = f"_rfsim_var_{self._placeholder_counter}"
        self._placeholder_counter += 1
        return name

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
            # Use ast.unparse for modern Python versions if available, otherwise fallback
            try:
                source_segment = ast.unparse(node)
            except AttributeError:
                 source_segment = "a complex expression"
            raise param_exc.ParameterSyntaxError(
                owner_fqn=self.owner_fqn,
                user_input=self.raw_expr,
                source_yaml_path=self.source_path,
                details=f"Unsupported expression structure: attribute access on a non-identifier part: '{source_segment}'."
            )

    def visit_Attribute(self, node: ast.Attribute) -> ast.Name:
        """Transforms qualified identifiers like 'amp1.R1.resistance' into a safe placeholder."""
        fqn_candidate = self._resolve_attribute_chain(node)
        if fqn_candidate in self.scope:
            resolved_fqn = self.scope[fqn_candidate]
            placeholder = self._get_placeholder()
            self.symbol_map[placeholder] = sympy.Symbol(resolved_fqn)
            return ast.Name(id=placeholder, ctx=ast.Load())
        else:
            raise param_exc.ParameterScopeError(
                owner_fqn=self.owner_fqn,
                unresolved_symbol=fqn_candidate,
                user_input=self.raw_expr,
                source_yaml_path=self.source_path,
                resolution_path_details=f"The identifier '{fqn_candidate}' was not found in any accessible scope."
            )

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Transforms standalone identifiers like 'gain' or 'freq' into safe placeholders."""
        name = node.id
        # Reserved keywords like 'freq' are treated as valid symbols.
        if name in self.reserved_keywords:
            placeholder = self._get_placeholder()
            self.symbol_map[placeholder] = sympy.Symbol(name)
            return ast.Name(id=placeholder, ctx=ast.Load())

        if name in self.scope:
            resolved_fqn = self.scope[name]
            placeholder = self._get_placeholder()
            self.symbol_map[placeholder] = sympy.Symbol(resolved_fqn)
            return ast.Name(id=placeholder, ctx=ast.Load())
        
        # If not a parameter, check if it's a known SymPy global function or constant.
        from .parameters import ParameterManager
        if name in ParameterManager._PARSE_GLOBALS:
            return node # Leave functions like 'log' and constants like 'pi' untouched.
        
        # Otherwise, it is an unresolved symbol.
        raise param_exc.ParameterScopeError(
            owner_fqn=self.owner_fqn,
            unresolved_symbol=name,
            user_input=self.raw_expr,
            source_yaml_path=self.source_path,
            resolution_path_details=f"The identifier '{name}' was not found in any accessible scope."
        )


class ExpressionPreprocessor:
    """
    Implements robust, FQN-based expression handling. It parses an expression
    string, and uses an AST transformation to prepare it for safe SymPy parsing.
    Its sole purpose is to determine the FQN dependencies of an expression.
    """
    def preprocess(
        self,
        definition: 'ParameterDefinition',
        scope: ChainMap,
        reserved_keywords: Set[str]
    ) -> sympy.Expr:
        """
        Transforms a string expression into a SymPy expression with fully-qualified symbols.
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

        transformer = _AstTransformer(
            owner_fqn=definition.fqn,
            raw_expr=definition.raw_value_or_expression_str,
            source_path=definition.source_yaml_path,
            scope=scope,
            reserved_keywords=reserved_keywords
        )
        transformed_tree = transformer.visit(tree)
        
        # The local_dict maps the safe placeholders to their final FQN-based SymPy symbols.
        local_sympy_dict = transformer.symbol_map

        try:
            from .parameters import ParameterManager
            safe_expr_str = ast.unparse(transformed_tree)

            sympy_expr = sympy.parse_expr(
                safe_expr_str,
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
                details=f"Error creating SymPy expression from transformed AST: {e}"
            ) from e