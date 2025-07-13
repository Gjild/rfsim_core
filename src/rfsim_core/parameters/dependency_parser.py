# src/rfsim_core/parameters/dependency_parser.py

"""
Provides the ASTDependencyExtractor service for grammatically-correct dependency analysis.

This module is the definitive solution for parsing parameter expressions. It replaces
fragile regex-based methods with Python's own Abstract Syntax Tree (ast) module,
fulfilling the "Correctness by Construction" architectural mandate.

The core service, `ASTDependencyExtractor`, has a clear and explicit contract:
 - Given a syntactically valid Python expression, it will return a set of all
   simple, name-based dependencies (e.g., 'gain', 'sub1.R1.resistance').
 - Given a syntactically invalid expression, it will raise a `SyntaxError`,
   forcing the calling context to handle the error immediately and provide
   an actionable diagnostic to the user.
"""

import ast
import logging
from typing import Set

logger = logging.getLogger(__name__)


class _DependencyVisitor(ast.NodeVisitor):
    """
    An AST visitor that traverses a parsed expression tree and collects
    all identifiers (ast.Name) and qualified identifiers (ast.Attribute chains).
    """
    def __init__(self):
        self.dependencies: Set[str] = set()

    def _reconstruct_attribute_chain(self, node: ast.Attribute) -> str:
        """
        Recursively traverses an Attribute chain to build the full
        dot-separated identifier string (e.g., "sub1.R1.resistance").
        """
        parts = []
        curr = node
        while isinstance(curr, ast.Attribute):
            parts.append(curr.attr)
            curr = curr.value

        # The base of the chain must be a simple name. If it's another
        # expression type, it's not a valid parameter dependency.
        # The above is ENTIRELY intentional. COMPLEX EXPRESSIONS
        # (e.g., `(a+b).c` or `my_func().attribute`) ARE NOT VALID!
        if isinstance(curr, ast.Name):
            parts.append(curr.id)
            return ".".join(reversed(parts))
        else:
            # This handles invalid dependency structures like `(a+b).c` or
            # `my_func().attribute`. These are not simple parameter dependencies.
            # We return an empty string to signify this, which is handled by the caller.
            logger.debug(
                "Ignoring complex attribute access in dependency analysis: %s",
                ast.dump(node)
            )
            return ""

    def visit_Name(self, node: ast.Name):
        """
        Captures standalone identifiers like 'gain' or 'freq'.
        The calling context is responsible for filtering out keywords.
        """
        self.dependencies.add(node.id)

    def visit_Attribute(self, node: ast.Attribute):
        """
        Captures and reconstructs a full hierarchical identifier.
        If the attribute chain is a simple dependency (e.g. `a.b.c`), it is
        added as a single unit and the visitor does NOT descend further.
        If it is part of a complex expression (e.g. `(x+y).c`), it is
        ignored as a unit, and the visitor DESCENDS to find the child
        dependencies (e.g. `x` and `y`).
        """
        full_chain = self._reconstruct_attribute_chain(node)
        if full_chain:
            # We successfully processed the entire chain as one dependency.
            # Add it and DO NOT visit its children to avoid double-counting
            # components of the chain (e.g., adding 'sub1' for 'sub1.R1').
            self.dependencies.add(full_chain)
        else:
            # This was a complex attribute access like `(a+b).c` or `my_func().c`.
            # We couldn't process it as a single unit, so we MUST
            # descend into its children to find the dependencies within.
            self.generic_visit(node)


class ASTDependencyExtractor:
    """
    A stateless service to extract all potential dependencies from a
    Python expression string using Abstract Syntax Trees.
    """
    def get_dependencies(self, expression_str: str) -> Set[str]:
        """
        Parses an expression and returns a set of all identifiers found.

        Args:
            expression_str: The Python expression to analyze.

        Returns:
            A set of strings, where each string is a potential dependency.
            e.g., for "sub1.R1.resistance * gain", it returns
            {'sub1.R1.resistance', 'gain'}.

        Raises:
            SyntaxError: If the expression_str is not valid Python syntax.
                         This is the explicit contract of this service. The
                         caller is responsible for handling this error.
        """
        # An empty string is a valid expression that evaluates to nothing.
        if not expression_str.strip():
            return set()
            
        tree = ast.parse(expression_str, mode='eval')
        visitor = _DependencyVisitor()
        visitor.visit(tree)
        return visitor.dependencies