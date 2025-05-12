# --- src/rfsim_core/parameters.py ---
import logging
from typing import Dict, Any, Set, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
import pint
import sympy
# Import necessary SymPy objects for parsing, validation and typing
from sympy import (
    Symbol, Integer, Float, Rational, Add, Mul, Pow, I, Function,
    Abs, sqrt, log, exp,
    sin, cos, tan, asin, acos, atan, atan2,
    re, im, arg, conjugate, Expr, pi
)
# Import types for validation checks
from sympy.core.relational import Relational
from sympy.logic.boolalg import BooleanFunction
from sympy.calculus.util import AccumulationBounds # Base for Derivative/Integral args? Or check types directly.
from sympy import Derivative, Integral, Lambda, Piecewise

from sympy.parsing.sympy_parser import parse_expr
import networkx as nx
import numpy as np
import re # Kept for now, but usage changed for owner_ids

from .units import ureg, Quantity

logger = logging.getLogger(__name__)

# --- Helper for SymPy parsing with dotted names ---
class _InstanceProxy:
    __slots__ = ('_owner',)
    def __init__(self, owner): self._owner = owner
    def __getattr__(self, item):
        if not item.isidentifier():
            raise AttributeError(f"Invalid attribute name '{item}' for SymPy symbol generation.")
        # Generate symbol. Assume real=True initially, can be refined if needed.
        return sympy.Symbol(f"{self._owner}.{item}", real=True)

# --- Exceptions ---
class ParameterError(ValueError): pass
class ParameterSyntaxError(ParameterError): pass
class ParameterDefinitionError(ParameterError): pass
class ParameterScopeError(ParameterError): pass
class CircularParameterDependencyError(ParameterError):
    def __init__(self, cycle: List[str], *args):
        self.cycle = cycle
        cycle_display = cycle + [cycle[0]] if cycle and cycle[0] != cycle[-1] else cycle
        message = f"Circular dependency detected: {' -> '.join(cycle_display)}"
        super().__init__(message, *args)

# --- Data Structures ---
@dataclass
class ParameterDefinition:
    name: str
    scope: str
    owner_id: Optional[str] = None
    expression_str: Optional[str] = None
    constant_value_str: Optional[str] = None
    declared_dimension_str: Optional[str] = None
    is_sweepable: bool = False
    is_value_provided: bool = True # Added to track if value came from YAML

    def __post_init__(self):
        owner_ref = f"{self.owner_id}." if self.scope == 'instance' else 'global.'
        param_ref = f"'{owner_ref}{self.name}'"

        if not isinstance(self.name, str) or not self.name:
            raise ValueError("ParameterDefinition name must be a non-empty string.")
        if self.scope not in ['global', 'instance']:
            raise ValueError(f"Invalid scope '{self.scope}' for {param_ref}. Must be 'global' or 'instance'.")
        
        # Modified check: only require expression or constant if value is provided
        if self.is_value_provided and not self.expression_str and not self.constant_value_str:
            raise ValueError(f"ParameterDefinition {param_ref} must have either expression_str or constant_value_str when is_value_provided is True.")
        
        if self.expression_str and self.constant_value_str: # This check remains
            raise ValueError(f"ParameterDefinition {param_ref} cannot have both expression_str and constant_value_str.")
        
        if self.scope == 'instance' and not (self.owner_id and isinstance(self.owner_id, str)):
            raise ValueError(f"Instance-scoped parameter {param_ref} must have a valid non-empty string owner_id (got: {self.owner_id}).")
        if self.scope == 'global' and self.owner_id is not None:
            raise ValueError(f"Global-scoped parameter {param_ref} cannot have an owner_id (should be None, got: {self.owner_id}).")
        if self.declared_dimension_str is None or not isinstance(self.declared_dimension_str, str):
             raise ValueError(f"ParameterDefinition {param_ref} must have a valid declared_dimension_str string.")
                        
# Type alias for context map value
ContextInfo = Dict[str, Any] # Contains 'definition', 'declared_dimension', 'dependencies', 'sympy_expr'


class ParameterManager:
    GLOBAL_SCOPE_PREFIX = "global"

    def log10(x):
        return log(x, 10)

    # Allowed constants/functions for user expressions
    ALLOWED_SYMPY_FUNCTIONS = {
        Abs, sqrt, log, exp,
        log10,
        sin, cos, tan, asin, acos, atan, atan2,
        re, im, arg, conjugate,
    }
    ALLOWED_SYMPY_SYMBOLS = { sympy.pi, sympy.E, sympy.I }

    _PARSE_GLOBALS = {
        "Symbol": Symbol, "Integer": Integer, "Float": Float, "Rational": Rational,
        "Add": Add, "Mul": Mul, "Pow": Pow, "I": I,
        "Function": Function,
        "pi": sympy.pi, "E": sympy.E,
        **{func.__name__: func for func in ALLOWED_SYMPY_FUNCTIONS}
    }

    def __init__(self):
        self._ureg = ureg
        self._raw_definitions: List[ParameterDefinition] = []
        self._parameter_context_map: Dict[str, ContextInfo] = {}
        self._dependency_graph = nx.DiGraph()
        self._build_complete = False
        self._parsed_constants: Dict[str, Quantity] = {}
        self._compiled_functions: Dict[str, Callable] = {}
        logger.info("ParameterManager initialized (empty).")

    def add_definitions(self, definitions: List[ParameterDefinition]):
        if self._build_complete: raise ParameterError("Cannot add definitions after build.")
        if not isinstance(definitions, list): raise TypeError("Definitions must be list.")
        for i, d in enumerate(definitions):
            if not isinstance(d, ParameterDefinition): raise TypeError(f"Item {i} not ParameterDefinition.")
        self._raw_definitions.extend(definitions)
        logger.debug(f"Added {len(definitions)} raw parameter definitions.")

    def build(self):
        if self._build_complete: return
        if not self._raw_definitions:
            logger.info("No parameter definitions provided to ParameterManager. Build complete (empty).")
            self._build_complete = True
            return
        logger.info("Building ParameterManager context and dependency graph...")
        try:
            self._create_context_map_base()
            self._parse_expressions_and_find_dependencies()
            self._build_dependency_graph()
            self._check_circular_dependencies()
            self._parse_and_cache_constants() # Aware of is_value_provided
            self._validate_and_compile_all_expressions()
        except ParameterError as e:
            logger.error(f"ParameterManager build failed: {e}", exc_info=False)
            self._clear_build_state(); raise
        except Exception as e:
            logger.error(f"Unexpected error during ParameterManager build: {e}", exc_info=True)
            self._clear_build_state(); raise ParameterError(f"Unexpected error during build: {e}") from e

        self._build_complete = True
        self._raw_definitions = [] # Clear raw definitions after successful build
        logger.info(f"ParameterManager build complete. Defined parameters: {len(self._parameter_context_map)}. Compiled functions: {len(self._compiled_functions)}.")
        logger.debug(f"Cached constants: {len(self._parsed_constants)}")

    def _clear_build_state(self):
        self._parameter_context_map = {}
        self._dependency_graph = nx.DiGraph()
        self._parsed_constants = {}
        self._compiled_functions = {}
        self._build_complete = False

    def _get_internal_name(self, d: ParameterDefinition) -> str:
        if d.scope == 'global':
            if not d.name or '.' in d.name: raise ParameterDefinitionError(f"Invalid global name: '{d.name}'")
            return f"{self.GLOBAL_SCOPE_PREFIX}.{d.name}"
        elif d.scope == 'instance':
            if not d.name or '.' in d.name: raise ParameterDefinitionError(f"Invalid instance name: '{d.name}' for owner '{d.owner_id}'")
            return f"{d.owner_id}.{d.name}"
        else: raise ParameterDefinitionError(f"Unknown scope '{d.scope}' for {d.name}")

    def _parse_internal_name(self, name: str) -> Tuple[str, Optional[str], str]:
        if not isinstance(name, str): raise TypeError("Internal name must be str")
        parts = name.split('.', 1); scope_or_owner, base_name = parts
        if len(parts) != 2 or not scope_or_owner or not base_name:
            raise ParameterScopeError(f"Invalid internal name format: '{name}'. Expected 'scope.name' or 'owner.name'.")
        if scope_or_owner == self.GLOBAL_SCOPE_PREFIX:
            return ('global', None, base_name)
        else: # Assumed to be owner_id
            return ('instance', scope_or_owner, base_name)

    def _create_context_map_base(self):
        logger.debug("Creating base parameter context map...")
        temp_map: Dict[str, ContextInfo] = {}
        processed_names: Set[str] = set()
        for definition in self._raw_definitions:
            internal_name = self._get_internal_name(definition)
            if internal_name in processed_names:
                raise ParameterDefinitionError(f"Duplicate internal name '{internal_name}' derived from definition: {definition!r}")
            processed_names.add(internal_name)

            dim_str = definition.declared_dimension_str
            try:
                if dim_str != "dimensionless": # Special case for dimensionless
                    units = self._ureg.parse_units(dim_str)
                    if not isinstance(units, (pint.Unit, pint.util.UnitsContainer)): # Robust check
                        raise ValueError("Parsed unit object is not a valid Pint Unit or UnitsContainer.")
            except Exception as e:
                raise ParameterDefinitionError(f"Invalid declared_dimension_str '{dim_str}' for parameter '{internal_name}': {e}") from e

            temp_map[internal_name] = {
                'definition': definition,
                'declared_dimension': dim_str,
                'dependencies': set(), 
                'sympy_expr': None    
            }
        self._parameter_context_map = temp_map
        logger.debug(f"Base context map created: {len(temp_map)} entries.")

    def _parse_expressions_and_find_dependencies(self):
        logger.debug("Parsing expressions/references and identifying dependencies...")
        sympy_local_dict = {}

        owner_ids_from_definitions = {
            ctx_info['definition'].owner_id
            for ctx_info in self._parameter_context_map.values()
            if ctx_info['definition'].scope == 'instance' and ctx_info['definition'].owner_id
        }
        sympy_local_dict.update({
            oid: _InstanceProxy(oid) for oid in owner_ids_from_definitions if oid.isidentifier()
        })
        logger.debug(f"Populated sympy_local_dict for parsing with {len(sympy_local_dict)} owner proxies: {list(sympy_local_dict.keys())}")

        for internal_name, context_info in self._parameter_context_map.items():
            definition = context_info['definition']
            resolved_deps: Set[str] = set()
            context_info['dependencies'] = resolved_deps 
            context_info['sympy_expr'] = None      

            if not definition.is_value_provided: # If value not from YAML, no expression/constant to parse
                context_info['dependencies'] = set()
                logger.debug(f"Parameter '{internal_name}' has no value provided from YAML. Skipping dependency parsing.")
                continue

            if definition.constant_value_str and not definition.expression_str:
                value_str = definition.constant_value_str
                try:
                    _ = self._ureg.Quantity(value_str) 
                    context_info['dependencies'] = set() 
                except (pint.UndefinedUnitError, pint.DimensionalityError, TypeError, ValueError):
                    symbol_name_ref = value_str.strip()
                    try:
                        if not symbol_name_ref: raise ValueError("Constant string reference is empty.")
        
                        if '.' in symbol_name_ref: 
                            if symbol_name_ref.startswith(f"{self.GLOBAL_SCOPE_PREFIX}."):
                                if symbol_name_ref not in self._parameter_context_map:
                                    raise ParameterScopeError(f"Global parameter reference '{symbol_name_ref}' (from const value of '{internal_name}') not found.")
                                resolved_deps = {symbol_name_ref}
                            else: 
                                parts = symbol_name_ref.split('.', 1)
                                if len(parts) == 2 and parts[0].isidentifier() and parts[1].isidentifier():
                                    if symbol_name_ref not in self._parameter_context_map:
                                        raise ParameterScopeError(f"Explicit instance parameter reference '{symbol_name_ref}' (from const value of '{internal_name}') not found.")
                                    resolved_deps = {symbol_name_ref}
                                else:
                                    raise ParameterScopeError(f"Invalid explicit instance parameter reference format: '{symbol_name_ref}' for '{internal_name}'.")
                        elif symbol_name_ref.isidentifier(): 
                            resolved_deps = {self._resolve_symbol_to_internal_name(symbol_name_ref, definition.scope, definition.owner_id)}
                        else:
                            raise ValueError(f"Constant string '{symbol_name_ref}' for '{internal_name}' is not a valid literal quantity nor a valid parameter reference.")
                        context_info['dependencies'] = resolved_deps
                        logger.debug(f"Identified reference dependency for constant '{internal_name}' ('{value_str}'): {resolved_deps}")
                    except (ParameterScopeError, ValueError) as ref_err:
                        raise ParameterDefinitionError(f"Constant value string '{value_str}' for parameter '{internal_name}' is neither a valid literal quantity (e.g., '10 pF') nor a resolvable parameter reference. Error: {ref_err}") from ref_err
        
            elif definition.expression_str:
                expr_str = definition.expression_str
                try:
                    parsed_expr = parse_expr(expr_str, local_dict=sympy_local_dict, global_dict=self._PARSE_GLOBALS, evaluate=False)
                    context_info['sympy_expr'] = parsed_expr

                    free_symbols: Set[sympy.Symbol] = parsed_expr.free_symbols
                    for symbol in free_symbols:
                        symbol_name_str = str(symbol) 

                        if symbol_name_str in self._PARSE_GLOBALS or symbol in self.ALLOWED_SYMPY_SYMBOLS:
                            continue
                        if symbol_name_str == 'freq':
                            resolved_deps.add('freq')
                            continue

                        if '.' in symbol_name_str: 
                            if symbol_name_str not in self._parameter_context_map:
                                _, owner_cand, param_cand = self._parse_internal_name(symbol_name_str)
                                scope_hint = f"Owner: '{owner_cand}'" if owner_cand else "Global"
                                raise ParameterScopeError(
                                    f"Parameter reference '{symbol_name_str}' from expression for '{internal_name}' ('{expr_str}') "
                                    f"not found in defined parameters. ({scope_hint}, Param: '{param_cand}')"
                                )
                            resolved_deps.add(symbol_name_str)
                        else: 
                            resolved_internal_dep_name = self._resolve_symbol_to_internal_name(
                                symbol_name_str, definition.scope, definition.owner_id
                            )
                            resolved_deps.add(resolved_internal_dep_name)
                    context_info['dependencies'] = resolved_deps
                    logger.debug(f"Parsed expression for '{internal_name}': expr='{expr_str}', deps={resolved_deps}")
                except ParameterScopeError as e:
                     raise ParameterScopeError(f"Dependency resolution failed for expression parameter '{internal_name}' ('{expr_str}'): {e}") from e
                except (SyntaxError, TypeError, AttributeError, NameError) as e:
                    raise ParameterSyntaxError(f"Syntax error parsing expression for '{internal_name}': '{expr_str}'. Error: {type(e).__name__}('{e}')") from e
                except Exception as e:
                     if isinstance(e, ParameterError): raise
                     raise ParameterSyntaxError(f"Unexpected error parsing expression for '{internal_name}': '{expr_str}'. Error: {type(e).__name__}('{e}')") from e
        logger.debug("Finished parsing and dependency identification.")


    def _resolve_symbol_to_internal_name(self, symbol_name: str, current_scope: str, current_owner_id: Optional[str]) -> str:
        if '.' in symbol_name:
            # This case should ideally be handled before calling this function,
            # as _resolve_symbol_to_internal_name is for unqualified symbols.
            # If it's already qualified, check if it exists.
            if symbol_name in self._parameter_context_map:
                return symbol_name
            else:
                raise ParameterScopeError(f"Qualified symbol '{symbol_name}' not found in parameter context map.")


        if current_scope == 'instance' and current_owner_id:
            inst_name = f"{current_owner_id}.{symbol_name}"
            if inst_name in self._parameter_context_map:
                return inst_name

        glob_name = f"{self.GLOBAL_SCOPE_PREFIX}.{symbol_name}"
        if glob_name in self._parameter_context_map:
            return glob_name

        scope_info = f"scope='{current_scope}'" + (f", owner='{current_owner_id}'" if current_owner_id else "")
        searched_inst = f"'{current_owner_id}.{symbol_name}'" if current_scope == 'instance' and current_owner_id else "<N/A for instance>"
        searched_glob = f"'{self.GLOBAL_SCOPE_PREFIX}.{symbol_name}'"
        raise ParameterScopeError(
            f"Parameter symbol '{symbol_name}' referenced in an expression could not be resolved. "
            f"Context: {scope_info}. Searched for {searched_inst} and {searched_glob}."
        )


    def _build_dependency_graph(self):
        logger.debug("Building dependency graph...")
        self._dependency_graph = nx.DiGraph()
        self._dependency_graph.add_nodes_from(self._parameter_context_map.keys())

        if any('freq' in ctx_info.get('dependencies', set()) for ctx_info in self._parameter_context_map.values()):
            self._dependency_graph.add_node('freq') 

        for name, ctx_info in self._parameter_context_map.items():
            for dep_name in ctx_info.get('dependencies', set()):
                 if not self._dependency_graph.has_node(dep_name):
                     raise ParameterScopeError(f"Internal Error: Dependency '{dep_name}' for parameter '{name}' not found in graph nodes. This should have been caught earlier.")
                 self._dependency_graph.add_edge(name, dep_name) 
        logger.debug(f"Dependency graph built. Nodes: {self._dependency_graph.number_of_nodes()}, Edges: {self._dependency_graph.number_of_edges()}")

    def _check_circular_dependencies(self):
        logger.debug("Checking for circular dependencies...")
        try:
            cycles = list(nx.simple_cycles(self._dependency_graph))
        except nx.NetworkXError as e: 
            raise ParameterError(f"NetworkX error during cycle check: {e}") from e

        if cycles:
            raise CircularParameterDependencyError(cycles[0])
        logger.debug("No circular dependencies found.")

    def _parse_and_cache_constants(self):
        logger.debug("Pre-parsing and caching literal constant parameter values...")
        self._parsed_constants = {}
        for name, ctx_info in self._parameter_context_map.items():
            definition = ctx_info['definition']
            # Only parse if value was provided and it's a constant_value_str with no dependencies
            if definition.is_value_provided and definition.constant_value_str and not ctx_info.get('dependencies'):
                try:
                    self._parsed_constants[name] = self._ureg.Quantity(definition.constant_value_str)
                except Exception as e:
                    raise ParameterDefinitionError(f"Error parsing literal constant '{name}' with value '{definition.constant_value_str}': {e}") from e
        logger.debug(f"Finished caching {len(self._parsed_constants)} literal constants.")

    def _validate_and_compile_all_expressions(self):
        logger.info("Validating and compiling parameter expressions...")
        compile_errors = []
        num_compiled = 0
        for internal_name, context_info in self._parameter_context_map.items():
             if context_info.get('sympy_expr'): # Only if it has a sympy_expr (i.e., was an expression string)
                  try:
                       self._compile_expression(internal_name) 
                       num_compiled += 1
                  except ParameterError as e:
                       err_msg = f"Error for parameter '{internal_name}': {e}"
                       compile_errors.append(err_msg)
                  except Exception as e: 
                       compile_errors.append(f"Unexpected error processing expression for '{internal_name}': {type(e).__name__} - {e}")

        if compile_errors:
             full_error_message = "Parameter build failed due to expression validation/compilation errors:\n- " + "\n- ".join(compile_errors)
             raise ParameterError(full_error_message)
        logger.info(f"Successfully validated and compiled {num_compiled} expressions.")


    def _validate_expression_subset(self, internal_name: str, sympy_expr: Expr):
        definition = self._parameter_context_map[internal_name]['definition']
        expr_str_for_error = definition.expression_str or "<Expression not available>"
        resolved_dependencies = self._parameter_context_map[internal_name].get('dependencies', set())
    				
        allowed_dep_symbol_names = {'freq'}
        for dep_internal_name in resolved_dependencies - {'freq'}:
                allowed_dep_symbol_names.add(dep_internal_name) 
                _, _, base_name = self._parse_internal_name(dep_internal_name)
                allowed_dep_symbol_names.add(base_name) 
                
        for node in sympy.preorder_traversal(sympy_expr):
            node_type = type(node)

            if isinstance(node, (Relational, BooleanFunction, Derivative, Integral, Lambda, Piecewise)):
                raise ParameterSyntaxError(f"Disallowed operation type '{node_type.__name__}' in expression for '{internal_name}' ('{expr_str_for_error}').")

            elif node.is_Atom:
                if node.is_Function: 
                    if node not in self.ALLOWED_SYMPY_FUNCTIONS: 
                        if hasattr(node, 'func') and node.func not in self.ALLOWED_SYMPY_FUNCTIONS:
                            raise ParameterSyntaxError(f"Disallowed function atom '{node}' (func: {node.func.__name__}) in expression for '{internal_name}' ('{expr_str_for_error}').")
                elif node.is_Symbol:
                    if node in self.ALLOWED_SYMPY_SYMBOLS: continue 
                    if str(node) == 'freq': continue 
                    if str(node) in allowed_dep_symbol_names: continue

                    raise ParameterSyntaxError(
                        f"Disallowed or unresolved symbol '{str(node)}' in expression for '{internal_name}' ('{expr_str_for_error}'). "
                        f"Allowed symbols: {[str(s) for s in self.ALLOWED_SYMPY_SYMBOLS]}, 'freq'. "
                        f"Resolved dependencies imply allowed variable names: {allowed_dep_symbol_names}."
                    )
                elif node.is_Number:
                    # Check for SymPy's infinity/NaN representations explicitly
                    if node in (sympy.oo, -sympy.oo, sympy.zoo, sympy.nan): # ADDED THIS CHECK
                        raise ParameterSyntaxError(f"Disallowed number '{node}' (Infinity/NaN) in expression for '{internal_name}' ('{expr_str_for_error}').")
                    if node.is_finite is False: # Catches other cases if any
                        raise ParameterSyntaxError(f"Disallowed number '{node}' (Infinity/NaN) in expression for '{internal_name}' ('{expr_str_for_error}').")
                    if not isinstance(node, (Integer, Float, Rational)):
                        pass # Allow other valid numbers
            elif node.is_Function: 
                if node.func not in self.ALLOWED_SYMPY_FUNCTIONS:
                    raise ParameterSyntaxError(f"Disallowed function call '{node.func.__name__}' in expression for '{internal_name}' ('{expr_str_for_error}').")

            elif isinstance(node, (Add, Mul, Pow)):
                continue 
            else: 
                # This case should ideally not be hit if all SymPy expression types are categorized
                # However, it serves as a fallback to prevent unexpected elements.
                # Check if it's a known constant like oo, zoo, nan again, as they might not be Atoms.
                if node in (sympy.oo, -sympy.oo, sympy.zoo, sympy.nan): # ADDED THIS CHECK
                    raise ParameterSyntaxError(f"Disallowed constant '{node}' (Infinity/NaN) in expression for '{internal_name}' ('{expr_str_for_error}').")
                
                # If it's not an explicitly allowed operation or known constant, raise error.
                # This helps catch things like sympy.Sum, sympy.Product if they slip through.
                raise ParameterSyntaxError(f"Unexpected expression element '{node}' of type '{node_type.__name__}' in expression for '{internal_name}' ('{expr_str_for_error}'). Review allowed operations and SymPy object types.")

    def _compile_expression(self, internal_name: str):
        if internal_name in self._compiled_functions:
            return 

        context_info = self._parameter_context_map[internal_name]
        sympy_expr = context_info.get('sympy_expr')
        definition = context_info['definition']

        if not sympy_expr: 
            # This can happen if is_value_provided was false, or it was a constant string ref.
            # Only compile if sympy_expr was actually created.
            logger.debug(f"Skipping compilation for '{internal_name}' as it has no SymPy expression (e.g. constant or no value provided).")
            return

        expr_str_for_error = definition.expression_str or "<Expression not available>"
        logger.debug(f"Validating and compiling expression for '{internal_name}': {expr_str_for_error}")

        try:
            self._validate_expression_subset(internal_name, sympy_expr)
            logger.debug(f"Expression subset validation passed for '{internal_name}'.")

            free_symbols_in_expr = list(sympy_expr.free_symbols)
            arg_symbols = []
            has_freq = False

            for s in free_symbols_in_expr:
                if str(s) == 'freq':
                    has_freq = True
                elif s not in self.ALLOWED_SYMPY_SYMBOLS:
                    arg_symbols.append(s)

            arg_symbols.sort(key=str)
            final_args_for_lambdify = ([Symbol('freq')] if has_freq else []) + arg_symbols
            
            compiled_func = sympy.lambdify(
                final_args_for_lambdify,
                sympy_expr,
                modules=['numpy'], 
                cse=True 
            )
            logger.debug(f"Lambdify compilation successful for '{internal_name}'. Args: {[str(a) for a in final_args_for_lambdify]}")

            self._compiled_functions[internal_name] = compiled_func
        except ParameterSyntaxError: 
            raise
        except Exception as e:
            logger.error(f"Failed to compile expression for '{internal_name}' ('{expr_str_for_error}') with lambdify: {e}", exc_info=True) 
            raise ParameterError(f"Compilation failed for expression parameter '{internal_name}' ('{expr_str_for_error}'): {type(e).__name__} - {e}") from e

    def get_parameter_definition(self, internal_name: str) -> ParameterDefinition:
        self._check_build_complete()
        try:
            return self._parameter_context_map[internal_name]['definition']
        except KeyError:
            raise ParameterScopeError(f"Parameter with internal name '{internal_name}' not found.")

    def get_declared_dimension(self, internal_name: str) -> str:
        self._check_build_complete()
        try:
            return self._parameter_context_map[internal_name]['declared_dimension']
        except KeyError:
            raise ParameterScopeError(f"Parameter with internal name '{internal_name}' not found.")

    def get_dependencies(self, internal_name: str) -> Set[str]:
        self._check_build_complete()
        try:
            return self._parameter_context_map[internal_name]['dependencies'].copy()
        except KeyError:
            raise ParameterScopeError(f"Parameter with internal name '{internal_name}' not found.")

    def is_constant(self, internal_name: str) -> bool:
        self._check_build_complete()
        
        context_info = self._parameter_context_map.get(internal_name)
        if not context_info:
            raise ParameterScopeError(f"Parameter '{internal_name}' not found.")
        
        definition = context_info['definition']
        if not definition.is_value_provided:
            return False # Cannot be constant if no value was provided from YAML

        # If value was provided, then check if it's in _parsed_constants or has no compiled func
        if internal_name in self._parsed_constants:
            return True
        if internal_name in self._compiled_functions:
            return False 
        # If it's in context_map, value was provided, but not an expression and not a direct literal: it's a constant reference
        # This should have been resolved into _parsed_constants by _resolve_constant_recursive if it's truly constant.
        # The most reliable check after build is: is it in _parsed_constants (literal or resolved ref)
        # OR is it a definition that is not an expression (no compiled_func) and has no 'freq' dependency.
        
        # More robust check:
        if 'freq' in context_info.get('dependencies', set()):
            return False # Depends on frequency
        
        # If no 'freq' dependency, check if all its parameter dependencies are themselves constant
        # This is essentially what _resolve_constant_recursive does.
        try:
            self._resolve_constant_recursive(internal_name, set()) # This will attempt to resolve and cache
            return internal_name in self._parsed_constants # If successful, it's cached
        except ParameterError: # If it fails to resolve to a constant (e.g. dep is expression)
            return False


    def get_constant_value(self, internal_name: str) -> Quantity:
        self._check_build_complete()
        
        # First, ensure the parameter definition indicates a value was provided
        definition = self.get_parameter_definition(internal_name) # Handles not found
        if not definition.is_value_provided:
            raise ParameterError(f"Cannot get constant value for parameter '{internal_name}': No value was provided in the netlist YAML.")

        if internal_name in self._parsed_constants:
            return self._parsed_constants[internal_name]

        if internal_name in self._compiled_functions:
            raise ParameterError(f"Parameter '{internal_name}' is an expression ('{definition.expression_str}') and cannot be retrieved as a simple constant value. Use resolve_parameter().")

        try:
            # _resolve_constant_recursive will also check is_value_provided for dependencies
            return self._resolve_constant_recursive(internal_name, set())
        except ParameterError as e: 
            raise ParameterError(f"Failed to resolve parameter '{internal_name}' to a constant value: {e}") from e
        except KeyError: 
            # This path should ideally not be hit if get_parameter_definition succeeded.
            raise ParameterScopeError(f"Parameter '{internal_name}' not found during constant value resolution (internal error).")


    def _resolve_constant_recursive(self, internal_name: str, visited: Set[str]) -> Quantity:
        if internal_name in visited:
            raise CircularParameterDependencyError(list(visited) + [internal_name])
        visited.add(internal_name)

        if internal_name in self._parsed_constants:
            return self._parsed_constants[internal_name]

        try:
            context_info = self._parameter_context_map[internal_name]
            definition = context_info['definition']

            if not definition.is_value_provided:
                # This check is crucial for dependencies.
                raise ParameterError(f"Dependency parameter '{internal_name}' required for constant resolution had no value provided in YAML.")

            if internal_name in self._compiled_functions: # It's an expression
                 raise ParameterError(f"Parameter '{internal_name}' ('{definition.expression_str}') is an expression, cannot resolve as constant recursively.")

            if not definition.constant_value_str: 
                # This means it's not a literal, not an expression, but should be a reference.
                # However, if is_value_provided is true, constant_value_str should exist for non-expressions.
                # This indicates an issue if definition.is_value_provided but constant_value_str is None.
                # The ParameterDefinition __post_init__ should prevent this state if is_value_provided is True.
                raise ParameterError(f"Internal logic error: _resolve_constant_recursive called on '{internal_name}' which has is_value_provided=True but no constant_value_str and is not an expression.")

            dependencies = context_info.get('dependencies', set())
            if 'freq' in dependencies:
                raise ParameterError(f"Parameter '{internal_name}' depends on 'freq' and cannot be resolved as a simple constant.")

            if not dependencies: # Should have been caught by _parse_and_cache_constants
                # This implies it's a literal constant that wasn't cached.
                if definition.constant_value_str: # It must be a literal
                    try:
                        val_qty = self._ureg.Quantity(definition.constant_value_str)
                        self._parsed_constants[internal_name] = val_qty
                        return val_qty
                    except Exception as e:
                         raise ParameterError(f"Failed to parse literal constant '{internal_name}' ('{definition.constant_value_str}') during recursive resolution: {e}") from e
                else: # Should not happen due to ParameterDefinition post_init
                    raise ParameterError(f"Internal Error: Parameter '{internal_name}' has no dependencies, no expression, but also no constant_value_str despite is_value_provided=True.")

            if len(dependencies) == 1:
                dep_name = list(dependencies)[0]
                resolved_dep_quantity = self._resolve_constant_recursive(dep_name, visited.copy()) 
                self._parsed_constants[internal_name] = resolved_dep_quantity # Cache result
                return resolved_dep_quantity
            else: # Should not happen for a constant_value_str that is a reference
                raise ParameterError(f"Internal Error: Parameter '{internal_name}' defined as constant reference ('{definition.constant_value_str}') has unexpected multiple dependencies: {dependencies}.")

        except KeyError: 
            raise ParameterScopeError(f"Dependency parameter '{internal_name}' not found during recursive constant resolution.")


    def get_all_internal_names(self) -> List[str]:
        self._check_build_complete()
        return list(self._parameter_context_map.keys())

    def _check_build_complete(self):
        if not self._build_complete:
            raise ParameterError("ParameterManager has not been built. Call build() after adding all definitions.")

    def get_compiled_function(self, internal_name: str) -> Optional[Callable]:
        self._check_build_complete()
        return self._compiled_functions.get(internal_name)

    def resolve_parameter(self, internal_name: str, freq_hz: np.ndarray, target_dimension_str: str, evaluation_context: Dict[Tuple[str, str], Quantity]) -> Quantity:
        """
        Resolves a parameter to a pint.Quantity.
        Handles constants, expressions, dependencies, and frequency.
        Uses context-aware unit handling: numerical evaluation first, then Quantity creation
        with the target_dimension_str. The magnitude of the returned Quantity will be a NumPy array
        if freq_hz is an array, even for constants.

        Args:
            internal_name: The fully qualified internal name of the parameter (e.g., "R1.resistance", "global.gain").
            freq_hz: NumPy array of frequencies in Hertz (unitless). Lambdify functions expect this.
                     Typically a 1-element array, e.g., np.array([current_frequency_scalar]).
            target_dimension_str: The dimension string (e.g., "ohm", "farad", "dimensionless")
                                  that the final Quantity result must have.
            evaluation_context: A dictionary used for memoization (caching) of resolved parameters
                                within a single top-level resolution call (e.g., per frequency point).
                                Key: (internal_name, target_dimension_str), Value: resolved Quantity.

        Returns:
            A pint.Quantity object representing the resolved parameter value, compatible with target_dimension_str.
            The .magnitude of the Quantity will be a NumPy array matching the shape of freq_hz if freq_hz was an array.

        Raises:
            ParameterError (and subtypes): If resolution fails.
            pint.DimensionalityError: If a resolved constant is incompatible with target_dimension_str.
        """
        self._check_build_complete()
        
        # Ensure freq_hz is a NumPy array.
        if not isinstance(freq_hz, np.ndarray):
            logger.warning(f"resolve_parameter for '{internal_name}' received freq_hz of type {type(freq_hz)}, expected np.ndarray. Attempting conversion.")
            if isinstance(freq_hz, (list, tuple)) and not freq_hz: # Handle empty list/tuple explicitly
                 freq_hz_arr = np.array([], dtype=float)
            elif isinstance(freq_hz, (list, tuple, np.ndarray)):
                 freq_hz_arr = np.array(freq_hz, dtype=float)
            else: # Scalar case
                 freq_hz_arr = np.array([freq_hz], dtype=float)
        else:
            freq_hz_arr = freq_hz
        
        # Ensure freq_hz_arr is at least 1D for operations like np.full_like, unless it's intentionally empty
        if freq_hz_arr.ndim == 0:
            freq_hz_arr = freq_hz_arr.reshape(1)
        elif freq_hz_arr.ndim > 1: # Should not happen if caller (MNA) passes 1D array
            logger.warning(f"resolve_parameter for '{internal_name}' received freq_hz_arr with ndim > 1 ({freq_hz_arr.shape}). Flattening.")
            freq_hz_arr = freq_hz_arr.flatten()


        logger.debug(f"Resolving parameter '{internal_name}' for target dimension '{target_dimension_str}' (freq_hz_arr shape: {freq_hz_arr.shape})")

        # 1. Check evaluation_context (memoization cache)
        cache_key = (internal_name, target_dimension_str)
        if cache_key in evaluation_context:
            logger.debug(f"  Cache hit for {cache_key}. Returning cached value.")
            return evaluation_context[cache_key]

        # 2. Lookup parameter context
        try:
            context_info = self._parameter_context_map[internal_name]
            definition = context_info['definition']
        except KeyError:
            raise ParameterScopeError(f"Parameter '{internal_name}' not found in context map during resolution.")

        # 3. Handle literal constants and resolvable constant references
        try:
            # get_constant_value will raise ParameterError if value not provided
            constant_quantity = self.get_constant_value(internal_name)
            logger.debug(f"  '{internal_name}' resolved as constant: {constant_quantity:~P}")

            try:
                target_units = self._ureg.parse_units(target_dimension_str) if target_dimension_str != "dimensionless" else self._ureg.dimensionless
            except Exception as e: 
                raise ParameterError(f"Invalid target_dimension_str '{target_dimension_str}' for parameter '{internal_name}': {e}") from e

            if not constant_quantity.is_compatible_with(target_units):
                raise pint.DimensionalityError(
                    constant_quantity.units, target_units,
                    constant_quantity.dimensionality, target_units.dimensionality,
                    extra_msg=f"Resolved constant value for '{internal_name}' ({constant_quantity:~P}) is not compatible with target dimension '{target_dimension_str}'"
                )

            final_qty_scalar = constant_quantity.to(target_units)
            
            final_qty_magnitude = final_qty_scalar.magnitude
            if isinstance(final_qty_magnitude, (int, float, complex)): 
                if freq_hz_arr.size > 0: 
                    broadcasted_magnitude = np.full_like(freq_hz_arr, final_qty_magnitude, dtype=np.result_type(final_qty_magnitude, float)) # ensure complex if needed, or float
                elif freq_hz_arr.size == 0: 
                    broadcasted_magnitude = np.array([], dtype=np.result_type(final_qty_magnitude, float))
                else: # Should be caught by ndim reshape, fallback for safety.
                    broadcasted_magnitude = np.array([final_qty_magnitude], dtype=np.result_type(final_qty_magnitude, float))
                final_qty = Quantity(broadcasted_magnitude, final_qty_scalar.units)
                logger.debug(f"  Broadcasted constant '{internal_name}' magnitude to match freq_hz_arr shape {freq_hz_arr.shape}.")
            elif isinstance(final_qty_magnitude, np.ndarray) and final_qty_magnitude.shape != freq_hz_arr.shape and freq_hz_arr.size > 0:
                logger.warning(f"Constant parameter '{internal_name}' unexpectedly yielded an array-like magnitude {final_qty_magnitude.shape} "
                               f"that does not match freq_hz_arr shape {freq_hz_arr.shape}. This might indicate an issue if the constant was expected to be scalar. Using as is.")
                final_qty = final_qty_scalar # Use the original array-like Quantity
            else: # Magnitude is already an array of correct shape, or freq_hz_arr is empty
                final_qty = final_qty_scalar


            evaluation_context[cache_key] = final_qty
            logger.debug(f"  Constant '{internal_name}' successfully validated, (broadcasted), and converted. Cached.")
            return final_qty

        except ParameterError as e:
            # This exception could be from get_constant_value (e.g. value not provided, or it's an expression)
            # or from incompatible target_dimension_str for a constant.
            # If it's an expression, we proceed. If it's truly unresolvable (e.g. no value AND no expression), we error.
            compiled_func_check = self.get_compiled_function(internal_name)
            if compiled_func_check is None: # Not an expression
                # If get_constant_value failed (e.g. no value provided, or circular ref for consts)
                # AND it's not an expression, then it's an error that should be re-raised.
                # SemanticValidator should have reported missing values to user.
                # Here, it means an attempt to *use* an unprovided, non-expression parameter.
                logger.debug(f"  Parameter '{internal_name}' is not a resolvable constant ({e}) and not a compiled expression. Re-raising.")
                raise # Re-raise the error from get_constant_value or dimensionality check
            
            logger.debug(f"  '{internal_name}' is an expression (get_constant_value failed as expected with: {e}). Proceeding with expression evaluation.")

        # 4. If it's an expression, get its compiled function
        compiled_func = self.get_compiled_function(internal_name) 
        if not compiled_func: # Should be redundant due to check above, but for safety.
            raise ParameterError(f"Internal logic error: Parameter '{internal_name}' has no compiled function despite passing earlier checks (is_value_provided={definition.is_value_provided}).")

        # 5. Resolve dependencies recursively and prepare arguments for the compiled function
        dependencies_set = context_info.get('dependencies', set()) 
        func_args_values = [] 

        sympy_expr = context_info['sympy_expr']
        if not sympy_expr: 
             raise ParameterError(f"Internal Error: No sympy_expr for compiled parameter '{internal_name}'.")

        free_symbols_in_expr = list(sympy_expr.free_symbols)
        lambdify_arg_symbols_ordered = [] 
        has_freq_in_lambdify_args = False
        temp_dep_symbols_for_ordering = [] 

        for s_obj in free_symbols_in_expr:
            s_name = str(s_obj)
            if s_name == 'freq':
                has_freq_in_lambdify_args = True
            elif s_obj not in self.ALLOWED_SYMPY_SYMBOLS: 
                temp_dep_symbols_for_ordering.append(s_obj)
        
        temp_dep_symbols_for_ordering.sort(key=str) 

        if has_freq_in_lambdify_args:
            freq_symbol_obj = next((s for s in free_symbols_in_expr if str(s) == 'freq'), sympy.Symbol('freq'))
            lambdify_arg_symbols_ordered.append(freq_symbol_obj)
        lambdify_arg_symbols_ordered.extend(temp_dep_symbols_for_ordering)
        
        logger.debug(f"  Expression '{internal_name}' will be called with lambdify args: {[str(s) for s in lambdify_arg_symbols_ordered]}")

        for arg_symbol_obj in lambdify_arg_symbols_ordered:
            arg_name_str = str(arg_symbol_obj) 

            if arg_name_str == 'freq':
                func_args_values.append(freq_hz_arr)
            else:
                dep_internal_name = None
                # Try to map arg_name_str (from SymPy symbol) back to a full internal name
                # This logic assumes arg_name_str is either a direct internal name (if it contains '.')
                # or an unqualified name that needs context (scope, owner) to resolve.
                if '.' in arg_name_str: # Potentially already fully qualified if from _InstanceProxy
                    if arg_name_str in dependencies_set and arg_name_str in self._parameter_context_map:
                        dep_internal_name = arg_name_str
                    else: # Should not happen if dependencies were correctly identified
                        raise ParameterScopeError(f"Symbol '{arg_name_str}' from expression for '{internal_name}' appears to be a qualified parameter name "
                                                  f"but is not a resolved dependency or not defined. Dependencies: {dependencies_set}")
                else: # Unqualified symbol, needs resolution based on current parameter's context
                    try:
                        dep_internal_name = self._resolve_symbol_to_internal_name(
                            arg_name_str, definition.scope, definition.owner_id
                        )
                        if dep_internal_name not in dependencies_set:
                             raise ParameterError(f"Internal inconsistency: Resolved dependency '{dep_internal_name}' for symbol '{arg_name_str}' "
                                                  f"is not in the pre-calculated dependencies set {dependencies_set} for '{internal_name}'.")
                    except ParameterScopeError as e:
                        raise ParameterError(f"Failed to map symbol '{arg_name_str}' to internal name for '{internal_name}': {e}") from e

                if not dep_internal_name: # Should be caught by above logic
                    raise ParameterError(f"Internal inconsistency: Lambdify argument symbol '{arg_name_str}' for parameter '{internal_name}' could not be mapped to a known dependency.")
                
                dep_declared_dimension_str = self.get_declared_dimension(dep_internal_name)
                logger.debug(f"    Recursing for dependency '{dep_internal_name}' (needed by '{internal_name}'), own declared_dim='{dep_declared_dimension_str}'")
                
                resolved_dep_qty = self.resolve_parameter(
                    dep_internal_name, freq_hz_arr, dep_declared_dimension_str, evaluation_context
                )
                
                # Lambdify expects numerical arrays, not Quantities
                func_args_values.append(resolved_dep_qty.magnitude)

        # 6. Call compiled function
        try:
            arg_shapes_str = [f"{arg.shape if isinstance(arg,np.ndarray) else type(arg)}" for arg in func_args_values]
            logger.debug(f"  Calling compiled func for '{internal_name}' with {len(func_args_values)} args. Shapes/Types: {arg_shapes_str}")
            
            current_err_settings = np.seterr(all='raise') 
            numerical_result = compiled_func(*func_args_values)
            np.seterr(**current_err_settings)

            if isinstance(numerical_result, np.ndarray) and (np.any(np.isnan(numerical_result)) or np.any(np.isinf(numerical_result))):
                logger.warning(f"Numerical evaluation of '{internal_name}' resulted in NaN/Inf: {numerical_result}")
            
            if not isinstance(numerical_result, np.ndarray): # Ensure it's an array
                numerical_result = np.array(numerical_result) 
            
            # Ensure result matches frequency array shape if it was scalar and freq was array
            if numerical_result.ndim == 0 and freq_hz_arr.ndim > 0 : 
                if freq_hz_arr.size > 0: # If freq_hz_arr is not empty
                    # Use result_type to handle complex numbers correctly if numerical_result is complex
                    # and ensure the output array is float/complex, not object.
                    dtype_res = np.result_type(numerical_result.item() if numerical_result.size > 0 else float, float)
                    numerical_result = np.full_like(freq_hz_arr, numerical_result.item(), dtype=dtype_res)
                elif freq_hz_arr.size == 0: # If freq_hz_arr is empty, result should be empty
                    dtype_res = np.result_type(numerical_result.item() if numerical_result.size > 0 else float, float)
                    numerical_result = np.array([], dtype=dtype_res)
                logger.debug(f"  Broadcasted scalar expression result for '{internal_name}' to match freq_hz_arr shape {freq_hz_arr.shape}.")
            elif numerical_result.ndim == 0 and freq_hz_arr.ndim == 1 and freq_hz_arr.size == 1: 
                numerical_result = numerical_result.reshape(1) # Make it (1,) if freq was (1,)


            logger.debug(f"  Numerical result for '{internal_name}' (shape: {numerical_result.shape}): {numerical_result if numerical_result.size < 5 else str(numerical_result[:5])+'...'}")
        
        except FloatingPointError as fpe: 
            np.seterr(**current_err_settings) 
            expr_str = definition.expression_str or "<N/A>"
            arg_details_str = self._format_arg_details_for_error(lambdify_arg_symbols_ordered, func_args_values)
            raise ParameterError(
                f"Numerical floating point error during evaluation of '{internal_name}' (expression: '{expr_str}'). "
                f"Args: [{arg_details_str}]. Error: {type(fpe).__name__} - {fpe}"
            ) from fpe
        except Exception as e:
            np.seterr(**current_err_settings) 
            expr_str = definition.expression_str or "<N/A>"
            arg_details_str = self._format_arg_details_for_error(lambdify_arg_symbols_ordered, func_args_values)

            raise ParameterError(
                f"Error evaluating compiled expression for '{internal_name}' (expression: '{expr_str}'). "
                f"Args given to lambdified function ({len(func_args_values)} total): [{arg_details_str}]. Error: {type(e).__name__} - {e}"
            ) from e

        # 7. Create Quantity using the *parameter's own declared dimension first*, then convert to target_dimension_str
        try:
            if isinstance(numerical_result, np.ndarray) and numerical_result.size == 0 and not numerical_result.dtype.kind in 'fc':
                numerical_result = numerical_result.astype(float) # Ensure empty arrays have float/complex type
            
            current_param_declared_dim_str = definition.declared_dimension_str
            intermediate_qty = Quantity(numerical_result, current_param_declared_dim_str)
            result_qty = intermediate_qty.to(target_dimension_str)
        except (pint.UndefinedUnitError, pint.DimensionalityError, TypeError, ValueError) as e:
            res_shape = numerical_result.shape if isinstance(numerical_result, np.ndarray) else 'scalar'
            res_dtype = numerical_result.dtype if isinstance(numerical_result, np.ndarray) else type(numerical_result)
            current_param_own_dim = definition.declared_dimension_str 
            raise ParameterError(
                f"Error processing evaluated result for '{internal_name}'. Numerical result (shape {res_shape}, dtype {res_dtype}) "
                f"with its declared dimension '{current_param_own_dim}' could not be converted to "
                f"target dimension '{target_dimension_str}': {e}"
            ) from e

        # 8. Cache and return
        evaluation_context[cache_key] = result_qty
        val_str = f"{result_qty:~P}" if (isinstance(result_qty.magnitude, np.ndarray) and result_qty.magnitude.size < 5) or not isinstance(result_qty.magnitude, np.ndarray) else f"{str(result_qty.magnitude[:5]) if isinstance(result_qty.magnitude, np.ndarray) else result_qty.magnitude}... {result_qty.units}"
        logger.debug(f"  Successfully resolved expression '{internal_name}' to quantity with target dimension. Value: {val_str}. Cached.")
        return result_qty
    
    def _format_arg_details_for_error(self, arg_symbols_ordered, arg_values) -> str:
							arg_details_list = []
							for i, val in enumerate(arg_values):
								arg_symbol_name = str(arg_symbols_ordered[i]) if i < len(arg_symbols_ordered) else "UNKNOWN_ARG"
								val_repr = f"shape {val.shape}" if isinstance(val, np.ndarray) else str(val)
								val_short = val_repr[:50] + '...' if len(val_repr) > 50 else val_repr
								arg_details_list.append(f"{arg_symbol_name}: {val_short}")
							return "; ".join(arg_details_list)