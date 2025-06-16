# --- tests/test_parameters.py ---

import pytest
import numpy as np
import pint
import sympy
import networkx as nx
from typing import List, Dict, Any
import re

# Imports from the source code
from src.rfsim_core import (SemanticValidationError, MnaInputError)
from src.rfsim_core.units import ureg, Quantity
from src.rfsim_core.parameters import (
    ParameterManager,
    ParameterDefinition,
    ParameterError,
    ParameterSyntaxError,
    ParameterDefinitionError,
    ParameterScopeError,
    CircularParameterDependencyError,
)
from src.rfsim_core.simulation.mna import MnaAssembler # For integration test
from src.rfsim_core.circuit_builder import CircuitBuilder
from tests.conftest import create_and_build_circuit

# Mock Component Registry for testing instance param dimension lookup context
# In a real setup, this might be handled differently, but sufficient for unit testing PM
MOCK_COMPONENT_REGISTRY = {
    "Resistor": {
        "params": {"resistance": "ohm"},
        "ports": ["p1", "p2"]
    },
    "Capacitor": {
         "params": {"capacitance": "farad"},
         "ports": ["p1", "p2"]
     },
    "VSource": {
        "params": {"voltage": "volt", "frequency": "hertz"},
         "ports": ["p1", "p2"]
    }
}

# Helper to create definitions more easily in tests
# Mimics what Parser/Builder would do regarding dimension population
# Helper function to create ParameterDefinition objects easily
def create_defs(param_dicts: List[Dict[str, Any]]) -> List[ParameterDefinition]:
    defs = []
    for p in param_dicts:
        # Determine dimension source
        dim = p.get('dimension') # Explicitly provided?
        if dim is None and 'constant' in p and p['scope'] == 'global':
            # Infer dimension for global constants
            try:
                q = ureg.Quantity(str(p['constant']))
                if q.dimensionless:
                    dim = "dimensionless"
                else:
                    dim = str(q.units) # Use inferred unit string
            except Exception as e:
                 # Propagate error or handle as needed for test setup
                 raise ValueError(f"Test setup error: Could not infer unit string for global constant '{p['name']}': {e}")

        if dim is None:
             # Raise error if dimension is still missing (e.g., global expr without dimension)
             if 'expression' in p and p['scope'] == 'global':
                  raise ValueError(f"Test setup error: Global expression '{p['name']}' missing 'dimension'.")
             # Instance params should always get dimension passed in test data
             # or default dimension for test component type if not provided
             raise ValueError(f"Test setup error: Dimension missing for parameter '{p.get('name', 'UNKNOWN')}' and could not be inferred.")

        defs.append(ParameterDefinition(
            name=p['name'],
            scope=p['scope'],
            owner_id=p.get('owner_id'),
            expression_str=p.get('expression'),
            constant_value_str=str(p['constant']) if 'constant' in p else None,
            declared_dimension_str=dim # Use determined dimension string
        ))
    return defs

# --- Test Fixtures ---
@pytest.fixture
def empty_pm():
    """Returns an empty ParameterManager."""
    return ParameterManager()

# --- Test ParameterDefinition ---
def test_parameter_definition_validation():
    # Test valid cases (owner_id=None for global is now handled by default)
    ParameterDefinition(name="R", scope="global", constant_value_str="10 ohm", declared_dimension_str="ohm")
    ParameterDefinition(name="C", scope="instance", owner_id="C1", expression_str="1/(2*pi*freq*Z)", declared_dimension_str="farad")

    # Test invalid cases caught by __post_init__
    with pytest.raises(ValueError, match="must have a valid declared_dimension_str string."):
         ParameterDefinition(name="R", scope="global", constant_value_str="10 ohm") # Missing dimension
    with pytest.raises(ValueError, match="Invalid scope"):
         ParameterDefinition(name="R", scope="invalid", constant_value_str="10 ohm", declared_dimension_str="ohm")
    with pytest.raises(ValueError, match="must have either expression_str or constant_value_str"):
         ParameterDefinition(name="R", scope="global", declared_dimension_str="ohm")
    with pytest.raises(ValueError, match="cannot have both expression_str and constant_value_str"):
         ParameterDefinition(name="R", scope="global", expression_str="1", constant_value_str="1", declared_dimension_str="dimensionless")
    with pytest.raises(ValueError, match="Instance-scoped parameter .* must have a valid non-empty string owner_id"):
         # Test None owner_id for instance scope
         ParameterDefinition(name="R", scope="instance", owner_id=None, constant_value_str="1", declared_dimension_str="dimensionless")
    with pytest.raises(ValueError, match="Instance-scoped parameter .* must have a valid non-empty string owner_id"):
         # Test empty string owner_id for instance scope
         ParameterDefinition(name="R", scope="instance", owner_id="", constant_value_str="1", declared_dimension_str="dimensionless")
    with pytest.raises(ValueError, match="Global-scoped parameter .* cannot have an owner_id"):
         # Test owner_id provided for global scope
         ParameterDefinition(name="R", scope="global", owner_id="C1", constant_value_str="1", declared_dimension_str="dimensionless")


# --- Test ParameterManager Initialization and Build ---

class TestParameterManagerBuildSuccess:

    def test_build_empty(self, empty_pm):
        """Test building with no definitions."""
        pm = empty_pm
        pm.build()
        assert pm._build_complete
        assert not pm._parameter_context_map
        assert pm._dependency_graph.number_of_nodes() == 0
        assert not pm._parsed_constants
        assert pm.get_all_internal_names() == []

    def test_build_only_constants(self, empty_pm):
        """Test building with only constant parameters."""
        pm = empty_pm
        defs = create_defs([
            {'name': 'R0', 'scope': 'global', 'constant': '50 ohm', 'dimension': 'ohm'},
            {'name': 'L_val', 'scope': 'global', 'constant': '10 nH', 'dimension': 'henry'},
            {'name': 'resistance', 'scope': 'instance', 'owner_id': 'R1', 'constant': '1 kohm', 'dimension': 'ohm'},
        ])
        pm.add_definitions(defs)
        pm.build()

        assert pm._build_complete
        assert len(pm._parameter_context_map) == 3
        assert len(pm._parsed_constants) == 3
        assert pm._dependency_graph.number_of_nodes() == 3
        assert pm._dependency_graph.number_of_edges() == 0

        # Check context map
        assert '_rfsim_global_.R0' in pm._parameter_context_map
        assert pm._parameter_context_map['_rfsim_global_.R0']['declared_dimension'] == 'ohm'
        assert pm._parameter_context_map['_rfsim_global_.R0']['dependencies'] == set()
        assert pm._parameter_context_map['_rfsim_global_.R0']['sympy_expr'] is None

        assert '_rfsim_global_.L_val' in pm._parameter_context_map
        assert pm._parameter_context_map['_rfsim_global_.L_val']['declared_dimension'] == 'henry' # Dimensionality of henry
        assert 'R1.resistance' in pm._parameter_context_map

        # Check constant values
        assert pm.get_constant_value('_rfsim_global_.R0') == Quantity(50.0, 'ohm')
        assert pm.get_constant_value('_rfsim_global_.L_val') == Quantity(10.0, 'nH')
        assert pm.get_constant_value('R1.resistance') == Quantity(1.0, 'kohm')

        # Check dependencies via accessor
        assert pm.get_dependencies('_rfsim_global_.R0') == set()
        assert pm.get_dependencies('R1.resistance') == set()

        # Check graph directly (optional)
        assert set(pm._dependency_graph.nodes) == {'_rfsim_global_.R0', '_rfsim_global_.L_val', 'R1.resistance'}
        assert list(pm._dependency_graph.edges) == []

    def test_build_simple_expression_global_dep(self, empty_pm):
        """Test building with an expression depending on a global constant."""
        pm = empty_pm
        defs = create_defs([
            {'name': 'freq_ghz', 'scope': 'global', 'constant': '1 GHz', 'dimension': 'hertz'},
            {'name': 'cap_val', 'scope': 'global', 'expression': '1 / (2 * pi * freq * 50)', 'dimension': 'farad'},
            {'name': 'res_val', 'scope': 'global', 'expression': 'R0', 'dimension': 'ohm'}, # Depends on R0
            {'name': 'R0', 'scope': 'global', 'constant': '100 ohm', 'dimension': 'ohm'},
        ])
        pm.add_definitions(defs)
        pm.build()

        assert pm._build_complete
        assert len(pm._parameter_context_map) == 4
        assert len(pm._parsed_constants) == 2 # freq_ghz, R0
        # NEW assertion (Task 5.2 integration expectation)
        assert len(pm._compiled_functions) == 2 # Check that cap_val and res_val were compiled
        # Optional: Add more specific checks
        assert '_rfsim_global_.cap_val' in pm._compiled_functions
        assert '_rfsim_global_.res_val' in pm._compiled_functions
        assert callable(pm._compiled_functions['_rfsim_global_.cap_val'])
        assert callable(pm._compiled_functions['_rfsim_global_.res_val'])

        assert '_rfsim_global_.cap_val' in pm._parameter_context_map
        assert pm._parameter_context_map['_rfsim_global_.cap_val']['dependencies'] == {'freq'} # Only depends on reserved 'freq'
        assert isinstance(pm._parameter_context_map['_rfsim_global_.cap_val']['sympy_expr'], sympy.Expr)

        assert '_rfsim_global_.res_val' in pm._parameter_context_map
        assert pm._parameter_context_map['_rfsim_global_.res_val']['dependencies'] == {'_rfsim_global_.R0'} # Resolved correctly
        assert isinstance(pm._parameter_context_map['_rfsim_global_.res_val']['sympy_expr'], sympy.Expr)

        assert pm.get_dependencies('_rfsim_global_.freq_ghz') == set()
        assert pm.get_dependencies('_rfsim_global_.R0') == set()
        assert pm.get_dependencies('_rfsim_global_.cap_val') == {'freq'}
        assert pm.get_dependencies('_rfsim_global_.res_val') == {'_rfsim_global_.R0'}

        # Check graph
        assert pm._dependency_graph.has_node('freq')
        assert pm._dependency_graph.has_edge('_rfsim_global_.cap_val', 'freq')
        assert pm._dependency_graph.has_edge('_rfsim_global_.res_val', '_rfsim_global_.R0')
        assert pm._dependency_graph.number_of_edges() == 2

    def test_build_expression_instance_and_global_dep(self, empty_pm):
        """Test expression in instance param depending on another instance param and a _rfsim_global_."""
        pm = empty_pm
        defs = create_defs([
            {'name': 'global_gain', 'scope': 'global', 'constant': '10', 'dimension': 'dimensionless'}, # Fix 3
            {'name': 'base_res', 'scope': 'instance', 'owner_id': 'R1', 'constant': '100 ohm', 'dimension': 'ohm'},
            {'name': 'scaled_res', 'scope': 'instance', 'owner_id': 'R1', 'expression': 'base_res * global_gain', 'dimension': 'ohm'},
            {'name': 'other_res', 'scope': 'instance', 'owner_id': 'R2', 'constant': '50 ohm', 'dimension': 'ohm'},
             # R2 depends on R1's scaled_res - *Explicitly* name it
            {'name': 'dep_res', 'scope': 'instance', 'owner_id': 'R2', 'expression': 'R1.scaled_res / 2', 'dimension': 'ohm'},
        ])
        pm.add_definitions(defs)
        pm.build()

        assert pm._build_complete
        assert len(pm._parameter_context_map) == 5

        # Check dependencies for R1.scaled_res
        assert pm.get_dependencies('R1.scaled_res') == {'R1.base_res', '_rfsim_global_.global_gain'}

        # Check dependencies for R2.dep_res (now handles explicit R1.scaled_res)
        assert pm.get_dependencies('R2.dep_res') == {'R1.scaled_res'}

        # Check graph
        assert pm._dependency_graph.has_edge('R1.scaled_res', 'R1.base_res')
        assert pm._dependency_graph.has_edge('R1.scaled_res', '_rfsim_global_.global_gain')
        assert pm._dependency_graph.has_edge('R2.dep_res', 'R1.scaled_res')
        assert not pm._dependency_graph.has_edge('R2.dep_res', 'R2.other_res') # No longer depends on this
        assert pm._dependency_graph.number_of_edges() == 3

    def test_build_sympy_evaluate_false(self, empty_pm):
        """Test that sympy parsing with evaluate=False prevents premature evaluation."""
        pm = empty_pm
        defs = create_defs([
            {'name': 'R1.resistance', 'scope': 'global', 'constant': '100 ohm', 'dimension': 'ohm'}, # Name with '.'
            {'name': 'test', 'scope': 'global', 'expression': 'R1.resistance / 2', 'dimension': 'ohm'},
        ])
        pm.add_definitions(defs)
        # Expect this to fail because 'R1.resistance' is not a valid *global* name
        with pytest.raises(ParameterDefinitionError, match="Invalid global name: 'R1.resistance'"):
            pm.build()

        # Try again with valid names
        pm = ParameterManager()
        defs = create_defs([
             {'name': 'r_one', 'scope': 'global', 'constant': '100 ohm', 'dimension': 'ohm'},
             {'name': 'test', 'scope': 'global', 'expression': 'r_one / 2', 'dimension': 'ohm'},
        ])
        pm.add_definitions(defs)
        pm.build()
        # Check that the parsed sympy expression still contains the symbol 'r_one'
        sympy_expr = pm._parameter_context_map['_rfsim_global_.test']['sympy_expr']
        assert isinstance(sympy_expr, sympy.Expr)
        assert sympy.Symbol('r_one') in sympy_expr.free_symbols


    def test_scope_resolution_instance_over_global(self, empty_pm):
        """Test that instance parameters shadow global ones during resolution."""
        pm = empty_pm
        defs = create_defs([
            {'name': 'X', 'scope': 'global', 'constant': '10 V', 'dimension': 'volt'},
            {'name': 'X', 'scope': 'instance', 'owner_id': 'Comp1', 'constant': '5 V', 'dimension': 'volt'},
            {'name': 'Y', 'scope': 'instance', 'owner_id': 'Comp1', 'expression': 'X * 2', 'dimension': 'volt'}, # Should use Comp1.X
            {'name': 'Z', 'scope': 'global', 'expression': 'X * 3', 'dimension': 'volt'}, # Should use _rfsim_global_.X
        ])
        pm.add_definitions(defs)
        pm.build()

        assert pm.get_dependencies('Comp1.Y') == {'Comp1.X'}
        assert pm.get_dependencies('_rfsim_global_.Z') == {'_rfsim_global_.X'}

        assert pm._dependency_graph.has_edge('Comp1.Y', 'Comp1.X')
        assert pm._dependency_graph.has_edge('_rfsim_global_.Z', '_rfsim_global_.X')
        assert not pm._dependency_graph.has_edge('Comp1.Y', '_rfsim_global_.X')
        assert not pm._dependency_graph.has_edge('_rfsim_global_.Z', 'Comp1.X')

# --- Test ParameterManager Error Handling During Build ---

class TestParameterManagerBuildErrors:

    def test_error_duplicate_internal_name_global(self, empty_pm):
        pm = empty_pm
        defs = create_defs([
            {'name': 'R0', 'scope': 'global', 'constant': '50 ohm', 'dimension': 'ohm'},
            {'name': 'R0', 'scope': 'global', 'constant': '100 ohm', 'dimension': 'ohm'},
        ])
        pm.add_definitions(defs)
        with pytest.raises(ParameterDefinitionError, match="Duplicate internal name '_rfsim_global_.R0'"):
            pm.build()

    def test_error_duplicate_internal_name_instance(self, empty_pm):
        pm = empty_pm
        defs = create_defs([
            {'name': 'val', 'scope': 'instance', 'owner_id':'R1', 'constant': '50 ohm', 'dimension': 'ohm'},
            {'name': 'val', 'scope': 'instance', 'owner_id':'R1', 'constant': '100 ohm', 'dimension': 'ohm'},
        ])
        pm.add_definitions(defs)
        with pytest.raises(ParameterDefinitionError, match="Duplicate internal name 'R1.val'"):
            pm.build()

    def test_error_invalid_dimension_string(self, empty_pm):
        pm = empty_pm
        # Test data provides explicit dimension string
        defs = create_defs([
            {'name': 'R0', 'scope': 'global', 'constant': '50 ohm', 'dimension': 'not_a_dimension'}, # Invalid dimension
        ])
        pm.add_definitions(defs)
        # --- MODIFICATION ---
        # Update the regex to match the actual error message format
        expected_error_msg_regex = r"Invalid declared_dimension_str 'not_a_dimension' for parameter '_rfsim_global_.R0':"
        with pytest.raises(ParameterDefinitionError, match=expected_error_msg_regex):
            pm.build()

    def test_error_invalid_constant_value(self, empty_pm):
        """Test error when a constant string cannot be parsed by Pint."""
        pm = empty_pm
        defs = create_defs([
            {'name': 'R0', 'scope': 'global', 'constant': '50 qqq', 'dimension': 'ohm'}, # Invalid unit
        ])
        pm.add_definitions(defs)
        # This error now happens during _parse_and_cache_constants
        with pytest.raises(ParameterDefinitionError, match="Constant value string '50 qqq' for parameter '_rfsim_global_.R0'"):
            pm.build()

    def test_error_expression_syntax_error(self, empty_pm):
        pm = empty_pm
        defs = create_defs([
            {'name': 'R0', 'scope': 'global', 'constant': '50 ohm', 'dimension': 'ohm'},
            {'name': 'X', 'scope': 'global', 'expression': 'R0 * (2 + )', 'dimension': 'ohm'}, # Syntax error
        ])
        pm.add_definitions(defs)
        with pytest.raises(ParameterSyntaxError, match="Syntax error parsing expression for '_rfsim_global_.X'.*'R0 \* \(2 \+ \)'"):
            pm.build()

    def test_error_expression_dependency_not_found(self, empty_pm):
        pm = empty_pm
        defs = create_defs([
            {'name': 'R0', 'scope': 'global', 'constant': '50 ohm', 'dimension': 'ohm'},
            {'name': 'X', 'scope': 'global', 'expression': 'R0 * R1', 'dimension': 'ohm'}, # R1 not defined
        ])
        pm.add_definitions(defs)
        with pytest.raises(ParameterScopeError, match=re.escape("Dependency resolution failed for expression parameter '_rfsim_global_.X' ('R0 * R1'): Parameter symbol 'R1' referenced in an expression could not be resolved. Context: scope='global'. Searched for <N/A for instance> and '_rfsim_global_.R1'.")):
            pm.build()

    def test_error_instance_dependency_not_found(self, empty_pm):
        pm = empty_pm
        defs = create_defs([
            {'name': 'resistance', 'scope': 'instance', 'owner_id':'R1', 'constant': '50 ohm', 'dimension': 'ohm'},
            {'name': 'scaled', 'scope': 'instance', 'owner_id':'R1', 'expression': 'resistance * gain', 'dimension': 'ohm'}, # gain not defined
        ])
        pm.add_definitions(defs)
        with pytest.raises(ParameterScopeError, match=re.escape("Dependency resolution failed for expression parameter 'R1.scaled' ('resistance * gain'): Parameter symbol 'gain' referenced in an expression could not be resolved. Context: scope='instance', owner='R1'. Searched for 'R1.gain' and '_rfsim_global_.gain'.")):
            pm.build()

    def test_error_explicit_instance_dependency_not_found(self, empty_pm):
        """Test when an explicit R1.gain is used but R1.gain doesn't exist."""
        pm = empty_pm
        defs = create_defs([
            {'name': 'resistance', 'scope': 'instance', 'owner_id':'R1', 'constant': '50 ohm', 'dimension': 'ohm'},
            {'name': 'scaled', 'scope': 'instance', 'owner_id':'R1', 'expression': 'resistance * R2.gain', 'dimension': 'ohm'}, # R2.gain not defined
        ])
        pm.add_definitions(defs)
        with pytest.raises(ParameterSyntaxError, match=re.escape("Syntax error parsing expression for 'R1.scaled': 'resistance * R2.gain'. Error: AttributeError(''Symbol' object has no attribute 'gain'')")):
             pm.build()

    def test_error_circular_dependency_direct(self, empty_pm):
        pm = empty_pm
        defs = create_defs([
            {'name': 'A', 'scope': 'global', 'expression': 'B', 'dimension': 'dimensionless'}, # Fix 3
            {'name': 'B', 'scope': 'global', 'expression': 'A', 'dimension': 'dimensionless'}, # Fix 3
        ])
        pm.add_definitions(defs)
        with pytest.raises(CircularParameterDependencyError, match="Circular dependency detected"):
            pm.build()

    def test_error_circular_dependency_indirect(self, empty_pm):
        pm = empty_pm
        defs = create_defs([
            {'name': 'A', 'scope': 'global', 'expression': 'B * 2', 'dimension': 'dimensionless'}, # Fix 3
            {'name': 'B', 'scope': 'global', 'expression': 'C + 1', 'dimension': 'dimensionless'}, # Fix 3
            {'name': 'C', 'scope': 'global', 'expression': 'A / 3', 'dimension': 'dimensionless'}, # Fix 3
            {'name': 'D', 'scope': 'global', 'constant': '10', 'dimension': 'dimensionless'}, # Fix 3 - Unrelated
        ])
        pm.add_definitions(defs)
        # The exact cycle order reported by networkx might vary
        with pytest.raises(CircularParameterDependencyError, match="Circular dependency detected: _rfsim_global_.[ABC] -> _rfsim_global_.[ABC] -> _rfsim_global_.[ABC] -> _rfsim_global_.[ABC]"):
             pm.build()

    def test_error_circular_dependency_instance_global(self, empty_pm):
        pm = empty_pm
        defs = create_defs([
            {'name': 'A', 'scope': 'global', 'expression': 'R1.X', 'dimension': 'dimensionless'}, # Fix 3
            {'name': 'X', 'scope': 'instance', 'owner_id': 'R1', 'expression': 'A', 'dimension': 'dimensionless'}, # Fix 3
        ])
        pm.add_definitions(defs)
        with pytest.raises(CircularParameterDependencyError, match="Circular dependency detected:"):
            pm.build()

    def test_error_invalid_name_global(self, empty_pm):
        pm = empty_pm
        defs = create_defs([
            {'name': 'bad.name', 'scope': 'global', 'constant': '1', 'dimension': 'dimensionless'}, # Fix 3
        ])
        pm.add_definitions(defs)
        with pytest.raises(ParameterDefinitionError, match="Invalid global name: 'bad.name'"):
            pm.build()

    def test_error_invalid_name_instance_base(self, empty_pm):
        pm = empty_pm
        defs = create_defs([
            {'name': 'bad.name', 'scope': 'instance', 'owner_id': 'R1', 'constant': '1', 'dimension': 'dimensionless'}, # Fix 3
        ])
        pm.add_definitions(defs)
        with pytest.raises(ParameterDefinitionError, match="Invalid instance name: 'bad.name' for owner 'R1'"):
            pm.build()

    def test_error_use_before_build(self, empty_pm):
        """Test accessing methods before build() is called."""
        pm = empty_pm
        defs = create_defs([{'name': 'R0', 'scope': 'global', 'constant': '50 ohm', 'dimension': 'ohm'}])
        pm.add_definitions(defs) # Added but not built

        with pytest.raises(ParameterError, match="ParameterManager has not been built."):
            pm.get_all_internal_names()
        with pytest.raises(ParameterError, match="ParameterManager has not been built."):
            pm.get_parameter_definition("_rfsim_global_.R0")
        with pytest.raises(ParameterError, match="ParameterManager has not been built."):
            pm.get_dependencies("_rfsim_global_.R0")
        with pytest.raises(ParameterError, match="ParameterManager has not been built."):
            pm.is_constant("_rfsim_global_.R0")
        with pytest.raises(ParameterError, match="ParameterManager has not been built."):
            pm.get_constant_value("_rfsim_global_.R0")
        with pytest.raises(ParameterError, match="ParameterManager has not been built."):
            # Placeholder methods also need the check
            pm.get_compiled_function("_rfsim_global_.R0")
        with pytest.raises(ParameterError, match="ParameterManager has not been built."):
           pm.resolve_parameter("_rfsim_global_.R0", np.array([1e9]), "ohm", {})


# --- Test Accessors after successful build ---

class TestParameterManagerAccessors:

    @pytest.fixture(scope="class")
    def built_pm(self):
        """Provides a ParameterManager instance already built with a mix of params."""
        pm = ParameterManager()
        defs = create_defs([
            {'name': 'R0', 'scope': 'global', 'constant': '50 ohm', 'dimension': 'ohm'},
            {'name': 'L_val', 'scope': 'global', 'constant': '10 nH', 'dimension': 'henry'},
            {'name': 'Scale', 'scope': 'global', 'expression': 'L_val * 1e9', 'dimension': 'dimensionless'}, # Fix 3, dimless scale from nH
            {'name': 'resistance', 'scope': 'instance', 'owner_id': 'R1', 'constant': '1 kohm', 'dimension': 'ohm'},
            {'name': 'current', 'scope': 'instance', 'owner_id': 'R1', 'expression': '1 / resistance', 'dimension': 'siemens'}, # Use siemens directly
        ])
        pm.add_definitions(defs)
        pm.build()
        return pm

    def test_get_all_internal_names(self, built_pm):
        names = built_pm.get_all_internal_names()
        assert isinstance(names, list)
        assert set(names) == {'_rfsim_global_.R0', '_rfsim_global_.L_val', '_rfsim_global_.Scale', 'R1.resistance', 'R1.current'}

    def test_get_parameter_definition(self, built_pm):
        defn = built_pm.get_parameter_definition('_rfsim_global_.R0')
        assert isinstance(defn, ParameterDefinition)
        assert defn.name == 'R0'
        assert defn.scope == 'global'
        assert defn.constant_value_str == '50 ohm'
        assert defn.owner_id is None # Check owner_id default

        defn = built_pm.get_parameter_definition('R1.current')
        assert defn.name == 'current'
        assert defn.scope == 'instance'
        assert defn.owner_id == 'R1'
        assert defn.expression_str == '1 / resistance'

        with pytest.raises(ParameterScopeError, match="not found"):
            built_pm.get_parameter_definition('nonexistent.param')

    def test_get_declared_dimension(self, built_pm):
        assert built_pm.get_declared_dimension('_rfsim_global_.R0') == 'ohm' # ohm
        assert built_pm.get_declared_dimension('_rfsim_global_.L_val') == 'henry' # henry
        assert built_pm.get_declared_dimension('_rfsim_global_.Scale') == 'dimensionless' # Dimensionless
        assert built_pm.get_declared_dimension('R1.resistance') == 'ohm' # ohm
        # Check admittance dimension string formatting by Pint

        assert built_pm.get_declared_dimension('R1.current') == 'siemens' # siemens

        with pytest.raises(ParameterScopeError, match="not found"):
            built_pm.get_declared_dimension('nonexistent.param')

    def test_get_dependencies(self, built_pm):
        assert built_pm.get_dependencies('_rfsim_global_.R0') == set()
        assert built_pm.get_dependencies('_rfsim_global_.L_val') == set()
        assert built_pm.get_dependencies('_rfsim_global_.Scale') == {'_rfsim_global_.L_val'}
        assert built_pm.get_dependencies('R1.resistance') == set()
        assert built_pm.get_dependencies('R1.current') == {'R1.resistance'}

        with pytest.raises(ParameterScopeError, match="not found"):
            built_pm.get_dependencies('nonexistent.param')

    def test_is_constant(self, built_pm):
        assert built_pm.is_constant('_rfsim_global_.R0') is True
        assert built_pm.is_constant('_rfsim_global_.L_val') is True
        assert built_pm.is_constant('_rfsim_global_.Scale') is False # Expression
        assert built_pm.is_constant('R1.resistance') is True
        assert built_pm.is_constant('R1.current') is False # Expression

        with pytest.raises(ParameterScopeError, match="not found"):
             built_pm.is_constant('nonexistent.param')

    def test_get_constant_value(self, built_pm):
        assert built_pm.get_constant_value('_rfsim_global_.R0') == Quantity(50, 'ohm')
        assert built_pm.get_constant_value('_rfsim_global_.L_val') == Quantity(10, 'nH')
        assert built_pm.get_constant_value('R1.resistance') == Quantity(1, 'kohm')

        # Test error for expression param
        with pytest.raises(ParameterError, match=re.escape("Parameter '_rfsim_global_.Scale' is an expression ('L_val * 1e9') and cannot be retrieved as a simple constant value. Use resolve_parameter().")): # More specific match
            built_pm.get_constant_value('_rfsim_global_.Scale')
        with pytest.raises(ParameterError, match=re.escape("R1.current' is an expression ('1 / resistance') and cannot be retrieved as a simple constant value. Use resolve_parameter().")): # More specific match
            built_pm.get_constant_value('R1.current')

        # Test error for non-existent param
        with pytest.raises(ParameterError, match="not found"):
            built_pm.get_constant_value('nonexistent.param')


class TestParameterManagerResolveParameter:

    @pytest.fixture(scope="class")
    def comprehensive_built_pm(self):
        pm = ParameterManager()
        defs = create_defs([
            # Constants
            {'name': 'R_const', 'scope': 'global', 'constant': '50 ohm', 'dimension': 'ohm'},
            {'name': 'L_const_nH', 'scope': 'global', 'constant': '10 nH', 'dimension': 'henry'},
            {'name': 'Val_dimless', 'scope': 'global', 'constant': '2.5', 'dimension': 'dimensionless'},
            {'name': 'C_ref_R', 'scope': 'global', 'constant': 'R_const', 'dimension': 'ohm'}, 

            # Expressions
            {'name': 'Freq_dep_X', 'scope': 'global', 'expression': 'freq * 1e-12', 'dimension': 'dimensionless'},
            {'name': 'Calc_C', 'scope': 'global', 'expression': '1 / (2 * pi * freq * R_const)', 'dimension': 'farad'},
            {'name': 'L_scaled', 'scope': 'global', 'expression': 'L_const_nH * Val_dimless', 'dimension': 'henry'},
            {'name': 'Log_val', 'scope': 'global', 'expression': 'log10(freq)', 'dimension': 'dimensionless'},

            # Instance parameters
            {'name': 'r_inst_val', 'scope': 'instance', 'owner_id': 'R1', 'constant': '1 kohm', 'dimension': 'ohm'},
            {'name': 'c_inst_expr', 'scope': 'instance', 'owner_id': 'C1', 'expression': 'Calc_C / 2', 'dimension': 'farad'},
            {'name': 'gain', 'scope': 'instance', 'owner_id': 'AMP1', 'expression': 'sqrt(R1.r_inst_val/R_const)', 'dimension': 'dimensionless'},

            # For testing error in expression
            # REMOVE problematic Div_by_zero_expr for this general fixture
            # {'name': 'Div_by_zero_expr', 'scope': 'global', 'expression': '1/0', 'dimension': 'dimensionless'}, 
            {'name': 'Div_by_freq_expr', 'scope': 'global', 'expression': '1/freq', 'dimension': 'siemens'}, 

            # For testing dependency dimension lookup
            {'name': 'D1', 'scope': 'global', 'constant': '10 pF', 'dimension': 'farad'},
            {'name': 'D2_expr', 'scope': 'global', 'expression': 'D1 * 2', 'dimension': 'farad'}, 
            {'name': 'D3_expr', 'scope': 'global', 'expression': 'D2_expr * 3', 'dimension': 'farad'}, 
        ])
        pm.add_definitions(defs)
        pm.build() # This should now pass
        return pm

    # --- Tests for Constants ---
    def test_resolve_simple_constant(self, comprehensive_built_pm):
        pm = comprehensive_built_pm
        freq = np.array([1e9])
        context = {}
        q = pm.resolve_parameter("_rfsim_global_.R_const", freq, "ohm", context)
        assert isinstance(q, Quantity)
        assert q.magnitude == pytest.approx(np.array([50.0]))
        assert q.check("[resistance]") # Check base dimension

        q_kohm = pm.resolve_parameter("_rfsim_global_.R_const", freq, "kiloohm", context)
        assert q_kohm.magnitude == pytest.approx(np.array([0.05]))
        assert q_kohm.check("[resistance]")

    def test_resolve_constant_broadcast(self, comprehensive_built_pm):
        pm = comprehensive_built_pm
        freq = np.array([1e9, 2e9, 3e9])
        context = {}
        q = pm.resolve_parameter("_rfsim_global_.Val_dimless", freq, "dimensionless", context)
        assert q.magnitude.shape == freq.shape
        assert np.all(q.magnitude == pytest.approx(2.5))

    def test_resolve_constant_ref(self, comprehensive_built_pm):
        pm = comprehensive_built_pm
        freq = np.array([1e9])
        context = {}
        q = pm.resolve_parameter("_rfsim_global_.C_ref_R", freq, "ohm", context)
        assert q.magnitude == pytest.approx(np.array([50.0]))

    def test_resolve_constant_incompatible_target_dim(self, comprehensive_built_pm):
        pm = comprehensive_built_pm
        freq = np.array([1e9])
        context = {}
        # Adjusted regex to be more robust to pint's error message formatting
        with pytest.raises(pint.DimensionalityError, match=r"Cannot convert from 'ohm' .* to 'farad'"):
            pm.resolve_parameter("_rfsim_global_.R_const", freq, "farad", context)

    def test_resolve_constant_empty_freq(self, comprehensive_built_pm):
        pm = comprehensive_built_pm
        freq = np.array([], dtype=float)
        context = {}
        q = pm.resolve_parameter("_rfsim_global_.R_const", freq, "ohm", context)
        assert isinstance(q, Quantity)
        assert q.magnitude.shape == (0,)
        assert q.check("[resistance]")

    # --- Tests for Expressions ---
    def test_resolve_simple_expression(self, comprehensive_built_pm):
        pm = comprehensive_built_pm
        # Example: 'Val_dimless * 2' - needs new param
        # For now, let's use an existing expression: _rfsim_global_.L_scaled = L_const_nH * Val_dimless
        freq = np.array([1e9])
        context = {}
        # L_const_nH = 10 nH, Val_dimless = 2.5
        # L_scaled = 10 nH * 2.5 = 25 nH
        q = pm.resolve_parameter("_rfsim_global_.L_scaled", freq, "nanohenry", context)
        assert q.magnitude == pytest.approx(np.array([25.0]))
        assert q.check("[inductance]")

        q_H = pm.resolve_parameter("_rfsim_global_.L_scaled", freq, "henry", context)
        assert q_H.magnitude == pytest.approx(np.array([25e-9]))

    def test_resolve_freq_dependent_expression(self, comprehensive_built_pm):
        pm = comprehensive_built_pm
        freq = np.array([1e9]) # 1 GHz
        context = {}
        # _rfsim_global_.Freq_dep_X = freq * 1e-12
        q = pm.resolve_parameter("_rfsim_global_.Freq_dep_X", freq, "dimensionless", context)
        assert q.magnitude == pytest.approx(np.array([1e9 * 1e-12])) # 1e-3

        freq_multi = np.array([1e9, 2e9])
        context = {}
        q_multi = pm.resolve_parameter("_rfsim_global_.Freq_dep_X", freq_multi, "dimensionless", context)
        assert q_multi.magnitude == pytest.approx(np.array([1e-3, 2e-3]))

    def test_resolve_expression_with_deps(self, comprehensive_built_pm):
        pm = comprehensive_built_pm
        freq = np.array([1e9])
        context = {}
        # _rfsim_global_.Calc_C = 1 / (2 * pi * freq * R_const)
        # R_const = 50 ohm
        expected_c = 1 / (2 * np.pi * 1e9 * 50)
        q = pm.resolve_parameter("_rfsim_global_.Calc_C", freq, "farad", context)
        assert q.magnitude == pytest.approx(np.array([expected_c]))

    def test_resolve_expression_numpy_funcs(self, comprehensive_built_pm):
        pm = comprehensive_built_pm
        freq = np.array([1e8]) # 100 MHz for log10 to be 8
        context = {}
        q = pm.resolve_parameter("_rfsim_global_.Log_val", freq, "dimensionless", context)
        assert q.magnitude == pytest.approx(np.array([8.0]))

    def test_resolve_expression_empty_freq(self, comprehensive_built_pm):
        pm = comprehensive_built_pm
        freq = np.array([], dtype=float)
        context = {}
        q = pm.resolve_parameter("_rfsim_global_.Freq_dep_X", freq, "dimensionless", context)
        assert isinstance(q, Quantity)
        assert q.magnitude.shape == (0,)

    # --- Tests for Instance Parameters and Scoping in Resolution ---
    def test_resolve_instance_constant(self, comprehensive_built_pm):
        pm = comprehensive_built_pm
        freq = np.array([1e9])
        context = {}
        q = pm.resolve_parameter("R1.r_inst_val", freq, "kohm", context)
        assert q.magnitude == pytest.approx(np.array([1.0]))

    def test_resolve_instance_expression(self, comprehensive_built_pm):
        pm = comprehensive_built_pm
        freq = np.array([1e9])
        context = {}
        # C1.c_inst_expr = _rfsim_global_.Calc_C / 2
        # _rfsim_global_.Calc_C = 1 / (2 * pi * freq * R_const)
        # R_const = 50 ohm
        calc_c_val = 1 / (2 * np.pi * 1e9 * 50)
        expected_c_inst = calc_c_val / 2
        q = pm.resolve_parameter("C1.c_inst_expr", freq, "farad", context)
        assert q.magnitude == pytest.approx(np.array([expected_c_inst]))

    def test_resolve_instance_expr_multi_scope_deps(self, comprehensive_built_pm):
        pm = comprehensive_built_pm
        freq = np.array([1e9]) # freq doesn't matter for this expression
        context = {}
        # AMP1.gain = sqrt(R1.r_inst_val / _rfsim_global_.R_const)
        # R1.r_inst_val = 1 kohm = 1000 ohm
        # _rfsim_global_.R_const = 50 ohm
        expected_gain = np.sqrt(1000 / 50)
        q = pm.resolve_parameter("AMP1.gain", freq, "dimensionless", context)
        assert q.magnitude == pytest.approx(np.array([expected_gain]))

    # --- Test Recursive Dependency Dimension Handling ---
    def test_resolve_recursive_dependency_dimensions(self, comprehensive_built_pm):
        pm = comprehensive_built_pm
        freq = np.array([1e9])
        context = {}

        # Mock resolve_parameter to check intermediate calls
        original_resolve = pm.resolve_parameter
        resolved_interమediates = {}

        def mocked_resolve(internal_name, freq_hz, target_dimension_str, eval_context):
            # Store what target_dimension_str it was called with
            resolved_interమediates[internal_name] = target_dimension_str
            # Call the original to proceed, but prevent infinite recursion for this mock
            # For this specific test, we are interested in the *target_dimension_str* used,
            # not necessarily preventing the full computation if we make this mock too simple.
            # A more robust mock would only intercept specific calls.
            # For now, let's assume this structure is enough to see the target_dim.
            # To avoid issues with re-entry, check if we are already in a mocked call for the same name
            # or simply call original_resolve
            return original_resolve(internal_name, freq_hz, target_dimension_str, eval_context)

        pm.resolve_parameter = mocked_resolve
        try:
            # D3_expr = D2_expr * 3; D2_expr = D1 * 2; D1 = 10 pF
            # Resolve D3_expr, target_dimension_str 'farad'
            _ = pm.resolve_parameter("_rfsim_global_.D3_expr", freq, "farad", context)

            # Check target dimensions used for dependencies:
            # When resolving D3_expr, it needs D2_expr. D2_expr's declared_dimension is 'farad'.
            assert resolved_interమediates.get("_rfsim_global_.D2_expr") == "farad"
            # When D2_expr was resolved (possibly initiated by D3_expr's call), it needed D1.
            # D1's declared_dimension is 'farad'.
            # This might be called multiple times if context is not passed correctly or mock is simple.
            # The key is that *a* call to resolve D1 eventually happened with 'farad'.
            assert resolved_interమediates.get("_rfsim_global_.D1") == "farad"

        finally:
            pm.resolve_parameter = original_resolve # Restore

    # --- Test Memoization (evaluation_context) ---
    def test_resolve_memoization(self, comprehensive_built_pm):
        pm = comprehensive_built_pm
        freq = np.array([1e9])
        context = {} # Fresh context

        # To check memoization, we can see if a computationally intensive part (like compiled_func)
        # is called multiple times.
        # Let's spy on a compiled function.
        original_compiled_func = pm._compiled_functions.get("_rfsim_global_.Calc_C")
        assert original_compiled_func is not None
        call_count = 0

        def spy_compiled_func(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_compiled_func(*args, **kwargs)

        pm._compiled_functions["_rfsim_global_.Calc_C"] = spy_compiled_func

        try:
            _ = pm.resolve_parameter("_rfsim_global_.Calc_C", freq, "farad", context)
            assert call_count == 1 # Called once

            _ = pm.resolve_parameter("_rfsim_global_.Calc_C", freq, "farad", context) # Same context
            assert call_count == 1 # Should be memoized, not called again

            context2 = {} # Different context
            _ = pm.resolve_parameter("_rfsim_global_.Calc_C", freq, "farad", context2)
            assert call_count == 2 # Called again with new context

        finally:
            pm._compiled_functions["_rfsim_global_.Calc_C"] = original_compiled_func # Restore


    # --- Tests for Error Handling in resolve_parameter ---
    def test_resolve_error_non_existent_param(self, comprehensive_built_pm):
        pm = comprehensive_built_pm
        freq = np.array([1e9])
        context = {}
        with pytest.raises(ParameterScopeError, match="Parameter '_rfsim_global_.NonExistent' not found in context map"):
            pm.resolve_parameter("_rfsim_global_.NonExistent", freq, "dimensionless", context)

    def test_resolve_error_numerical_in_expression(self, comprehensive_built_pm):
        pm = comprehensive_built_pm
        freq = np.array([1e9]) 
        context = {}
        
        # Test for '_rfsim_global_.Div_by_zero_expr' which should fail during ParameterManager.build()
        # So, we can't resolve it here. This part of the test needs to reflect that.
        # This part of the original test is now invalid because '1/0' should be caught at build time.
        # We'll test it by trying to build a PM with it separately.
        temp_pm_div_zero = ParameterManager()
        div_zero_defs = create_defs([
            {'name': 'Div_by_zero_expr', 'scope': 'global', 'expression': '1/0', 'dimension': 'dimensionless'},
        ])
        temp_pm_div_zero.add_definitions(div_zero_defs)
        with pytest.raises(ParameterError, match="Parameter build failed due to expression validation/compilation errors"):
            temp_pm_div_zero.build()


        # Test for '_rfsim_global_.Div_by_freq_expr' = '1/freq'
        freq_zero = np.array([0.0])
        context_f0 = {}
        with pytest.raises(ParameterError) as excinfo:
            pm.resolve_parameter("_rfsim_global_.Div_by_freq_expr", freq_zero, "siemens", context_f0)
        
        assert "Numerical floating point error during evaluation" in str(excinfo.value) or \
               "division by zero" in str(excinfo.value).lower()
        assert "_rfsim_global_.Div_by_freq_expr" in str(excinfo.value)
        assert "1/freq" in str(excinfo.value) # Check expression is in error

        # Test that it works for non-zero frequency
        freq_ok = np.array([1e9])
        context_ok = {}
        try:
            q_ok = pm.resolve_parameter("_rfsim_global_.Div_by_freq_expr", freq_ok, "siemens", context_ok)
            assert isinstance(q_ok, Quantity)
            assert np.isclose(q_ok.magnitude, 1e-9)
        except ParameterError:
            pytest.fail("resolve_parameter failed for Div_by_freq_expr with non-zero frequency.")

    def test_resolve_error_bad_target_dimension_str_for_quantity(self, comprehensive_built_pm):
        pm = comprehensive_built_pm
        freq = np.array([1e9])
        context = {}
        # _rfsim_global_.Freq_dep_X will evaluate to a number.
        # Try to create Quantity with an invalid target_dimension_str string
        with pytest.raises(ParameterError, match=re.escape(r"Error processing evaluated result for '_rfsim_global_.Freq_dep_X'. Numerical result (shape (1,), dtype float64) with its declared dimension 'dimensionless' could not be converted to target dimension 'invalid_unit_str': 'invalid_unit_str' is not defined in the unit registry")):
            pm.resolve_parameter("_rfsim_global_.Freq_dep_X", freq, "invalid_unit_str", context)

    def test_resolve_freq_hz_not_numpy_array(self, comprehensive_built_pm):
        pm = comprehensive_built_pm
        context = {}
        # freq_hz as list
        q_list = pm.resolve_parameter("_rfsim_global_.R_const", [1e9, 2e9], "ohm", context)
        assert isinstance(q_list.magnitude, np.ndarray)
        assert q_list.magnitude.shape == (2,)

        # freq_hz as float scalar
        context = {} # new context
        q_float = pm.resolve_parameter("_rfsim_global_.R_const", 1e9, "ohm", context)
        assert isinstance(q_float.magnitude, np.ndarray)
        assert q_float.magnitude.shape == (1,) # Will be converted to [1e9] internally

        # freq_hz as 0D numpy array
        context = {}
        q_0d = pm.resolve_parameter("_rfsim_global_.R_const", np.array(1e9), "ohm", context)
        assert isinstance(q_0d.magnitude, np.ndarray)
        assert q_0d.magnitude.shape == (1,)


    def test_resolve_param_definition_error_during_constant_resolve(self, empty_pm):
        # Test case where get_constant_value might fail due to a bad underlying definition that
        # somehow passed initial build (e.g. a reference to a param that is an expression but
        # wasn't caught as such, or a non-Quantity parseable string that isn't a reference)
        # This is more of an internal consistency check for ParameterManager error propagation.
        pm = empty_pm
        defs = [
            ParameterDefinition(name="P1", scope="global", constant_value_str="P2", declared_dimension_str="ohm"),
            # P2 is an expression, so P1 is not truly a constant that can be resolved by get_constant_value directly
            ParameterDefinition(name="P2", scope="global", expression_str="freq", declared_dimension_str="ohm")
        ]
        pm.add_definitions(defs)
        pm.build()

        with pytest.raises(ParameterError, match=r"Failed to resolve parameter '_rfsim_global_\.P1' to a constant value: Parameter '_rfsim_global_\.P2' .* is an expression"):
            pm.resolve_parameter("_rfsim_global_.P1", np.array([1e9]), "ohm", {})

    def test_resolve_dimensionally_inconsistent_expression(self, circuit_builder_instance):
        """
        Tests that an expression which is numerically evaluable but dimensionally
        inconsistent (e.g., "10 meter + 5") results in a Quantity with the
        parameter's declared dimension, using the numerical result.
        This behavior is by design: user is responsible for dimensional correctness
        within expressions if physical meaning is desired beyond numerical evaluation.
        """
        # Define a circuit where R1.resistance (ohms) is set by an expression "length_param + some_number"
        # length_param is "10 meter", some_number is 5.
        # Expected numerical evaluation: 10 + 5 = 15.
        # Expected final Quantity for R1.resistance: Quantity(15, "ohm").
        components_def = [
            ("R1", "Resistor", 
             # MODIFIED: Use the new prefix for global parameter in expression
             {"resistance": {"expression": "_rfsim_global_.length_param + 5"}}, 
             {"0": "N1", "1": "gnd"})
        ]
        global_params_def = {
            "length_param": "10 meter" # This will be inferred as 'meter'
        }
        
        # 1. Test ParameterManager resolution directly
        pm = ParameterManager()
        # Manually create ParameterDefinitions like CircuitBuilder would
        # Note: CircuitBuilder infers 'ohm' for R1.resistance from Resistor.declare_parameters()
        # and 'meter' for _rfsim_global_.length_param from its value.
        defs = [
            ParameterDefinition(name="length_param", scope="global", constant_value_str="10 meter", declared_dimension_str="meter"),
            ParameterDefinition(name="resistance", scope="instance", owner_id="R1", 
                                # Expression "length_param + 5" refers to an unqualified 'length_param'.
                                # ParameterManager will try to resolve 'length_param' to 'R1.length_param' (not found),
                                # then to '_rfsim_global_.length_param' (found).
                                expression_str="length_param + 5", declared_dimension_str="ohm")
        ]
        pm.add_definitions(defs)
        pm.build()

        freq_array = np.array([1e9])
        eval_context = {}
        
        # ParameterManager should resolve _rfsim_global_.length_param as 10 meter
        # MODIFIED: Use the new internal name for the global parameter
        len_q = pm.resolve_parameter("_rfsim_global_.length_param", freq_array, "meter", eval_context)
        assert len_q.magnitude == pytest.approx(np.array([10.0]))
        assert len_q.units == ureg.meter

        # R1.resistance resolves:
        # Expression "length_param + 5" -> SymPy uses magnitudes -> 10 (from _rfsim_global_.length_param) + 5 = 15
        # Result 15 is then given the declared dimension "ohm" for R1.resistance
        res_q = pm.resolve_parameter("R1.resistance", freq_array, "ohm", eval_context)
        
        assert isinstance(res_q, Quantity)
        assert res_q.magnitude == pytest.approx(np.array([15.0]))
        assert res_q.units == ureg.ohm
        assert res_q.check("[resistance]")

        # 2. Test integration through MNA assembly
        # Create the circuit using the helper which calls CircuitBuilder
        sim_circuit_inconsistent_expr = create_and_build_circuit(
            circuit_builder_instance,
            components_def, # components_def already updated with _rfsim_global_
            global_params_def=global_params_def,
            external_ports_def={"N1": "50 ohm"},
            circuit_name="TestInconsistentExpr"
        )

        # Basic check: Ensure circuit built and parameters are resolvable by MNAAssembler
        # (MNAAssembler will internally call ParameterManager.resolve_parameter)
        try:
            assembler = MnaAssembler(sim_circuit_inconsistent_expr)
            # If assembly of the R1 component doesn't crash on parameter resolution, it's a good sign.
            # We expect R1's resistance to be resolved to 15 ohm.
            # We can even check the stamp if we want to be very thorough here,
            # but for this test, not crashing is the main goal for the integration part.
            _ = assembler.assemble(freq_hz=1e9) # Try to assemble
            # Further check: retrieve the actual resolved resistance if possible
            # (This requires more complex access to internal MNA data or component state,
            #  which might not be exposed. For now, not crashing is sufficient for this integration test part).
            r1_sim_comp = sim_circuit_inconsistent_expr.sim_components.get("R1")
            assert r1_sim_comp is not None

            # Check the actual resolved parameter by the component (if a method exists to do so,
            # or by checking its stamp value). For now, let's assume if assemble works, it's fine.
            # This is more of a "does it integrate without syntax error" check.

        except (ParameterError, SemanticValidationError, MnaInputError) as e:
            pytest.fail(f"Circuit building or MNA assembly failed for dimensionally inconsistent expression test: {e}")