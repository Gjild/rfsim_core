# --- tests/test_parameters.py ---

import pytest
import numpy as np
import pint
import sympy
import networkx as nx
from typing import List, Dict, Any
import re

# Imports from the source code
from rfsim_core.units import ureg, Quantity
from rfsim_core.parameters import (
    ParameterManager,
    ParameterDefinition,
    ParameterError,
    ParameterSyntaxError,
    ParameterDefinitionError,
    ParameterScopeError,
    CircularParameterDependencyError,
)
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
        assert 'global.R0' in pm._parameter_context_map
        assert pm._parameter_context_map['global.R0']['declared_dimension'] == 'ohm'
        assert pm._parameter_context_map['global.R0']['dependencies'] == set()
        assert pm._parameter_context_map['global.R0']['sympy_expr'] is None

        assert 'global.L_val' in pm._parameter_context_map
        assert pm._parameter_context_map['global.L_val']['declared_dimension'] == 'henry' # Dimensionality of henry
        assert 'R1.resistance' in pm._parameter_context_map

        # Check constant values
        assert pm.get_constant_value('global.R0') == Quantity(50.0, 'ohm')
        assert pm.get_constant_value('global.L_val') == Quantity(10.0, 'nH')
        assert pm.get_constant_value('R1.resistance') == Quantity(1.0, 'kohm')

        # Check dependencies via accessor
        assert pm.get_dependencies('global.R0') == set()
        assert pm.get_dependencies('R1.resistance') == set()

        # Check graph directly (optional)
        assert set(pm._dependency_graph.nodes) == {'global.R0', 'global.L_val', 'R1.resistance'}
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
        assert 'global.cap_val' in pm._compiled_functions
        assert 'global.res_val' in pm._compiled_functions
        assert callable(pm._compiled_functions['global.cap_val'])
        assert callable(pm._compiled_functions['global.res_val'])

        assert 'global.cap_val' in pm._parameter_context_map
        assert pm._parameter_context_map['global.cap_val']['dependencies'] == {'freq'} # Only depends on reserved 'freq'
        assert isinstance(pm._parameter_context_map['global.cap_val']['sympy_expr'], sympy.Expr)

        assert 'global.res_val' in pm._parameter_context_map
        assert pm._parameter_context_map['global.res_val']['dependencies'] == {'global.R0'} # Resolved correctly
        assert isinstance(pm._parameter_context_map['global.res_val']['sympy_expr'], sympy.Expr)

        assert pm.get_dependencies('global.freq_ghz') == set()
        assert pm.get_dependencies('global.R0') == set()
        assert pm.get_dependencies('global.cap_val') == {'freq'}
        assert pm.get_dependencies('global.res_val') == {'global.R0'}

        # Check graph
        assert pm._dependency_graph.has_node('freq')
        assert pm._dependency_graph.has_edge('global.cap_val', 'freq')
        assert pm._dependency_graph.has_edge('global.res_val', 'global.R0')
        assert pm._dependency_graph.number_of_edges() == 2

    def test_build_expression_instance_and_global_dep(self, empty_pm):
        """Test expression in instance param depending on another instance param and a global."""
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
        assert pm.get_dependencies('R1.scaled_res') == {'R1.base_res', 'global.global_gain'}

        # Check dependencies for R2.dep_res (now handles explicit R1.scaled_res)
        assert pm.get_dependencies('R2.dep_res') == {'R1.scaled_res'}

        # Check graph
        assert pm._dependency_graph.has_edge('R1.scaled_res', 'R1.base_res')
        assert pm._dependency_graph.has_edge('R1.scaled_res', 'global.global_gain')
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
        sympy_expr = pm._parameter_context_map['global.test']['sympy_expr']
        assert isinstance(sympy_expr, sympy.Expr)
        assert sympy.Symbol('r_one') in sympy_expr.free_symbols


    def test_scope_resolution_instance_over_global(self, empty_pm):
        """Test that instance parameters shadow global ones during resolution."""
        pm = empty_pm
        defs = create_defs([
            {'name': 'X', 'scope': 'global', 'constant': '10 V', 'dimension': 'volt'},
            {'name': 'X', 'scope': 'instance', 'owner_id': 'Comp1', 'constant': '5 V', 'dimension': 'volt'},
            {'name': 'Y', 'scope': 'instance', 'owner_id': 'Comp1', 'expression': 'X * 2', 'dimension': 'volt'}, # Should use Comp1.X
            {'name': 'Z', 'scope': 'global', 'expression': 'X * 3', 'dimension': 'volt'}, # Should use global.X
        ])
        pm.add_definitions(defs)
        pm.build()

        assert pm.get_dependencies('Comp1.Y') == {'Comp1.X'}
        assert pm.get_dependencies('global.Z') == {'global.X'}

        assert pm._dependency_graph.has_edge('Comp1.Y', 'Comp1.X')
        assert pm._dependency_graph.has_edge('global.Z', 'global.X')
        assert not pm._dependency_graph.has_edge('Comp1.Y', 'global.X')
        assert not pm._dependency_graph.has_edge('global.Z', 'Comp1.X')

# --- Test ParameterManager Error Handling During Build ---

class TestParameterManagerBuildErrors:

    def test_error_duplicate_internal_name_global(self, empty_pm):
        pm = empty_pm
        defs = create_defs([
            {'name': 'R0', 'scope': 'global', 'constant': '50 ohm', 'dimension': 'ohm'},
            {'name': 'R0', 'scope': 'global', 'constant': '100 ohm', 'dimension': 'ohm'},
        ])
        pm.add_definitions(defs)
        with pytest.raises(ParameterDefinitionError, match="Duplicate internal name 'global.R0'"):
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
        expected_error_msg_regex = r"Invalid declared_dimension_str 'not_a_dimension' for parameter 'global.R0':"
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
        with pytest.raises(ParameterDefinitionError, match="Constant value string '50 qqq' for parameter 'global.R0'"):
            pm.build()

    def test_error_expression_syntax_error(self, empty_pm):
        pm = empty_pm
        defs = create_defs([
            {'name': 'R0', 'scope': 'global', 'constant': '50 ohm', 'dimension': 'ohm'},
            {'name': 'X', 'scope': 'global', 'expression': 'R0 * (2 + )', 'dimension': 'ohm'}, # Syntax error
        ])
        pm.add_definitions(defs)
        with pytest.raises(ParameterSyntaxError, match="Syntax error parsing expression for 'global.X'.*'R0 \* \(2 \+ \)'"):
            pm.build()

    def test_error_expression_dependency_not_found(self, empty_pm):
        pm = empty_pm
        defs = create_defs([
            {'name': 'R0', 'scope': 'global', 'constant': '50 ohm', 'dimension': 'ohm'},
            {'name': 'X', 'scope': 'global', 'expression': 'R0 * R1', 'dimension': 'ohm'}, # R1 not defined
        ])
        pm.add_definitions(defs)
        with pytest.raises(ParameterScopeError, match=re.escape("Dependency resolution failed for expression parameter 'global.X' ('R0 * R1'): Parameter symbol 'R1' referenced in an expression could not be resolved. Context: scope='global'. Searched for <N/A for instance> and 'global.R1'.")):
            pm.build()

    def test_error_instance_dependency_not_found(self, empty_pm):
        pm = empty_pm
        defs = create_defs([
            {'name': 'resistance', 'scope': 'instance', 'owner_id':'R1', 'constant': '50 ohm', 'dimension': 'ohm'},
            {'name': 'scaled', 'scope': 'instance', 'owner_id':'R1', 'expression': 'resistance * gain', 'dimension': 'ohm'}, # gain not defined
        ])
        pm.add_definitions(defs)
        with pytest.raises(ParameterScopeError, match=re.escape("Dependency resolution failed for expression parameter 'R1.scaled' ('resistance * gain'): Parameter symbol 'gain' referenced in an expression could not be resolved. Context: scope='instance', owner='R1'. Searched for 'R1.gain' and 'global.gain'.")):
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
        with pytest.raises(CircularParameterDependencyError, match="Circular dependency detected: global.[ABC] -> global.[ABC] -> global.[ABC] -> global.[ABC]"):
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
            pm.get_parameter_definition("global.R0")
        with pytest.raises(ParameterError, match="ParameterManager has not been built."):
            pm.get_dependencies("global.R0")
        with pytest.raises(ParameterError, match="ParameterManager has not been built."):
            pm.is_constant("global.R0")
        with pytest.raises(ParameterError, match="ParameterManager has not been built."):
            pm.get_constant_value("global.R0")
        with pytest.raises(ParameterError, match="ParameterManager has not been built."):
            # Placeholder methods also need the check
            pm.get_compiled_function("global.R0")
        with pytest.raises(ParameterError, match="ParameterManager has not been built."):
           pm.resolve_parameter("global.R0", np.array([1e9]), "ohm", {})


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
        assert set(names) == {'global.R0', 'global.L_val', 'global.Scale', 'R1.resistance', 'R1.current'}

    def test_get_parameter_definition(self, built_pm):
        defn = built_pm.get_parameter_definition('global.R0')
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
        assert built_pm.get_declared_dimension('global.R0') == 'ohm' # ohm
        assert built_pm.get_declared_dimension('global.L_val') == 'henry' # henry
        assert built_pm.get_declared_dimension('global.Scale') == 'dimensionless' # Dimensionless
        assert built_pm.get_declared_dimension('R1.resistance') == 'ohm' # ohm
        # Check admittance dimension string formatting by Pint

        assert built_pm.get_declared_dimension('R1.current') == 'siemens' # siemens

        with pytest.raises(ParameterScopeError, match="not found"):
            built_pm.get_declared_dimension('nonexistent.param')

    def test_get_dependencies(self, built_pm):
        assert built_pm.get_dependencies('global.R0') == set()
        assert built_pm.get_dependencies('global.L_val') == set()
        assert built_pm.get_dependencies('global.Scale') == {'global.L_val'}
        assert built_pm.get_dependencies('R1.resistance') == set()
        assert built_pm.get_dependencies('R1.current') == {'R1.resistance'}

        with pytest.raises(ParameterScopeError, match="not found"):
            built_pm.get_dependencies('nonexistent.param')

    def test_is_constant(self, built_pm):
        assert built_pm.is_constant('global.R0') is True
        assert built_pm.is_constant('global.L_val') is True
        assert built_pm.is_constant('global.Scale') is False # Expression
        assert built_pm.is_constant('R1.resistance') is True
        assert built_pm.is_constant('R1.current') is False # Expression

        with pytest.raises(ParameterScopeError, match="not found"):
             built_pm.is_constant('nonexistent.param')

    def test_get_constant_value(self, built_pm):
        assert built_pm.get_constant_value('global.R0') == Quantity(50, 'ohm')
        assert built_pm.get_constant_value('global.L_val') == Quantity(10, 'nH')
        assert built_pm.get_constant_value('R1.resistance') == Quantity(1, 'kohm')

        # Test error for expression param
        with pytest.raises(ParameterError, match=re.escape("Parameter 'global.Scale' is an expression ('L_val * 1e9') and cannot be retrieved as a simple constant value. Use resolve_parameter().")): # More specific match
            built_pm.get_constant_value('global.Scale')
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
        q = pm.resolve_parameter("global.R_const", freq, "ohm", context)
        assert isinstance(q, Quantity)
        assert q.magnitude == pytest.approx(np.array([50.0]))
        assert q.check("[resistance]") # Check base dimension

        q_kohm = pm.resolve_parameter("global.R_const", freq, "kiloohm", context)
        assert q_kohm.magnitude == pytest.approx(np.array([0.05]))
        assert q_kohm.check("[resistance]")

    def test_resolve_constant_broadcast(self, comprehensive_built_pm):
        pm = comprehensive_built_pm
        freq = np.array([1e9, 2e9, 3e9])
        context = {}
        q = pm.resolve_parameter("global.Val_dimless", freq, "dimensionless", context)
        assert q.magnitude.shape == freq.shape
        assert np.all(q.magnitude == pytest.approx(2.5))

    def test_resolve_constant_ref(self, comprehensive_built_pm):
        pm = comprehensive_built_pm
        freq = np.array([1e9])
        context = {}
        q = pm.resolve_parameter("global.C_ref_R", freq, "ohm", context)
        assert q.magnitude == pytest.approx(np.array([50.0]))

    def test_resolve_constant_incompatible_target_dim(self, comprehensive_built_pm):
        pm = comprehensive_built_pm
        freq = np.array([1e9])
        context = {}
        # Adjusted regex to be more robust to pint's error message formatting
        with pytest.raises(pint.DimensionalityError, match=r"Cannot convert from 'ohm' .* to 'farad'"):
            pm.resolve_parameter("global.R_const", freq, "farad", context)

    def test_resolve_constant_empty_freq(self, comprehensive_built_pm):
        pm = comprehensive_built_pm
        freq = np.array([], dtype=float)
        context = {}
        q = pm.resolve_parameter("global.R_const", freq, "ohm", context)
        assert isinstance(q, Quantity)
        assert q.magnitude.shape == (0,)
        assert q.check("[resistance]")

    # --- Tests for Expressions ---
    def test_resolve_simple_expression(self, comprehensive_built_pm):
        pm = comprehensive_built_pm
        # Example: 'Val_dimless * 2' - needs new param
        # For now, let's use an existing expression: global.L_scaled = L_const_nH * Val_dimless
        freq = np.array([1e9])
        context = {}
        # L_const_nH = 10 nH, Val_dimless = 2.5
        # L_scaled = 10 nH * 2.5 = 25 nH
        q = pm.resolve_parameter("global.L_scaled", freq, "nanohenry", context)
        assert q.magnitude == pytest.approx(np.array([25.0]))
        assert q.check("[inductance]")

        q_H = pm.resolve_parameter("global.L_scaled", freq, "henry", context)
        assert q_H.magnitude == pytest.approx(np.array([25e-9]))

    def test_resolve_freq_dependent_expression(self, comprehensive_built_pm):
        pm = comprehensive_built_pm
        freq = np.array([1e9]) # 1 GHz
        context = {}
        # global.Freq_dep_X = freq * 1e-12
        q = pm.resolve_parameter("global.Freq_dep_X", freq, "dimensionless", context)
        assert q.magnitude == pytest.approx(np.array([1e9 * 1e-12])) # 1e-3

        freq_multi = np.array([1e9, 2e9])
        context = {}
        q_multi = pm.resolve_parameter("global.Freq_dep_X", freq_multi, "dimensionless", context)
        assert q_multi.magnitude == pytest.approx(np.array([1e-3, 2e-3]))

    def test_resolve_expression_with_deps(self, comprehensive_built_pm):
        pm = comprehensive_built_pm
        freq = np.array([1e9])
        context = {}
        # global.Calc_C = 1 / (2 * pi * freq * R_const)
        # R_const = 50 ohm
        expected_c = 1 / (2 * np.pi * 1e9 * 50)
        q = pm.resolve_parameter("global.Calc_C", freq, "farad", context)
        assert q.magnitude == pytest.approx(np.array([expected_c]))

    def test_resolve_expression_numpy_funcs(self, comprehensive_built_pm):
        pm = comprehensive_built_pm
        freq = np.array([1e8]) # 100 MHz for log10 to be 8
        context = {}
        q = pm.resolve_parameter("global.Log_val", freq, "dimensionless", context)
        assert q.magnitude == pytest.approx(np.array([8.0]))

    def test_resolve_expression_empty_freq(self, comprehensive_built_pm):
        pm = comprehensive_built_pm
        freq = np.array([], dtype=float)
        context = {}
        q = pm.resolve_parameter("global.Freq_dep_X", freq, "dimensionless", context)
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
        # C1.c_inst_expr = global.Calc_C / 2
        # global.Calc_C = 1 / (2 * pi * freq * R_const)
        # R_const = 50 ohm
        calc_c_val = 1 / (2 * np.pi * 1e9 * 50)
        expected_c_inst = calc_c_val / 2
        q = pm.resolve_parameter("C1.c_inst_expr", freq, "farad", context)
        assert q.magnitude == pytest.approx(np.array([expected_c_inst]))

    def test_resolve_instance_expr_multi_scope_deps(self, comprehensive_built_pm):
        pm = comprehensive_built_pm
        freq = np.array([1e9]) # freq doesn't matter for this expression
        context = {}
        # AMP1.gain = sqrt(R1.r_inst_val / global.R_const)
        # R1.r_inst_val = 1 kohm = 1000 ohm
        # global.R_const = 50 ohm
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
            _ = pm.resolve_parameter("global.D3_expr", freq, "farad", context)

            # Check target dimensions used for dependencies:
            # When resolving D3_expr, it needs D2_expr. D2_expr's declared_dimension is 'farad'.
            assert resolved_interమediates.get("global.D2_expr") == "farad"
            # When D2_expr was resolved (possibly initiated by D3_expr's call), it needed D1.
            # D1's declared_dimension is 'farad'.
            # This might be called multiple times if context is not passed correctly or mock is simple.
            # The key is that *a* call to resolve D1 eventually happened with 'farad'.
            assert resolved_interమediates.get("global.D1") == "farad"

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
        original_compiled_func = pm._compiled_functions.get("global.Calc_C")
        assert original_compiled_func is not None
        call_count = 0

        def spy_compiled_func(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_compiled_func(*args, **kwargs)

        pm._compiled_functions["global.Calc_C"] = spy_compiled_func

        try:
            _ = pm.resolve_parameter("global.Calc_C", freq, "farad", context)
            assert call_count == 1 # Called once

            _ = pm.resolve_parameter("global.Calc_C", freq, "farad", context) # Same context
            assert call_count == 1 # Should be memoized, not called again

            context2 = {} # Different context
            _ = pm.resolve_parameter("global.Calc_C", freq, "farad", context2)
            assert call_count == 2 # Called again with new context

        finally:
            pm._compiled_functions["global.Calc_C"] = original_compiled_func # Restore


    # --- Tests for Error Handling in resolve_parameter ---
    def test_resolve_error_non_existent_param(self, comprehensive_built_pm):
        pm = comprehensive_built_pm
        freq = np.array([1e9])
        context = {}
        with pytest.raises(ParameterScopeError, match="Parameter 'global.NonExistent' not found in context map"):
            pm.resolve_parameter("global.NonExistent", freq, "dimensionless", context)

    def test_resolve_error_numerical_in_expression(self, comprehensive_built_pm):
        pm = comprehensive_built_pm
        freq = np.array([1e9]) 
        context = {}
        
        # Test for 'global.Div_by_zero_expr' which should fail during ParameterManager.build()
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


        # Test for 'global.Div_by_freq_expr' = '1/freq'
        freq_zero = np.array([0.0])
        context_f0 = {}
        with pytest.raises(ParameterError) as excinfo:
            pm.resolve_parameter("global.Div_by_freq_expr", freq_zero, "siemens", context_f0)
        
        assert "Numerical floating point error during evaluation" in str(excinfo.value) or \
               "division by zero" in str(excinfo.value).lower()
        assert "global.Div_by_freq_expr" in str(excinfo.value)
        assert "1/freq" in str(excinfo.value) # Check expression is in error

        # Test that it works for non-zero frequency
        freq_ok = np.array([1e9])
        context_ok = {}
        try:
            q_ok = pm.resolve_parameter("global.Div_by_freq_expr", freq_ok, "siemens", context_ok)
            assert isinstance(q_ok, Quantity)
            assert np.isclose(q_ok.magnitude, 1e-9)
        except ParameterError:
            pytest.fail("resolve_parameter failed for Div_by_freq_expr with non-zero frequency.")

    def test_resolve_error_bad_target_dimension_str_for_quantity(self, comprehensive_built_pm):
        pm = comprehensive_built_pm
        freq = np.array([1e9])
        context = {}
        # global.Freq_dep_X will evaluate to a number.
        # Try to create Quantity with an invalid target_dimension_str string
        with pytest.raises(ParameterError, match=re.escape(r"Error processing evaluated result for 'global.Freq_dep_X'. Numerical result (shape (1,), dtype float64) with its declared dimension 'dimensionless' could not be converted to target dimension 'invalid_unit_str': 'invalid_unit_str' is not defined in the unit registry")):
            pm.resolve_parameter("global.Freq_dep_X", freq, "invalid_unit_str", context)

    def test_resolve_freq_hz_not_numpy_array(self, comprehensive_built_pm):
        pm = comprehensive_built_pm
        context = {}
        # freq_hz as list
        q_list = pm.resolve_parameter("global.R_const", [1e9, 2e9], "ohm", context)
        assert isinstance(q_list.magnitude, np.ndarray)
        assert q_list.magnitude.shape == (2,)

        # freq_hz as float scalar
        context = {} # new context
        q_float = pm.resolve_parameter("global.R_const", 1e9, "ohm", context)
        assert isinstance(q_float.magnitude, np.ndarray)
        assert q_float.magnitude.shape == (1,) # Will be converted to [1e9] internally

        # freq_hz as 0D numpy array
        context = {}
        q_0d = pm.resolve_parameter("global.R_const", np.array(1e9), "ohm", context)
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

        with pytest.raises(ParameterError, match=r"Failed to resolve parameter 'global\.P1' to a constant value: Parameter 'global\.P2' .* is an expression"):
            pm.resolve_parameter("global.P1", np.array([1e9]), "ohm", {})