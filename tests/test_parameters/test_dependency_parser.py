# tests/test_parameters/test_dependency_parser.py
import pytest
from rfsim_core.parameters import ASTDependencyExtractor

@pytest.fixture
def extractor():
    return ASTDependencyExtractor()

# === Group 1: Core Functionality Tests ===

def test_simple_name(extractor):
    """Verifies a standalone identifier is found."""
    assert extractor.get_dependencies("gain * 2") == {"gain"}

def test_hierarchical_identifier_is_single_dependency(extractor):
    """Verifies a.b.c is treated as one dependency."""
    deps = extractor.get_dependencies("sub1.R1.resistance * 2")
    assert deps == {"sub1.R1.resistance"}

def test_multiple_hierarchical_dependencies(extractor):
    """Verifies multiple dependencies are found."""
    deps = extractor.get_dependencies("amp1.gain / lpf.C1.capacitance")
    assert deps == {"amp1.gain", "lpf.C1.capacitance"}

def test_function_call_with_dependencies(extractor):
    """
    CORRECTED TEST: Verifies dependencies inside a function call are found,
    and the function itself is correctly identified as a hierarchical dependency.
    """
    deps = extractor.get_dependencies("np.log(my_param) + pi")
    # The extractor correctly identifies `np.log` as the accessed attribute chain.
    # It also finds `my_param` and `pi` as standalone names.
    assert deps == {"np.log", "my_param", "pi"}

# === Group 2: Hardening and Edge Case Tests (These previously failed) ===

def test_hardened_visitor_finds_deps_in_complex_attributes(extractor):
    """
    VERIFIES THE FIX: Checks that the visitor descends into complex
    expressions that are part of an attribute access.
    """
    deps = extractor.get_dependencies("(a+b).c * valid.dep")
    # The visitor correctly ignores `(a+b).c` as a single unit but finds
    # the children `a` and `b`, plus the valid dependency.
    assert deps == {"a", "b", "valid.dep"}

def test_hardened_visitor_finds_deps_in_function_call_attributes(extractor):
    """
    VERIFIES THE FIX: Checks that the visitor descends into a function
    call that is part of an attribute access.
    """
    deps = extractor.get_dependencies("my_func().output * valid.dep")
    # The visitor correctly ignores `my_func().output` as a single unit
    # but finds the child `my_func`, plus the valid dependency.
    assert deps == {"my_func", "valid.dep"}

# === Group 3: Negative and Syntax Tests ===

def test_invalid_syntax_raises_syntax_error(extractor):
    """Verifies that unparsable syntax correctly raises SyntaxError."""
    with pytest.raises(SyntaxError):
        extractor.get_dependencies("5 * * 2")

def test_empty_string_is_valid(extractor):
    """Ensures an empty expression string parses to zero dependencies."""
    assert extractor.get_dependencies("") == set()

def test_literal_string_has_no_dependencies(extractor):
    """Ensures a simple literal value has no dependencies."""
    assert extractor.get_dependencies("'hello world'") == set() 