"""
Test sanitizer for LLM-generated test code.

Uses AST analysis (no LLM) to catch common bugs in generated tests
before execution: extra imports, hardcoded assertions, undefined names.
"""
import ast
import sys

from ..logging_config import get_logger

logger = get_logger("test_sanitizer")

# Stdlib modules for filtering
_STDLIB = frozenset(
    m.lower() for m in getattr(sys, "stdlib_module_names", set())
) or frozenset([
    "os", "sys", "json", "csv", "re", "math", "pathlib", "datetime",
    "collections", "itertools", "functools", "io", "tempfile",
    "subprocess", "threading", "logging", "unittest", "ast", "typing",
    "hashlib", "uuid", "socket", "http", "urllib", "sqlite3",
    "argparse", "string", "copy", "enum", "dataclasses", "abc",
    "contextlib", "shutil", "glob", "time", "random", "statistics",
    "decimal", "fractions", "textwrap", "pprint", "struct", "array",
    "bisect", "heapq", "operator", "numbers",
])

# Small constants that are legitimate in assertions
_SAFE_CONSTANTS = {0, 1, -1, True, False, None, "", 0.0, 1.0, -1.0}


class TestSanitizer:
    """Static analysis pass on LLM-generated test code."""

    @classmethod
    def sanitize(cls, original_code: str, test_code: str) -> tuple:
        """
        Clean LLM-generated test code.

        Returns (cleaned_test_code, list_of_warnings).
        """
        warnings = []

        # Parse both — if either fails, return test_code as-is
        try:
            orig_tree = ast.parse(original_code)
        except SyntaxError:
            return test_code, ["Could not parse original code"]

        try:
            test_tree = ast.parse(test_code)
        except SyntaxError:
            return test_code, ["Could not parse test code"]

        # Get imports from original code
        orig_imports = cls._extract_imports(orig_tree)

        # Check 1: Remove extra non-stdlib imports
        test_code, import_warnings = cls._remove_extra_imports(
            test_code, test_tree, orig_imports
        )
        warnings.extend(import_warnings)

        # Re-parse after potential modifications
        try:
            test_tree = ast.parse(test_code)
        except SyntaxError:
            return test_code, warnings + ["Test code became invalid after import removal"]

        # Check 2: Relax hardcoded numeric assertions
        test_code, assert_warnings = cls._relax_hardcoded_assertions(test_code, test_tree)
        warnings.extend(assert_warnings)

        # Check 3: Warn about undefined name references
        orig_names = cls.extract_public_names(original_code)
        undef_warnings = cls._check_undefined_refs(test_tree, orig_names, orig_imports)
        warnings.extend(undef_warnings)

        return test_code, warnings

    @staticmethod
    def extract_public_names(code: str) -> list:
        """Return function/class names defined at module level."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        names = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                names.append(node.name)
            elif isinstance(node, ast.ClassDef):
                names.append(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.append(target.id)
        return names

    @staticmethod
    def _extract_imports(tree: ast.Module) -> set:
        """Extract top-level module names imported in the code."""
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])
        return imports

    @classmethod
    def _remove_extra_imports(
        cls, test_code: str, test_tree: ast.Module, orig_imports: set
    ) -> tuple:
        """Remove imports in test code that aren't in original code and aren't stdlib."""
        warnings = []
        lines_to_remove = set()

        for node in ast.iter_child_nodes(test_tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name.split(".")[0]
                    if mod.lower() not in _STDLIB and mod not in orig_imports:
                        lines_to_remove.add(node.lineno)
                        warnings.append(
                            f"Removed extra import: {alias.name} (not in original code or stdlib)"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    mod = node.module.split(".")[0]
                    if mod.lower() not in _STDLIB and mod not in orig_imports:
                        lines_to_remove.add(node.lineno)
                        warnings.append(
                            f"Removed extra import: from {node.module} (not in original code or stdlib)"
                        )

        if not lines_to_remove:
            return test_code, warnings

        # Remove the offending lines
        code_lines = test_code.split("\n")
        cleaned = [
            line for i, line in enumerate(code_lines, 1)
            if i not in lines_to_remove
        ]
        return "\n".join(cleaned), warnings

    @classmethod
    def _relax_hardcoded_assertions(cls, test_code: str, test_tree: ast.Module) -> tuple:
        """Replace hardcoded numeric assertions with type checks."""
        warnings = []
        replacements = {}  # lineno -> new_line

        for node in ast.walk(test_tree):
            if not isinstance(node, ast.Assert):
                continue
            test_node = node.test
            # Pattern: assert expr == <literal>
            if not isinstance(test_node, ast.Compare):
                continue
            if len(test_node.ops) != 1 or not isinstance(test_node.ops[0], ast.Eq):
                continue
            if len(test_node.comparators) != 1:
                continue

            comparator = test_node.comparators[0]
            if not isinstance(comparator, ast.Constant):
                continue

            val = comparator.value
            # Skip safe small constants
            if val in _SAFE_CONSTANTS:
                continue
            # Only relax numeric hardcoded values
            if not isinstance(val, (int, float)):
                continue

            # This is a hardcoded numeric assertion — relax it
            lineno = node.lineno
            warnings.append(
                f"Relaxed hardcoded assertion at line {lineno}: == {val} -> isinstance check"
            )

            # Build replacement: get the left-hand side as source text
            # We need to reconstruct from AST
            try:
                lhs_source = ast.get_source_segment(test_code, test_node.left)
            except Exception:
                lhs_source = None

            if lhs_source:
                # Determine the expected types
                if isinstance(val, float):
                    type_check = "(int, float)"
                else:
                    type_check = "(int, float)"
                replacements[lineno] = (
                    f"assert isinstance({lhs_source}, {type_check}), "
                    f"f\"Expected numeric type, got {{type({lhs_source}).__name__}}\""
                )

        if not replacements:
            return test_code, warnings

        code_lines = test_code.split("\n")
        for lineno, new_line in replacements.items():
            if lineno <= len(code_lines):
                # Preserve indentation
                old_line = code_lines[lineno - 1]
                indent = len(old_line) - len(old_line.lstrip())
                code_lines[lineno - 1] = " " * indent + new_line

        return "\n".join(code_lines), warnings

    @classmethod
    def _check_undefined_refs(
        cls, test_tree: ast.Module, orig_names: list, orig_imports: set
    ) -> list:
        """Warn if test code calls names not defined in original code."""
        warnings = []
        # Collect all Name nodes used as function calls in tests
        for node in ast.walk(test_tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    name = func.id
                    # Skip builtins and common test utilities
                    builtins = {"print", "len", "str", "int", "float", "list", "dict",
                                "set", "tuple", "type", "isinstance", "hasattr", "range",
                                "enumerate", "zip", "map", "filter", "sorted", "reversed",
                                "min", "max", "sum", "abs", "round", "any", "all", "open",
                                "getattr", "setattr", "callable", "super", "repr", "bool",
                                "bytes", "bytearray", "memoryview", "complex", "frozenset",
                                "property", "staticmethod", "classmethod", "id", "hash",
                                "format", "input", "vars", "dir", "hex", "oct", "bin",
                                "ord", "chr", "ascii", "iter", "next", "slice", "object"}
                    if name in builtins:
                        continue
                    if name in orig_names:
                        continue
                    if name in orig_imports:
                        continue
                    warnings.append(
                        f"Test calls '{name}' which is not defined in original code"
                    )
        return warnings
