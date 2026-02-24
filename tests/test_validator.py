"""Tests for ValidationAgent - syntax and import validation."""
import pytest
from ai_intern.validation.validator import ValidationAgent


class TestValidateSyntax:
    def test_valid_code(self):
        valid, error = ValidationAgent.validate_syntax("x = 1 + 2")
        assert valid is True
        assert error == ""

    def test_valid_function(self):
        code = "def foo(x):\n    return x * 2"
        valid, _ = ValidationAgent.validate_syntax(code)
        assert valid is True

    def test_syntax_error(self):
        valid, error = ValidationAgent.validate_syntax("def foo(:")
        assert valid is False
        assert "SyntaxError" in error

    def test_incomplete_string(self):
        valid, error = ValidationAgent.validate_syntax("x = 'hello")
        assert valid is False

    def test_empty_code(self):
        valid, _ = ValidationAgent.validate_syntax("")
        assert valid is True


class TestValidateImports:
    def test_valid_stdlib_import(self):
        valid, error = ValidationAgent.validate_imports("import os\nimport json")
        assert valid is True

    def test_invalid_import(self):
        valid, error = ValidationAgent.validate_imports("import nonexistent_module_xyz")
        assert valid is False
        assert "ImportError" in error

    def test_from_import(self):
        valid, _ = ValidationAgent.validate_imports("from pathlib import Path")
        assert valid is True

    def test_no_imports(self):
        valid, _ = ValidationAgent.validate_imports("x = 1 + 2")
        assert valid is True

    def test_dangerous_code_not_executed(self):
        """Code with os.system should NOT be executed during import validation."""
        code = 'import os\nos.system("echo SHOULD_NOT_RUN")\nx = 1'
        valid, _ = ValidationAgent.validate_imports(code)
        # Should pass import check (os is valid) without executing os.system
        assert valid is True

    def test_syntax_error_passes(self):
        """Syntax errors should be caught by validate_syntax, not imports."""
        valid, _ = ValidationAgent.validate_imports("def foo(:")
        assert valid is True  # AST parse fails silently, defers to syntax check


class TestValidateTask:
    def test_non_code_task_short_result(self):
        from ai_intern.schemas import TaskSchema
        task = TaskSchema(plan_id="p", task_order=0, goal="research test", result="short")
        validated, _ = ValidationAgent.validate_task(task)
        assert validated.status == "failed"

    def test_non_code_task_valid_result(self):
        from ai_intern.schemas import TaskSchema
        task = TaskSchema(
            plan_id="p", task_order=0,
            goal="research python web frameworks",
            result="Python has many web frameworks including Flask, Django, and FastAPI. " * 5,
        )
        validated, _ = ValidationAgent.validate_task(task)
        assert validated.status == "validated"

    def test_no_result(self):
        from ai_intern.schemas import TaskSchema
        task = TaskSchema(plan_id="p", task_order=0, goal="write code")
        validated, _ = ValidationAgent.validate_task(task)
        assert validated.status == "failed"
