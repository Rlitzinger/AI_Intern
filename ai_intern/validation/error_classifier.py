"""
Error classifier for validation failures.

Classifies errors into retryable vs unrecoverable categories
to avoid wasting tokens on retries that will never succeed.
"""
from enum import Enum
import sys
import functools

from ..logging_config import get_logger

logger = get_logger("error_classifier")


class ErrorCategory(Enum):
    RETRYABLE_SYNTAX = "retryable_syntax"
    RETRYABLE_LOGIC = "retryable_logic"
    RETRYABLE_RUNTIME = "retryable_runtime"
    UNRECOVERABLE_IMPORT = "unrecoverable_import"
    UNRECOVERABLE_TIMEOUT = "unrecoverable_timeout"
    UNRECOVERABLE_SAFETY = "unrecoverable_safety"


class ErrorClassifier:
    """Classifies validation errors into retry categories. Pure string matching, no LLM."""

    # Commonly-requested packages that users might expect but aren't installed
    COMMONLY_REQUESTED = [
        "beautifulsoup4", "bs4", "django", "matplotlib", "numpy", "scipy",
        "selenium", "scrapy", "pillow", "PIL", "opencv-python", "cv2",
        "tensorflow", "torch", "pandas", "flask", "fastapi", "sqlalchemy",
        "celery", "redis", "pymongo", "boto3", "paramiko", "fabric",
        "plotly", "seaborn", "sklearn", "scikit-learn", "nltk", "spacy",
    ]

    # Mapping from import name to pip package name (where they differ)
    IMPORT_TO_PACKAGE = {
        "bs4": "beautifulsoup4",
        "cv2": "opencv-python",
        "PIL": "pillow",
        "sklearn": "scikit-learn",
    }

    @classmethod
    def classify(cls, task) -> ErrorCategory:
        """Classify a task's error_message into an ErrorCategory."""
        error = (task.error_message or "").lower()

        # Safety check first
        if "dangerous code" in error:
            return ErrorCategory.UNRECOVERABLE_SAFETY

        # Timeout check
        if "timed out" in error or "timeout" in error:
            return ErrorCategory.UNRECOVERABLE_TIMEOUT

        # Import errors
        if "importerror" in error or "modulenotfounderror" in error:
            # Extract the module name from the error
            module_name = cls._extract_module_name(task.error_message or "")
            if module_name:
                installed = cls._get_installed_packages()
                stdlib = cls._get_stdlib_modules()
                # Check if it's installed or stdlib but wrong import path
                if module_name in installed or module_name in stdlib:
                    return ErrorCategory.RETRYABLE_LOGIC
                # Not installed at all
                return ErrorCategory.UNRECOVERABLE_IMPORT
            # Can't determine module — assume unrecoverable import
            return ErrorCategory.UNRECOVERABLE_IMPORT

        # Syntax errors
        if "syntaxerror" in error:
            return ErrorCategory.RETRYABLE_SYNTAX

        # Test failures with runtime errors
        if "tests failed" in error or "test" in error:
            if "nameerror" in error:
                return ErrorCategory.RETRYABLE_RUNTIME
            if "attributeerror" in error:
                return ErrorCategory.RETRYABLE_RUNTIME
            if "typeerror" in error:
                return ErrorCategory.RETRYABLE_RUNTIME
            # Generic test failure = logic error
            return ErrorCategory.RETRYABLE_LOGIC

        # Default: retryable logic
        return ErrorCategory.RETRYABLE_LOGIC

    @staticmethod
    def _extract_module_name(error_message: str) -> str:
        """Extract the module name from an import error message."""
        # "No module named 'bs4'"
        # "No module named 'foo.bar'"
        # "ImportError: cannot import name 'X' from 'Y'"
        lower = error_message.lower()

        # Pattern: No module named 'xxx'
        for quote in ["'", '"']:
            marker = f"no module named {quote}"
            idx = lower.find(marker)
            if idx != -1:
                start = idx + len(marker)
                end = lower.find(quote, start)
                if end != -1:
                    module = error_message[start:start + (end - start)]
                    # Take the top-level package: "foo.bar" -> "foo"
                    return module.split(".")[0]

        # Pattern: cannot import name 'X' from 'Y'
        marker = "from '"
        idx = lower.find(marker)
        if idx != -1:
            start = idx + len(marker)
            end = lower.find("'", start)
            if end != -1:
                return error_message[start:start + (end - start)].split(".")[0]

        return ""

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _get_installed_packages() -> frozenset:
        """Return a frozenset of installed package names (lowercased)."""
        installed = set()
        try:
            import importlib.metadata
            for dist in importlib.metadata.distributions():
                installed.add(dist.metadata["Name"].lower())
        except Exception:
            pass

        # Also add importable top-level names
        # (e.g. "pydantic" is both the pip name and the import name)
        try:
            import pkgutil
            for importer, modname, ispkg in pkgutil.iter_modules():
                installed.add(modname.lower())
        except Exception:
            pass

        return frozenset(installed)

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _get_stdlib_modules() -> frozenset:
        """Return a frozenset of stdlib module names."""
        if hasattr(sys, "stdlib_module_names"):
            return frozenset(m.lower() for m in sys.stdlib_module_names)
        # Fallback for Python <3.10
        return frozenset([
            "os", "sys", "json", "csv", "re", "math", "pathlib", "datetime",
            "collections", "itertools", "functools", "io", "tempfile",
            "subprocess", "threading", "logging", "unittest", "ast", "typing",
            "hashlib", "hmac", "secrets", "uuid", "socket", "http", "urllib",
            "xml", "html", "sqlite3", "argparse", "textwrap", "string",
            "copy", "pprint", "enum", "dataclasses", "abc", "contextlib",
            "shutil", "glob", "fnmatch", "stat", "time", "calendar",
            "random", "statistics", "decimal", "fractions",
        ])

    @classmethod
    def get_available_libraries_block(cls) -> str:
        """Build a prompt block showing available vs unavailable packages."""
        installed = cls._get_installed_packages()
        stdlib = cls._get_stdlib_modules()

        # Curated list of commonly-requested packages to check
        available = []
        not_installed = []

        for pkg in cls.COMMONLY_REQUESTED:
            pkg_lower = pkg.lower()
            # Check both the import name and the pip name
            import_name = pkg_lower
            pip_name = cls.IMPORT_TO_PACKAGE.get(pkg, pkg).lower()

            if import_name in installed or pip_name in installed:
                available.append(pkg)
            else:
                not_installed.append(pkg)

        lines = ["AVAILABLE PYTHON PACKAGES (installed):"]
        if available:
            lines.append(f"  {', '.join(sorted(available))}")
        else:
            lines.append("  (none of the commonly-requested packages are installed)")
        lines.append("NOT INSTALLED (do NOT import these):")
        if not_installed:
            lines.append(f"  {', '.join(sorted(not_installed))}")
        else:
            lines.append("  (all common packages are available)")
        lines.append("Standard library modules (json, os, csv, pathlib, etc.) are always available.")

        return "\n".join(lines)
