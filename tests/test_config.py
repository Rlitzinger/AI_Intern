"""Tests for configuration management."""
import pytest
from ai_intern.config import Settings, settings


class TestSettings:
    def test_default_values(self):
        s = Settings()
        assert s.PLANNING_MODEL == "qwen2.5:7b-instruct"
        assert s.CODING_MODEL == "qwen2.5-coder:7b-instruct"
        assert s.MAX_RETRIES == 2
        assert s.CODE_EXEC_TIMEOUT == 15
        assert s.USE_HIERARCHICAL_PLANNING is True

    def test_temperatures_in_range(self):
        s = Settings()
        assert 0 <= s.PLANNING_TEMP <= 1
        assert 0 <= s.CODING_TEMP <= 1
        assert 0 <= s.VALIDATION_TEMP <= 1
        assert 0 <= s.RESEARCH_TEMP <= 1

    def test_singleton_exists(self):
        assert settings is not None
        assert isinstance(settings, Settings)

    def test_paths_exist(self):
        from pathlib import Path
        assert isinstance(settings.USER_DATA_DIR, Path)
        assert isinstance(settings.OUTPUT_DIR, Path)
        assert isinstance(settings.DB_PATH, Path)


class TestLLMHelpers:
    def test_estimate_tokens(self):
        from ai_intern.llm import estimate_tokens
        assert estimate_tokens("") == 0
        assert estimate_tokens("hello world") > 0
        # Rough check: ~3-4 chars per token
        assert 2 <= estimate_tokens("a" * 12) <= 5

    def test_truncate_to_budget(self):
        from ai_intern.llm import truncate_to_token_budget
        short = "hello"
        assert truncate_to_token_budget(short, 100) == short

        long_text = "a" * 10000
        result = truncate_to_token_budget(long_text, 10)
        assert len(result) < len(long_text)
        assert "[truncated]" in result
