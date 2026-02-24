"""Tests for TaskClassifier verdict mapping (pure logic, no LLM)."""
import pytest
from ai_intern.planning.classifier import TaskClassifier, TaskVerdict


class TestMapToVerdict:
    def test_simple_clear(self):
        assert TaskClassifier._map_to_verdict(False, False) == TaskVerdict.EXECUTE

    def test_complex_clear(self):
        assert TaskClassifier._map_to_verdict(True, False) == TaskVerdict.DECOMPOSE

    def test_simple_ambiguous(self):
        assert TaskClassifier._map_to_verdict(False, True) == TaskVerdict.CLARIFY

    def test_complex_ambiguous(self):
        assert TaskClassifier._map_to_verdict(True, True) == TaskVerdict.CLARIFY_THEN_DECOMPOSE


class TestTaskVerdictValues:
    def test_all_verdict_values(self):
        assert TaskVerdict.EXECUTE.value == "execute"
        assert TaskVerdict.DECOMPOSE.value == "decompose"
        assert TaskVerdict.CLARIFY.value == "clarify"
        assert TaskVerdict.CLARIFY_THEN_DECOMPOSE.value == "clarify_then_decompose"
