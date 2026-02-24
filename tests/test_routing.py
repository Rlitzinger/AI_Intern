"""Tests for TaskRouter classification using scoring system."""
import pytest
from ai_intern.schemas import TaskSchema
from ai_intern.orchestration.routing import TaskRouter


def _make_task(goal: str) -> TaskSchema:
    return TaskSchema(plan_id="test", task_order=0, goal=goal)


class TestTaskRouterClassification:
    """Test suite for TaskRouter.classify_task() - 25+ edge cases."""

    # === FILE operations ===
    def test_read_csv(self):
        assert TaskRouter.classify_task(_make_task("Read file Workouts.csv")) == "file"

    def test_read_json(self):
        assert TaskRouter.classify_task(_make_task("Load data from config.json")) == "file"

    def test_save_to_outputs(self):
        assert TaskRouter.classify_task(_make_task("Save to outputs/summary.txt")) == "file"

    def test_save_results(self):
        assert TaskRouter.classify_task(_make_task("Save the analysis results")) == "file"

    def test_list_files(self):
        assert TaskRouter.classify_task(_make_task("List files in user_data")) == "file"

    def test_write_to_file(self):
        assert TaskRouter.classify_task(_make_task("Write to file output.txt")) == "file"

    def test_export_csv(self):
        assert TaskRouter.classify_task(_make_task("Export to results.csv")) == "file"

    def test_save_as_py(self):
        assert TaskRouter.classify_task(_make_task("Save the code as calculator.py")) == "file"

    # === CODE operations ===
    def test_write_function(self):
        assert TaskRouter.classify_task(_make_task("Write a function to calculate fibonacci")) == "code"

    def test_implement_class(self):
        assert TaskRouter.classify_task(_make_task("Implement a Stack class")) == "code"

    def test_create_script(self):
        assert TaskRouter.classify_task(_make_task("Create a script to process data")) == "code"

    def test_build_flask_app(self):
        assert TaskRouter.classify_task(_make_task("Build a Flask hello world app")) == "code"

    def test_write_python_function(self):
        assert TaskRouter.classify_task(_make_task("Write a Python function that converts Fahrenheit to Celsius")) == "code"

    # === RESEARCH operations ===
    def test_research_topic(self):
        assert TaskRouter.classify_task(_make_task("Research the benefits of creatine")) == "research"

    def test_investigate(self):
        assert TaskRouter.classify_task(_make_task("Investigate Python web frameworks")) == "research"

    def test_find_information(self):
        assert TaskRouter.classify_task(_make_task("Find information about protein in chicken")) == "research"

    def test_search_for(self):
        assert TaskRouter.classify_task(_make_task("Search for best practices in error handling")) == "research"

    # === ANALYSIS operations ===
    def test_calculate_average(self):
        assert TaskRouter.classify_task(_make_task("Calculate average calories from the workout data")) == "analysis"

    def test_analyze_data(self):
        assert TaskRouter.classify_task(_make_task("Analyze my workout data")) == "analysis"

    def test_summarize_data(self):
        assert TaskRouter.classify_task(_make_task("Summarize the sales statistics")) == "analysis"

    def test_compare_values(self):
        assert TaskRouter.classify_task(_make_task("Compare calories vs duration")) == "analysis"

    # === EDGE CASES ===
    def test_create_summary_file_is_file(self):
        """'Create a summary file' should be file, not code."""
        result = TaskRouter.classify_task(_make_task("Create a summary file from the results"))
        assert result == "file"

    def test_write_results_to_output(self):
        """'Write results to output' should be file."""
        assert TaskRouter.classify_task(_make_task("Write the results to output file")) == "file"

    def test_unknown_defaults_gracefully(self):
        """Totally ambiguous task should return something."""
        result = TaskRouter.classify_task(_make_task("do something"))
        assert result in ("code", "unknown")

    def test_empty_goal(self):
        result = TaskRouter.classify_task(_make_task(""))
        assert result == "unknown"
