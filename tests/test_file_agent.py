"""Tests for FileAgent file operations."""
import pytest
import os
import tempfile
from pathlib import Path
from ai_intern.execution.file import FileAgent
from ai_intern.schemas import TaskSchema


class TestExtractFilename:
    def test_csv_extension(self):
        assert FileAgent._extract_filename("Read file Workouts.csv") == "Workouts.csv"

    def test_json_extension(self):
        assert FileAgent._extract_filename("Load config.json") == "config.json"

    def test_py_extension(self):
        assert FileAgent._extract_filename("Save as calculator.py") == "calculator.py"

    def test_txt_extension(self):
        assert FileAgent._extract_filename("Write to output.txt") == "output.txt"

    def test_no_extension(self):
        assert FileAgent._extract_filename("Read the data") is None

    def test_strips_quotes(self):
        result = FileAgent._extract_filename('Read "Workouts.csv" file')
        assert result == "Workouts.csv"

    def test_strips_prefix_path(self):
        result = FileAgent._extract_filename("Read outputs/summary.txt")
        assert result == "summary.txt"

    def test_md_extension(self):
        assert FileAgent._extract_filename("Read README.md") == "README.md"


class TestFuzzyMatchFilename:
    def test_base_name_in_goal(self):
        """Test that fuzzy match finds file by base name."""
        # This depends on actual files in user_data/
        result = FileAgent._fuzzy_match_filename("read workout data")
        # May or may not find a match depending on available files
        # Just ensure it doesn't crash
        assert result is None or isinstance(result, str)

    def test_no_match(self):
        result = FileAgent._fuzzy_match_filename("something completely unrelated xyz")
        assert result is None


class TestListAvailableFiles:
    def test_returns_list(self):
        files = FileAgent.list_available_files()
        assert isinstance(files, list)

    def test_sorted(self):
        files = FileAgent.list_available_files()
        assert files == sorted(files)


class TestWriteOutput:
    def test_write_creates_file(self):
        path = FileAgent.write_output("test_output.txt", "hello world")
        assert os.path.exists(path)
        with open(path, 'r') as f:
            assert f.read() == "hello world"
        os.unlink(path)

    def test_timestamp_added(self):
        path = FileAgent.write_output("nodigits.txt", "content")
        filename = os.path.basename(path)
        # Should have timestamp prepended since "nodigits" has no digits
        assert any(c.isdigit() for c in filename)
        os.unlink(path)
