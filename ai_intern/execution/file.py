from ..schemas import TaskSchema, TaskOutput
from ..config import settings
from ..logging_config import get_logger
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
from typing import Optional

logger = get_logger("file")


class FileAgent:
    """Agent that reads user files and writes outputs."""

    USER_DATA_DIR = settings.USER_DATA_DIR
    OUTPUT_DIR = settings.OUTPUT_DIR

    model = settings.PLANNING_MODEL
    temperature = settings.VALIDATION_TEMP

    @classmethod
    def list_available_files(cls, subdirectory: str = "") -> list[str]:
        """List all files available in user_data."""
        search_path = cls.USER_DATA_DIR / subdirectory if subdirectory else cls.USER_DATA_DIR

        if not search_path.exists():
            return []

        files = []
        for file_path in search_path.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(cls.USER_DATA_DIR)
                files.append(str(rel_path))

        return sorted(files)

    @classmethod
    def read_file(cls, filepath: str) -> tuple[str, dict]:
        """Read a file from user_data. Returns (content, metadata)."""
        full_path = cls.USER_DATA_DIR / filepath

        if not full_path.exists():
            available = cls.list_available_files()
            available_str = ", ".join(available) if available else "none"
            raise FileNotFoundError(
                f"File '{filepath}' not found. Available files: {available_str}"
            )

        logger.info(f"Reading: {filepath}")

        stat = full_path.stat()
        metadata = {
            "filename": filepath,
            "size_bytes": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "extension": full_path.suffix
        }

        if filepath.endswith('.csv'):
            content = cls._read_csv_as_text(full_path)
            content = content.lstrip('\ufeff')  # Strip BOM if present
            # Also get structured CSV metadata
            metadata["csv_info"] = cls._get_csv_info(full_path)
        elif filepath.endswith('.json'):
            content = cls._read_json_as_text(full_path)
        else:
            content = full_path.read_text(encoding='utf-8')

        logger.info(f"Read {len(content)} characters")

        return content, metadata

    @staticmethod
    def _read_csv_as_text(filepath: Path) -> str:
        """Read CSV and format as readable text."""
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return "Empty CSV file"

        output = []
        output.append(f"CSV File: {len(rows)} rows\n")
        output.append("Columns: " + ", ".join(rows[0].keys()) + "\n")
        output.append("-" * 80 + "\n")

        for i, row in enumerate(rows, 1):
            output.append(f"Row {i}:\n")
            for key, value in row.items():
                output.append(f"  {key}: {value}\n")
            output.append("\n")

        return "".join(output)

    @staticmethod
    def _get_csv_info(filepath: Path) -> dict:
        """Get structured CSV metadata for context passing."""
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return {"columns": [], "row_count": 0, "first_row": {}}

        return {
            "columns": list(rows[0].keys()),
            "row_count": len(rows),
            "first_row": dict(rows[0]),
            "file_path": str(filepath),
        }

    @staticmethod
    def _read_json_as_text(filepath: Path) -> str:
        """Read JSON and format as readable text."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return json.dumps(data, indent=2)

    @classmethod
    def write_output(cls, filename: str, content: str) -> str:
        """Write content to outputs directory. Returns path to created file."""
        cls.OUTPUT_DIR.mkdir(exist_ok=True)

        if not any(char.isdigit() for char in filename):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if '.' in filename:
                name, ext = filename.rsplit('.', 1)
            else:
                name = filename
                ext = 'txt'
            filename = f"{timestamp}_{name}.{ext}"

        output_path = cls.OUTPUT_DIR / filename
        output_path.write_text(content, encoding='utf-8')

        logger.info(f"Wrote: {filename}")
        return str(output_path)

    @classmethod
    def execute_task(cls, task: TaskSchema, context: dict = None) -> tuple[TaskSchema, int]:
        """Execute file operation task."""
        logger.info(f"FileAgent processing task: {task.task_id}")
        logger.info(f"Goal: {task.goal[:80]}...")

        task.status = "executing"
        tokens_used = 0
        goal_lower = task.goal.lower()

        try:
            if "list" in goal_lower and "file" in goal_lower:
                result = cls._handle_list_files(task)
            elif "read" in goal_lower or "load" in goal_lower:
                result = cls._handle_read_file(task)
            elif "write" in goal_lower or "save" in goal_lower or "create" in goal_lower:
                result = cls._handle_write_file(task, context)
            else:
                task.status = "failed"
                task.error_message = "Could not determine file operation from task goal"
                logger.error("Unknown file operation")
                return task, tokens_used

            task.result = result
            task.status = "complete"
            task.completed_at = datetime.now(timezone.utc)

            logger.info("File operation complete")
            logger.debug(f"Result length: {len(result)} characters")

        except Exception as e:
            task.status = "failed"
            task.error_message = f"File operation failed: {str(e)}"
            logger.error(f"File operation failed: {e}")
            logger.error(f"Goal was: {task.goal}")

        return task, tokens_used

    @classmethod
    def _handle_list_files(cls, task: TaskSchema) -> str:
        """Handle listing available files."""
        files = cls.list_available_files()
        if not files:
            return "No files found in user_data/"
        result = ["Available files in user_data/:"]
        for file in files:
            result.append(f"  - {file}")
        return "\n".join(result)

    @classmethod
    def _handle_read_file(cls, task: TaskSchema) -> str:
        """Handle reading a file. Populates task_output with structured data."""
        filename = cls._extract_filename(task.goal)

        # Fuzzy match if exact filename not found (#20)
        if not filename:
            filename = cls._fuzzy_match_filename(task.goal)

        if not filename:
            raise ValueError("Could not determine which file to read from task goal")

        content, metadata = cls.read_file(filename)

        # Build structured TaskOutput (#5)
        full_path = cls.USER_DATA_DIR / filename
        task_output = TaskOutput(
            output_type="data" if filename.endswith(('.csv', '.json')) else "text",
            raw_result=content,
            file_path=str(full_path),
        )

        # Add CSV-specific structured data
        if metadata.get("csv_info"):
            csv_info = metadata["csv_info"]
            task_output.column_names = csv_info["columns"]
            task_output.row_count = csv_info["row_count"]
            first_row_str = ", ".join(f"{k}: {v}" for k, v in csv_info["first_row"].items())
            task_output.data_summary = (
                f"CSV with {csv_info['row_count']} rows and columns: "
                f"{', '.join(csv_info['columns'])}. "
                f"First row: {first_row_str}. "
                f"File path: {csv_info['file_path']}"
            )

        task.task_output = task_output

        result = [
            f"File: {metadata['filename']}",
            f"Size: {metadata['size_bytes']} bytes",
            f"Modified: {metadata['modified']}",
            "",
            "Content:",
            "-" * 80,
            content
        ]
        return "\n".join(result)

    @classmethod
    def _handle_write_file(cls, task: TaskSchema, context: dict = None) -> str:
        """Handle writing output file."""
        filename = cls._extract_filename(task.goal)
        if not filename:
            filename = "output.txt"

        content = None

        if context and 'previous_tasks' in context and len(context['previous_tasks']) > 0:
            goal_lower = task.goal.lower()

            if filename and filename.endswith('.py'):
                prev_task = context['previous_tasks'][-1]
                if prev_task['status'] in ['validated', 'complete']:
                    content = prev_task['result']
                    logger.info(f"Using code from Task {prev_task['order']} (detected .py extension)")

            elif any(keyword in goal_lower for keyword in ['previous', 'from task', 'research', 'result', 'above', 'earlier',
                                                            'calculator', 'code', 'script', 'function', 'summary', 'analysis']):
                prev_task = context['previous_tasks'][-1]
                if prev_task['status'] in ['validated', 'complete']:
                    content = prev_task['result']
                    logger.info(f"Using result from Task {prev_task['order']}")
                else:
                    logger.warning(f"Previous task (Task {prev_task['order']}) failed with status: {prev_task['status']}")
                    raise ValueError(f"Cannot save output from failed task {prev_task['order']}")

        if content is None:
            content = task.result if task.result else "Output content placeholder"

        if content and content.strip().startswith(('def ', 'import ', 'from ', 'class ')):
            if filename and not filename.endswith('.py'):
                base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
                filename = f"{base_name}.py"
                logger.info("Detected Python code, using .py extension")

        output_path = cls.write_output(filename, content)
        return f"Wrote output to: {output_path}"

    @classmethod
    def _fuzzy_match_filename(cls, goal: str) -> Optional[str]:
        """Fuzzy match against available files if no exact filename found."""
        available = cls.list_available_files()
        goal_lower = goal.lower()

        for f in available:
            base = Path(f).stem.lower()
            if base in goal_lower:
                logger.info(f"Fuzzy matched '{f}' from goal")
                return f

        # If only one file of a type exists, match by extension keyword
        if "csv" in goal_lower:
            csvs = [f for f in available if f.endswith('.csv')]
            if len(csvs) == 1:
                logger.info(f"Auto-matched only CSV: {csvs[0]}")
                return csvs[0]

        return None

    @staticmethod
    def _extract_filename(goal: str) -> Optional[str]:
        """Extract filename from task goal."""
        extensions = ['.csv', '.json', '.txt', '.md', '.log', '.py']

        words = goal.split()
        for i, word in enumerate(words):
            if any(ext in word.lower() for ext in extensions):
                filename = word.strip('",\'')
                if '.' in filename:
                    name, ext = filename.rsplit('.', 1)
                    for prefix in ['outputs/', 'output/', 'user_data/']:
                        if name.lower().startswith(prefix):
                            name = name[len(prefix):]
                            break
                    filename = f"{name}.{ext}"
                return filename

        return None
