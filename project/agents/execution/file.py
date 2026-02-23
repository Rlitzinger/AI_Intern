from schemas import TaskSchema
from datetime import datetime
from pathlib import Path
import csv
import json
from typing import Optional


class FileAgent:
    """Agent that reads user files and writes outputs."""

    # Paths relative to project/ directory
    USER_DATA_DIR = Path(__file__).parent.parent.parent.parent / "user_data"
    OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "outputs"

    model = "qwen2.5:7b-instruct"
    temperature = 0.2

    @classmethod
    def list_available_files(cls, subdirectory: str = "") -> list[str]:
        """
        List all files available in user_data.

        Args:
            subdirectory: Optional subdirectory to search (e.g., "workouts")

        Returns:
            List of file paths relative to user_data/
        """
        search_path = cls.USER_DATA_DIR / subdirectory if subdirectory else cls.USER_DATA_DIR

        if not search_path.exists():
            return []

        files = []
        for file_path in search_path.rglob("*"):
            if file_path.is_file():
                # Get path relative to USER_DATA_DIR
                rel_path = file_path.relative_to(cls.USER_DATA_DIR)
                files.append(str(rel_path))

        return sorted(files)

    @classmethod
    def read_file(cls, filepath: str) -> tuple[str, dict]:
        """
        Read a file from user_data.

        Args:
            filepath: Path relative to user_data/ (e.g., "Workouts.csv")

        Returns:
            tuple: (file_content_as_string, metadata)
        """
        full_path = cls.USER_DATA_DIR / filepath

        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        print(f"      📂 Reading: {filepath}")

        # Get file metadata
        stat = full_path.stat()
        metadata = {
            "filename": filepath,
            "size_bytes": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "extension": full_path.suffix
        }

        # Read based on file type
        if filepath.endswith('.csv'):
            content = cls._read_csv_as_text(full_path)
        elif filepath.endswith('.json'):
            content = cls._read_json_as_text(full_path)
        else:
            content = full_path.read_text(encoding='utf-8')

        print(f"      ✓ Read {len(content)} characters")

        return content, metadata

    @staticmethod
    def _read_csv_as_text(filepath: Path) -> str:
        """Read CSV and format as readable text."""
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return "Empty CSV file"

        # Format as text table
        output = []
        output.append(f"CSV File: {len(rows)} rows\n")
        output.append("Columns: " + ", ".join(rows[0].keys()) + "\n")
        output.append("-" * 80 + "\n")

        # Add all rows
        for i, row in enumerate(rows, 1):
            output.append(f"Row {i}:\n")
            for key, value in row.items():
                output.append(f"  {key}: {value}\n")
            output.append("\n")

        return "".join(output)

    @staticmethod
    def _read_json_as_text(filepath: Path) -> str:
        """Read JSON and format as readable text."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Pretty print JSON
        return json.dumps(data, indent=2)

    @classmethod
    def write_output(cls, filename: str, content: str) -> str:
        """
        Write content to outputs directory.

        Args:
            filename: Name of file to create
            content: Content to write

        Returns:
            Path to created file
        """
        # Create outputs directory if it doesn't exist
        cls.OUTPUT_DIR.mkdir(exist_ok=True)

        # Add timestamp to filename if not present
        if not any(char.isdigit() for char in filename):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            if '.' in filename:
                name, ext = filename.rsplit('.', 1)
            else:
                name = filename
                ext = 'txt'  # Caller should have set extension already via _handle_write_file

            filename = f"{timestamp}_{name}.{ext}"

        output_path = cls.OUTPUT_DIR / filename
        output_path.write_text(content, encoding='utf-8')

        print(f"      📝 Wrote: {filename}")

        return str(output_path)

    @classmethod
    def execute_task(cls, task: TaskSchema, context: dict = None) -> tuple[TaskSchema, int]:
        """
        Execute file operation task.

        Detects whether task is:
        - Reading a file
        - Writing output
        - Listing files

        Returns:
            tuple: (updated_task, tokens_used)
        """
        print(f"\n📁 FileAgent processing task: {task.task_id}")
        print(f"📝 Goal: {task.goal[:80]}...")

        task.status = "executing"
        tokens_used = 0

        goal_lower = task.goal.lower()

        try:
            # Detect operation type
            if "list" in goal_lower and "file" in goal_lower:
                # List available files
                result = cls._handle_list_files(task)

            elif "read" in goal_lower or "load" in goal_lower:
                # Read a file
                result = cls._handle_read_file(task)

            elif "write" in goal_lower or "save" in goal_lower or "create" in goal_lower:
                # Write output
                result = cls._handle_write_file(task, context)

            else:
                # Unknown operation
                task.status = "failed"
                task.error_message = "Could not determine file operation from task goal"
                print(f"❌ Unknown file operation")
                return task, tokens_used

            # Store result
            task.result = result
            task.status = "complete"
            task.completed_at = datetime.utcnow()

            print(f"✅ File operation complete")
            print(f"   Result length: {len(result)} characters")

        except Exception as e:
            task.status = "failed"
            task.error_message = f"File operation failed: {str(e)}"
            print(f"❌ File operation failed: {e}")
            print(f"   Goal was: {task.goal}")
            if context and context.get('previous_tasks'):
                print(f"   Had {len(context['previous_tasks'])} previous task(s) in context")

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
        """Handle reading a file."""
        # Extract filename from goal
        filename = cls._extract_filename(task.goal)

        if not filename:
            raise ValueError("Could not determine which file to read from task goal")

        # Read the file
        content, metadata = cls.read_file(filename)

        # Format result
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
        # Extract filename
        filename = cls._extract_filename(task.goal)
        if not filename:
            filename = "output.txt"

        # Determine content source
        content = None

        # Check if task references previous task result
        if context and 'previous_tasks' in context and len(context['previous_tasks']) > 0:
            goal_lower = task.goal.lower()

            # If saving a .py file, likely want code from previous task
            if filename and filename.endswith('.py'):
                prev_task = context['previous_tasks'][-1]
                # Only use if previous task succeeded
                if prev_task['status'] in ['validated', 'complete']:
                    content = prev_task['result']
                    print(f"      ✓ Using code from Task {prev_task['order']} (detected .py extension)")

            # Otherwise check keywords
            elif any(keyword in goal_lower for keyword in ['previous', 'from task', 'research', 'result', 'above', 'earlier',
                                                            'calculator', 'code', 'script', 'function']):
                prev_task = context['previous_tasks'][-1]

                # Only use result if previous task succeeded
                if prev_task['status'] in ['validated', 'complete']:
                    content = prev_task['result']
                    print(f"      ✓ Using result from Task {prev_task['order']}")
                else:
                    # Previous task failed - cannot save its output
                    print(f"      ⚠️  Previous task (Task {prev_task['order']}) failed with status: {prev_task['status']}")
                    raise ValueError(f"Cannot save output from failed task {prev_task['order']}")

        # Fallback to task.result or placeholder
        if content is None:
            content = task.result if task.result else "Output content placeholder"

        # Write the file - if content looks like Python code, ensure .py extension
        if content and content.strip().startswith(('def ', 'import ', 'from ', 'class ')):
            if filename and not filename.endswith('.py'):
                base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
                filename = f"{base_name}.py"
                print(f"      ℹ️  Detected Python code, using .py extension")

        output_path = cls.write_output(filename, content)

        return f"Wrote output to: {output_path}"

    @staticmethod
    def _extract_filename(goal: str) -> Optional[str]:
        """
        Extract filename from task goal.

        Examples:
        - "Read Workouts.csv" → "Workouts.csv"
        - "Load file FoodLog.csv" → "FoodLog.csv"
        - "Read workouts/2024.csv" → "workouts/2024.csv"
        """
        goal_lower = goal.lower()

        # Look for common file extensions
        extensions = ['.csv', '.json', '.txt', '.md', '.log', '.py']

        words = goal.split()
        for i, word in enumerate(words):
            # Check if word contains a file extension
            if any(ext in word.lower() for ext in extensions):
                # Clean up the filename (remove quotes, commas, etc.)
                filename = word.strip('",\'')

                # Ensure we keep the extension by splitting on the last dot first
                if '.' in filename:
                    name, ext = filename.rsplit('.', 1)
                    # Remove any path prefixes from the name part only
                    for prefix in ['outputs/', 'output/', 'user_data/']:
                        if name.lower().startswith(prefix):
                            name = name[len(prefix):]
                            break
                    filename = f"{name}.{ext}"
                return filename

        return None
