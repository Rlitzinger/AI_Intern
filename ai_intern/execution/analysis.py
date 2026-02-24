from ..schemas import TaskSchema, TaskOutput
from ..llm import call_ollama_code
from ..config import settings
from ..logging_config import get_logger
from ..execution.file import FileAgent
from datetime import datetime, timezone
import subprocess
import tempfile
import os

logger = get_logger("analysis")


class AnalysisAgent:
    """Agent that analyzes data by generating and executing analysis code."""

    model = settings.CODING_MODEL
    temperature = settings.CODING_TEMP

    system_prompt = """You are a Python data analyst.
Write clean analysis code using pandas or csv module.
The code MUST print its results to stdout.
Include all necessary imports.

CRITICAL RULES:
- Return ONLY raw Python code
- NO markdown formatting
- NO example usage
- The script must PRINT its analysis results
- Use pandas if available, fall back to csv module
- Handle missing/invalid data gracefully"""

    @classmethod
    def execute_task(cls, task: TaskSchema, context: dict = None) -> tuple[TaskSchema, int]:
        """Execute analysis task: generate analysis code, run it, capture output."""
        logger.info(f"AnalysisAgent processing task: {task.task_id}")
        logger.info(f"Goal: {task.goal[:80]}...")

        task.status = "executing"
        tokens_used = 0

        try:
            # Build prompt with file/data context
            prompt = cls._build_analysis_prompt(task, context)

            # Generate analysis code
            code, tokens = call_ollama_code(
                model=cls.model,
                prompt=prompt,
                system=cls.system_prompt,
                temperature=cls.temperature,
            )
            tokens_used += tokens

            # Strip markdown fences
            code = code.strip()
            if code.startswith("```python"):
                code = code[len("```python"):].strip()
            if code.startswith("```"):
                code = code[3:].strip()
            if code.endswith("```"):
                code = code[:-3].strip()

            logger.debug(f"Generated analysis code ({len(code)} chars)")

            # Execute the analysis code
            output, success = cls._run_analysis(code)

            if success:
                task.result = output
                task.status = "complete"
                task.completed_at = datetime.now(timezone.utc)
                task.task_output = TaskOutput(
                    output_type="data",
                    raw_result=output,
                    data_summary=output[:500] if output else "No output",
                )
                logger.info(f"Analysis complete: {len(output)} chars output")
            else:
                task.status = "failed"
                task.error_message = f"Analysis code failed: {output}"
                logger.error(f"Analysis failed: {output[:200]}")

        except Exception as e:
            task.status = "failed"
            task.error_message = f"Analysis error: {str(e)}"
            logger.error(f"AnalysisAgent error: {e}")

        return task, tokens_used

    @classmethod
    def _build_analysis_prompt(cls, task: TaskSchema, context: dict = None) -> str:
        """Build prompt for analysis code generation."""
        parts = []

        # Add data context from previous tasks
        if context and context.get("previous_tasks"):
            for prev in context["previous_tasks"]:
                if prev.get("data_summary"):
                    parts.append(f"Available data: {prev['data_summary']}\n")
                if prev.get("file_path"):
                    parts.append(f"Data file path: {prev['file_path']}\n")
                elif prev.get("result"):
                    parts.append(f"Previous result:\n{prev['result'][:500]}\n")

        # If no file context, check if we can find files from the goal
        if not parts:
            available = FileAgent.list_available_files()
            if available:
                parts.append(f"Available data files in user_data/: {', '.join(available)}\n")
                # Try to find relevant file from goal
                goal_lower = task.goal.lower()
                for f in available:
                    if f.lower().replace('.csv', '').replace('.json', '') in goal_lower:
                        file_path = str(FileAgent.USER_DATA_DIR / f)
                        parts.append(f"Relevant file path: {file_path}\n")
                        break

        parts.append(f"\nAnalysis task: {task.goal}\n")
        parts.append(
            "\nWrite a Python script that performs this analysis and PRINTS the results. "
            "Use the exact file paths provided above. "
            "The output should be human-readable."
        )

        return "".join(parts)

    @staticmethod
    def _run_analysis(code: str) -> tuple[str, bool]:
        """Run analysis code in subprocess and capture output."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            temp_path = f.name

        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            result = subprocess.run(
                ["python", temp_path],
                capture_output=True,
                text=True,
                timeout=settings.CODE_EXEC_TIMEOUT,
                encoding="utf-8",
                env=env,
            )

            if result.returncode == 0:
                return result.stdout.strip(), True
            else:
                return result.stderr.strip(), False

        except subprocess.TimeoutExpired:
            return f"Analysis timed out after {settings.CODE_EXEC_TIMEOUT}s", False
        except Exception as e:
            return f"Execution error: {str(e)}", False
        finally:
            os.unlink(temp_path)
