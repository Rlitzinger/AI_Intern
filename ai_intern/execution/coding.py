from ..schemas import TaskSchema
from ..llm import call_ollama_code, estimate_tokens, truncate_to_token_budget
from ..config import settings
from ..logging_config import get_logger
from datetime import datetime, timezone

logger = get_logger("coding")


class CodingAgent:
    """Coding agent that generates Python code for tasks."""

    model = settings.CODING_MODEL
    temperature = settings.CODING_TEMP
    system_prompt = """You are an expert Python developer.

Write clean, working Python code that accomplishes the given task.
Include necessary imports, error handling, and brief comments.
Follow PEP 8 style guidelines.

CRITICAL RULES:
- Return ONLY raw Python code
- NO markdown formatting (no ```python``` blocks)
- NO example usage or test code
- NO if __name__ == "__main__" blocks
- NO explanatory text before or after the code
- Code should be importable with no side effects

Your response should start directly with Python code (imports or function definitions)."""

    @staticmethod
    def build_prompt(task: TaskSchema, context: dict = None) -> str:
        """Build the code generation prompt from a task."""

        prompt_parts = []

        # Add context if available (#5 - prefer data_summary over raw result)
        if context and context.get('previous_tasks'):
            prompt_parts.append("CONTEXT FROM PREVIOUS TASKS:\n")
            for prev_task in context['previous_tasks']:
                prompt_parts.append(f"Task {prev_task['order']}: {prev_task['goal']}\n")

                # Prefer compact data_summary over raw result
                if prev_task.get('data_summary'):
                    prompt_parts.append(f"Data: {prev_task['data_summary']}\n")
                    if prev_task.get('file_path'):
                        prompt_parts.append(f"File path: {prev_task['file_path']}\n")
                elif prev_task.get('result'):
                    result_preview = prev_task['result'][:500]
                    prompt_parts.append(f"Result:\n{result_preview}\n")

            prompt_parts.append("Use the above information when writing your code.\n\n")

        # Budget check: truncate context if prompt is too large
        context_text = "".join(prompt_parts)
        budget = settings.MAX_PROMPT_TOKENS - 500  # reserve 500 for task + instructions
        if estimate_tokens(context_text) > budget:
            prompt_parts = [truncate_to_token_budget(context_text, budget)]

        # Add error history for retries (#9)
        if context and context.get('error_history'):
            latest = context['error_history'][-1]
            prompt_parts.append("PREVIOUS ATTEMPT FAILED:\n")
            prompt_parts.append(f"Error: {latest['error']}\n")
            if latest.get('test_code'):
                prompt_parts.append(f"Failed test:\n{latest['test_code']}\n")
            prompt_parts.append("Fix the issue in your code.\n\n")

        # Add the main task
        prompt_parts.append(f"Task: {task.goal}\n")
        prompt_parts.append("""
Write complete Python code to accomplish this task.

Requirements:
- Include all necessary imports at the top
- Add error handling for common failures
- Use clear variable names
- Add brief comments explaining the logic
- Return ONLY the code - no markdown, no examples, no explanations

Your response must start with valid Python code (either imports or a function definition).
Do NOT wrap in ```python``` or ``` blocks.""")

        return "".join(prompt_parts)

    @classmethod
    def execute_task(cls, task: TaskSchema, context: dict = None) -> tuple[TaskSchema, int]:
        """
        Execute the coding agent to generate code for a task.
        Returns (updated_task, tokens_used).
        """
        logger.info(f"CodingAgent processing task: {task.task_id}")
        logger.info(f"Goal: {task.goal[:100]}...")

        task.status = "executing"

        prompt = cls.build_prompt(task, context)

        # Generate code
        code, tokens = call_ollama_code(
            model=cls.model,
            prompt=prompt,
            system=cls.system_prompt,
            temperature=cls.temperature
        )

        # Strip markdown blocks if present
        code = code.strip()
        if code.startswith("```python"):
            code = code[len("```python"):].strip()
        if code.startswith("```"):
            code = code[3:].strip()
        if code.endswith("```"):
            code = code[:-3].strip()

        # Store cleaned code
        task.result = code
        task.status = "complete"
        task.completed_at = datetime.now(timezone.utc)

        logger.info(f"Code generated ({len(code)} characters)")
        logger.debug(f"Tokens consumed: {tokens}")

        return task, tokens

    @staticmethod
    def _strip_example_usage(code: str) -> str:
        """
        Remove example usage comments and code at the end of generated code.
        Keeps only function/class definitions and imports.
        """
        lines = code.split('\n')
        cleaned_lines = []

        inside_function = False
        function_indent = 0

        for line in lines:
            stripped = line.strip()

            # Track if we're inside a function definition
            if stripped.startswith('def ') or stripped.startswith('class '):
                inside_function = True
                function_indent = len(line) - len(line.lstrip())
                cleaned_lines.append(line)
                continue

            # If we're inside a function, keep lines that are indented
            if inside_function:
                current_indent = len(line) - len(line.lstrip())

                # If line is at same or lower indent than function def, we've left the function
                if stripped and current_indent <= function_indent:
                    inside_function = False
                    # Check if this is a new function/class or example usage
                    if stripped.startswith('def ') or stripped.startswith('class '):
                        cleaned_lines.append(line)
                        inside_function = True
                        function_indent = current_indent
                    elif stripped.startswith('#'):
                        # Comment after function - likely example usage
                        break
                    else:
                        # Code after function - likely example usage
                        break
                else:
                    # Still inside function, keep the line
                    cleaned_lines.append(line)
            else:
                # Not inside a function - could be import or example usage
                if stripped.startswith('import ') or stripped.startswith('from '):
                    cleaned_lines.append(line)
                elif stripped.startswith('#'):
                    # Comment outside function - likely example usage
                    break
                elif stripped == '':
                    # Empty line - keep it (might be between functions)
                    cleaned_lines.append(line)
                else:
                    # Code outside function - likely example usage
                    break

        return '\n'.join(cleaned_lines).rstrip()
