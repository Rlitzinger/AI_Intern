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
- Only import packages listed in the AVAILABLE PYTHON PACKAGES block.
- If a needed package is listed under NOT INSTALLED, use an alternative or stdlib.

Your response should start directly with Python code (imports or function definitions)."""

    @staticmethod
    def build_prompt(task: TaskSchema, context: dict = None) -> str:
        """Build the code generation prompt from a task."""

        prompt_parts = []

        # Inject available library awareness (lazy import to avoid circular dependency)
        from ..validation.error_classifier import ErrorClassifier
        lib_block = ErrorClassifier.get_available_libraries_block()
        prompt_parts.append(f"{lib_block}\n\n")

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

                # Add structured key-value data if available
                if prev_task.get('key_values'):
                    kv = prev_task['key_values']
                    kv_lines = [f"  {k}: {v}" for k, v in kv.items()]
                    prompt_parts.append("Structured data from research:\n" + "\n".join(kv_lines) + "\n")
                    prompt_parts.append("Use these exact values in your code.\n")

            prompt_parts.append("Use the above information when writing your code.\n\n")

        # Inject workspace context for app-scale tasks with component dependencies
        if context and context.get('workspace_context'):
            prompt_parts.append("AVAILABLE COMPONENTS (already implemented — import these, do not rewrite):\n")
            prompt_parts.append(context['workspace_context'] + "\n")
            if context.get('workspace_imports'):
                prompt_parts.append("\nRequired imports for this task:\n")
                prompt_parts.append(context['workspace_imports'] + "\n")
            prompt_parts.append("\nYour code must import from the paths shown above.\n\n")

        # Budget check: truncate context if prompt is too large
        context_text = "".join(prompt_parts)
        budget = settings.MAX_PROMPT_TOKENS - 500  # reserve 500 for task + instructions
        if estimate_tokens(context_text) > budget:
            prompt_parts = [truncate_to_token_budget(context_text, budget)]

        # Add structured error history for retries (#9 + Tier 2 Phase 3)
        if context and context.get('error_history'):
            latest = context['error_history'][-1]
            attempt_num = latest.get('attempt', '?')
            max_attempts = len(context['error_history']) + 1  # rough estimate
            correction = CodingAgent._build_correction_prompt(latest)
            prompt_parts.append(correction)

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
    def _build_correction_prompt(error_entry: dict) -> str:
        """Build a category-aware correction prompt for retries."""
        category = error_entry.get('category', 'retryable_logic')
        error_msg = error_entry.get('error', 'Unknown error')
        attempt = error_entry.get('attempt', 1)
        test_code = error_entry.get('test_code', '')

        lines = [f"=== RETRY (attempt {attempt + 1}) ==="]

        if category == "retryable_syntax":
            lines.append("Your previous code had a SYNTAX ERROR and could not be parsed.")
            lines.append(f"Error: {error_msg}")
            lines.append("")
            lines.append("Fix: Ensure all brackets, quotes, and indentation are correct.")
            lines.append("Write the COMPLETE corrected code.")

        elif category == "retryable_runtime":
            lines.append("Your previous code had a RUNTIME CRASH during testing.")
            lines.append(f"Error: {error_msg}")
            if test_code:
                lines.append(f"\nFailing test:\n{test_code}")
            lines.append("")
            lines.append("Fix: Check all variables are defined before use and argument types are correct.")
            lines.append("Do NOT change function signatures or class names.")
            lines.append("Write the COMPLETE corrected code.")

        else:
            # retryable_logic (default) — test assertion failures
            lines.append("Your previous code FAILED TESTS. It ran but produced wrong results.")
            lines.append(f"Error: {error_msg}")
            if test_code:
                lines.append(f"\nFailing test:\n{test_code}")
            lines.append("")
            lines.append("Fix: Focus on the logic error indicated by the test.")
            lines.append("Do NOT change function signatures or class names.")
            lines.append("Write the COMPLETE corrected code.")

        lines.append("")
        return "\n".join(lines) + "\n"

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
