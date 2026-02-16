from schemas import RequestSchema, PlanSchema, TaskSchema
from llm import call_ollama_structured
from llm import call_ollama_code
from datetime import datetime
import tempfile
import subprocess
import os
from storage import save_plan_to_sqlite, load_plan_from_sqlite

class Orchestrator:
    """Orchestrates the full pipeline with error recovery."""
    
    def __init__(self, max_retries: int = 2):
        """
        Initialize orchestrator.
        
        Args:
            max_retries: Maximum retry attempts per task (default: 2)
                        Total attempts = 1 + max_retries (so 3 total with default)
        """
        self.max_retries = max_retries
    
    def execute_request(self, request: RequestSchema) -> PlanSchema:
        """
        Execute full pipeline: Planning → Execution → Validation (with retries).
        
        Args:
            request: The user's request
            
        Returns:
            PlanSchema: Completed plan with all tasks executed and validated
        """
        print("\n" + "=" * 80)
        print("🎯 ORCHESTRATOR: Starting Request Execution")
        print("=" * 80)
        
        # === STAGE 1: PLANNING ===
        print(f"\n📋 Stage 1: Planning")
        plan = PlanningAgent.create_plan(request)
        print(f"   Generated {len(plan.tasks)} task(s)")
        
        save_plan_to_sqlite(plan)
        print(f"   💾 Saved plan after planning")
        
        # === STAGE 2: EXECUTION + VALIDATION (with retries) ===
        print(f"\n💻 Stage 2: Execution & Validation")
        
        for i, task in enumerate(plan.tasks):
            print(f"\n{'─' * 80}")
            print(f"Task {i}: {task.goal[:60]}...")
            print(f"{'─' * 80}")
            
            success = self._execute_task_with_retry(task, plan)
            
            if success:
                print(f"✅ Task {i} completed successfully")
            else:
                print(f"❌ Task {i} failed after {task.retry_count} attempts")
        
        # Save after all execution/validation
        save_plan_to_sqlite(plan)
        print(f"\n💾 Saved plan after execution & validation")
        
        # === STAGE 3: FINAL STATUS ===
        print(f"\n📊 Stage 3: Final Status")
        
        validated_count = sum(1 for t in plan.tasks if t.status == "validated")
        failed_count = sum(1 for t in plan.tasks if t.status == "failed")
        
        if all(t.status == "validated" for t in plan.tasks):
            plan.status = "complete"
            print(f"   🎉 All tasks validated!")
        else:
            plan.status = "failed"
            print(f"   ⚠️  {failed_count}/{len(plan.tasks)} tasks failed")
        
        print(f"   Total tokens: {plan.token_usage}")
        
        # Final save with status
        save_plan_to_sqlite(plan)
        print(f"   💾 Saved final plan")
        
        print("\n" + "=" * 80)
        print(f"🎯 ORCHESTRATOR: Request {'Complete' if plan.status == 'complete' else 'Failed'}")
        print("=" * 80)
        
        return plan
    
    def _execute_task_with_retry(self, task: TaskSchema, plan: PlanSchema) -> bool:
        """
        Execute a task with retry logic.
        
        Args:
            task: The task to execute
            plan: The parent plan (for token tracking)
            
        Returns:
            bool: True if task succeeded, False if failed after all retries
        """
        attempt = 0
        
        while attempt <= self.max_retries:
            print(f"\n   Attempt {attempt + 1}/{self.max_retries + 1}")
            
            # Execute: Generate code
            updated_task, exec_tokens = CodingAgent.execute_task(task)
            plan.token_usage += exec_tokens
            
            # Update task reference in plan
            task.status = updated_task.status
            task.result = updated_task.result
            task.completed_at = updated_task.completed_at
            
            print(f"      Code generated ({exec_tokens} tokens)")
            
            # Validate: Test the code
            validated_task, val_tokens = ValidationAgent.validate_task(task)
            plan.token_usage += val_tokens
            
            # Update task reference in plan
            task.status = validated_task.status
            task.test_code = validated_task.test_code
            task.error_message = validated_task.error_message
            
            print(f"      Validation complete ({val_tokens} tokens)")
            
            # Check if successful
            if task.status == "validated":
                return True
            
            # Failed - check if we should retry
            task.retry_count = attempt + 1
            
            if attempt < self.max_retries:
                print(f"      ⚠️  Validation failed, preparing retry...")
                self._add_retry_feedback(task)
            else:
                print(f"      ❌ Max retries reached")
                return False
            
            attempt += 1
        
        return False
    
    def _add_retry_feedback(self, task: TaskSchema):
        """
        Add feedback to task goal to help CodingAgent fix issues on retry.
        
        Appends error information to the task goal so the next attempt
        has context about what went wrong.
        """
        if task.error_message:
            # Extract the key error info (first 200 chars)
            error_summary = task.error_message[:200]
            
            feedback = f"""

PREVIOUS ATTEMPT FAILED:
{error_summary}

Please fix the issue and regenerate the code.

Failed test code for reference:
{task.test_code[:300] if task.test_code else 'No test code available'}
"""
            
            # Append feedback to goal
            task.goal = task.goal + feedback
            
            print(f"      📝 Added error feedback to task goal")

class PlanningAgent:
    """Planning agent that converts user requests into structured plans."""
    
    model = "qwen2.5:7b-instruct"
    temperature = 0.3
    system_prompt = """You are a planning agent that breaks down user requests into actionable tasks.

Your job:
- Analyze the user's request
- Create 1-3 specific, ordered tasks
- Each task must be clear and measurable
- Tasks should build on each other logically

Return valid JSON matching the provided structure exactly."""
    
    @staticmethod
    def build_prompt(request: RequestSchema) -> str:
        """Build the planning prompt from a request."""
        return f"""User Request:
"{request.content}"

Analyze this request and create an appropriate plan.

PLANNING PRINCIPLES:
- If the request asks for "a function" or "a script", that's usually 1 task
- Don't break simple requests into unnecessary steps
- Each task should be independently executable code
- Avoid meta-tasks like "install library" or "set up environment"
- Focus on what code artifacts need to be created

Examples:
- "Write a function to X" → 1 task: "Write function that does X"
- "Create a script that does X and Y" → 2 tasks: "Write code for X", "Write code for Y"
- "Build a web scraper" → 2-3 tasks: "Write scraping function", "Write storage function", "Write main script"

Return JSON with this EXACT structure:
{{
    "request_id": "{request.request_id}",
    "tasks": [
        {{
            "plan_id": "placeholder",
            "task_order": 0,
            "goal": "Write a function/script that..."
        }}
    ]
}}

CRITICAL RULES:
1. task_order starts at 0 and increments by 1
2. Each goal must describe a CODE ARTIFACT to create
3. Include 1-3 tasks (prefer fewer for simple requests)
4. Use "placeholder" for plan_id
5. Return ONLY the JSON object

For this request, what code files/functions need to be written?"""
    
    @classmethod
    def create_plan(cls, request: RequestSchema) -> PlanSchema:
        """Execute the planning agent to create a plan from a request."""
        prompt = cls.build_prompt(request)
        
        print(f"\n🤖 Planning Agent processing request: {request.request_id}")
        print(f"📝 Content: {request.content[:100]}...")
        
        # Call LLM with structured output - now returns tokens too
        plan, tokens_used = call_ollama_structured(
            model=cls.model,
            prompt=prompt,
            system=cls.system_prompt,
            response_schema=PlanSchema,
            temperature=cls.temperature
        )
        
        # Set token usage on the plan
        plan.token_usage = tokens_used
        
        # Fix plan_id in each task to match the generated plan
        print(f"\n🔧 Fixing task plan_ids...")
        for task in plan.tasks:
            task.plan_id = plan.plan_id
        
        print(f"✅ Plan created: {plan.plan_id} with {len(plan.tasks)} tasks")
        print(f"🎫 Tokens consumed: {tokens_used}")
        
        return plan

class ValidationAgent:
    """Validation agent that checks if generated code is valid and functional."""
    
    model = "qwen2.5:7b-instruct"
    temperature = 0.2
    
    @staticmethod
    def validate_syntax(code: str) -> tuple[bool, str]:
        """
        Check if code is syntactically valid Python.
        Returns (is_valid, error_message).
        """
        try:
            compile(code, '<string>', 'exec')
            return True, ""
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"Compilation error: {str(e)}"
    
    @staticmethod
    def validate_imports(code: str) -> tuple[bool, str]:
        """
        Check if all imports in the code are available.
        Returns (is_valid, error_message).
        """
        try:
            # Execute the code to trigger import errors
            exec(code, {})
            return True, ""
        except ImportError as e:
            return False, f"ImportError: {str(e)}"
        except Exception as e:
            # Code executed but had runtime errors
            # For import validation, we only care about imports
            return True, ""
    
    @classmethod
    def generate_tests(cls, task: TaskSchema) -> tuple[str, int]:
        """
        Use LLM to generate test cases for the given code.
        Returns (test_code, tokens_used).
        """
        prompt = f"""Given this Python code:

    {task.result}

    Task goal: {task.goal}

    Generate test cases for this code. The tests should:
    1. Test normal cases with typical inputs
    2. Test edge cases (empty inputs, zero, boundary values)
    3. Test error cases (invalid inputs that should raise exceptions)

    CRITICAL: Return ONLY executable Python test code using assert statements.
    - Do NOT use unittest or pytest frameworks
    - Do NOT include example placeholders like "expected_value" 
    - Use actual, concrete values in your assertions
    - Each test should print a success message when it passes

    Requirements:
    - All variable names must be defined before use
    - All expected values must be concrete (not placeholders)
    - Code must be immediately runnable

    Format each test like this:
    # Test: description
    result = function_name(actual_input)
    assert result == actual_expected_output
    print("✓ Test passed: description")

    Return ONLY the test code that can be executed immediately."""

        system_prompt = """You are a test generation expert.
    Generate executable test code with concrete values only.
    Never use placeholder variables like 'expected_value' or 'some_input'.
    Every value must be real and runnable.
    Return ONLY Python code, no explanations."""

        # Call LLM and get actual token count
        test_code, tokens = call_ollama_code(
            model=cls.model,
            prompt=prompt,
            system=system_prompt,
            temperature=cls.temperature
        )
        
        # Strip markdown if present
        test_code = test_code.strip()
        if test_code.startswith("```python"):
            test_code = test_code[len("```python"):].strip()
        if test_code.startswith("```"):
            test_code = test_code[3:].strip()
        if test_code.endswith("```"):
            test_code = test_code[:-3].strip()
        
        return test_code, tokens
    
    @classmethod
    def run_tests(cls, original_code: str, test_code: str) -> tuple[bool, str]:
        """
        Execute the original code + test code in a temporary environment.
        Returns (tests_passed, output_or_error).
        """
        # Combine code and tests
        full_code = f"{original_code}\n\n# Generated Tests\n{test_code}"
        
        # Write to temporary file with UTF-8 encoding
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(full_code)
            temp_path = f.name
        
        try:
            # Set environment to force UTF-8
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            # Run the code
            result = subprocess.run(
                ['python', temp_path],
                capture_output=True,
                text=True,
                timeout=5,
                encoding='utf-8',
                env=env  # Use the UTF-8 environment
            )
            
            # Check if it succeeded
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            return False, "Test execution timed out (>5 seconds)"
        except Exception as e:
            return False, f"Test execution error: {str(e)}"
        finally:
            # Clean up temp file
            os.unlink(temp_path)
    
    @classmethod
    def validate_task(cls, task: TaskSchema) -> tuple[TaskSchema, int]:
        """
        Validate a task's generated code with functional tests.
        Returns (updated_task, tokens_used).
        """
        print(f"\n🔍 ValidationAgent processing task: {task.task_id}")
        print(f"📝 Goal: {task.goal[:80]}...")
        
        tokens_used = 0
        
        # Check if task has code to validate
        if not task.result:
            task.status = "failed"
            task.error_message = "No code generated to validate"
            print(f"❌ Validation failed: No code to validate")
            return task, tokens_used
        
        # Update status to validating
        task.status = "validating"
        
        # Step 1: Syntax validation
        print(f"   Checking syntax...")
        is_valid_syntax, syntax_error = cls.validate_syntax(task.result)
        
        if not is_valid_syntax:
            task.status = "failed"
            task.error_message = syntax_error
            print(f"❌ Validation failed: {syntax_error}")
            return task, tokens_used
        
        print(f"   ✓ Syntax valid")
        
        # Step 2: Import validation
        print(f"   Checking imports...")
        is_valid_imports, import_error = cls.validate_imports(task.result)
        
        if not is_valid_imports:
            task.status = "failed"
            task.error_message = import_error
            print(f"❌ Validation failed: {import_error}")
            return task, tokens_used
        
        print(f"   ✓ Imports valid")
        
        # Step 3: Generate tests (now returns actual tokens)
        print(f"   Generating tests...")
        test_code, test_tokens = cls.generate_tests(task)
        tokens_used += test_tokens
        
        # Store test code in task
        task.test_code = test_code
        
        print(f"   ✓ Tests generated ({test_tokens} tokens)")
        
        # Step 4: Run tests
        print(f"   Running tests...")
        tests_passed, output = cls.run_tests(task.result, test_code)
        
        if tests_passed:
            task.status = "validated"
            task.error_message = None
            print(f"✅ Validation passed")
            print(f"   Test output:\n{output}")
        else:
            task.status = "failed"
            task.error_message = f"Tests failed:\n{output}"
            print(f"❌ Validation failed: Tests did not pass")
            print(f"   Error output:\n{output}")
        
        return task, tokens_used
    
class CodingAgent:
    """Coding agent that generates Python code for tasks."""
    
    model = "qwen2.5-coder:7b-instruct"
    temperature = 0.1
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
    def build_prompt(task: TaskSchema) -> str:
        """Build the code generation prompt from a task."""
        return f"""Task: {task.goal}

Write complete Python code to accomplish this task.

Requirements:
- Include all necessary imports at the top
- Add error handling for common failures  
- Use clear variable names
- Add brief comments explaining the logic
- Return ONLY the code - no markdown, no examples, no explanations

Your response must start with valid Python code (either imports or a function definition).
Do NOT wrap in ```python``` or ``` blocks."""
    
    @classmethod
    def execute_task(cls, task: TaskSchema) -> tuple[TaskSchema, int]:
        """
        Execute the coding agent to generate code for a task.
        Returns (updated_task, tokens_used).
        """
        print(f"\n💻 CodingAgent processing task: {task.task_id}")
        print(f"📝 Goal: {task.goal[:100]}...")
        
        task.status = "executing"
        
        prompt = cls.build_prompt(task)
        
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
        
        # NEW: Strip example usage / comments at the end
        code = cls._strip_example_usage(code)
        
        # Store cleaned code
        task.result = code
        task.status = "complete"
        task.completed_at = datetime.utcnow()
        
        print(f"✅ Code generated ({len(code)} characters)")
        print(f"🎫 Tokens consumed: {tokens}")
        
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