from schemas import RequestSchema, PlanSchema, TaskSchema
from llm import call_ollama_structured
from llm import call_ollama_code
from datetime import datetime
import tempfile
import subprocess
import os
from storage import save_plan_to_sqlite, load_plan_from_sqlite
from ddgs import DDGS
import time
import requests
import socket
import signal
import psutil


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
            
            # Execute: Route to appropriate agent
            updated_task, exec_tokens = TaskRouter.route_task(task)  # <-- CHANGED
            plan.token_usage += exec_tokens
            
            # Update task reference in plan
            task.status = updated_task.status
            task.result = updated_task.result
            task.completed_at = updated_task.completed_at
            
            # Check if task was routed to unsupported agent
            if task.status == "failed" and "not yet implemented" in (task.error_message or ""):
                print(f"      ❌ Task type not supported, cannot retry")
                return False
            
            print(f"      Code generated ({exec_tokens} tokens)")
            
            # Validate: Test the code (only for code tasks)
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

class TaskRouter:
    """Routes tasks to appropriate agents based on task type."""
    
    @staticmethod
    def classify_task(task: TaskSchema) -> str:
        """
        Determine task type from goal.
        
        Returns: "code", "research", "analysis", "decision", or "unknown"
        """
        goal_lower = task.goal.lower()
        
        # Code generation tasks
        code_keywords = ["write", "code", "function", "script", "implement", "create", "build", "develop"]
        if any(word in goal_lower for word in code_keywords):
            return "code"
        
        # Research tasks
        research_keywords = ["research", "find", "investigate", "search", "explore", "identify"]
        if any(word in goal_lower for word in research_keywords):
            return "research"
        
        # Analysis tasks
        analysis_keywords = ["analyze", "evaluate", "compare", "assess", "review", "examine"]
        if any(word in goal_lower for word in analysis_keywords):
            return "analysis"
        
        # Decision tasks
        decision_keywords = ["decide", "choose", "select", "recommend", "suggest"]
        if any(word in goal_lower for word in decision_keywords):
            return "decision"
        
        # Unknown task type
        return "unknown"
    
    @classmethod
    def route_task(cls, task: TaskSchema) -> tuple[TaskSchema, int]:
        """
        Route task to the appropriate agent based on task type.
        
        Args:
            task: The task to execute
            
        Returns:
            tuple: (updated_task, tokens_used)
        """
        task_type = cls.classify_task(task)
        
        print(f"      🧭 Routing: Detected task type '{task_type}'")
        
        # Route to appropriate agent
        if task_type == "code":
            return CodingAgent.execute_task(task)
        
        elif task_type == "research":
            return ResearchAgent.execute_task(task)
        
        elif task_type == "analysis":
            # TODO: Implement AnalysisAgent
            task.status = "failed"
            task.error_message = "AnalysisAgent not yet implemented. Task type 'analysis' is not supported."
            print(f"      ❌ AnalysisAgent not available")
            return task, 0
        
        elif task_type == "decision":
            # TODO: Implement DecisionAgent
            task.status = "failed"
            task.error_message = "DecisionAgent not yet implemented. Task type 'decision' is not supported."
            print(f"      ❌ DecisionAgent not available")
            return task, 0
        
        else:
            # Unknown task type - try CodingAgent as fallback
            print(f"      ⚠️  Unknown task type, defaulting to CodingAgent")
            return CodingAgent.execute_task(task)

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
- Break the request into 1-3 specific, actionable tasks
- Each task should be independently executable
- Tasks should build on each other logically
- Don't break simple requests into unnecessary steps

TASK TYPES - Choose the right type for each task:

1. CODE TASKS - Use when the goal is to generate code
   - Keywords: "write", "code", "function", "script", "implement", "build"
   - Format: "Write a function that..."
   - Example: "Write a function to parse CSV files"

2. RESEARCH TASKS - Use when the goal is to gather information
   - Keywords: "research", "find", "investigate", "compare", "explore"
   - Format: "Research [topic] and identify..."
   - Example: "Research Python web frameworks and compare their features"

3. ANALYSIS TASKS - Use when the goal is to analyze data or information
   - Keywords: "analyze", "evaluate", "assess", "examine"
   - Format: "Analyze [data/info] and..."
   - Example: "Analyze the performance metrics and identify bottlenecks"

4. DECISION TASKS - Use when the goal is to make a recommendation
   - Keywords: "decide", "choose", "recommend", "suggest"
   - Format: "Recommend [option] based on..."
   - Example: "Recommend the best framework based on project requirements"

EXAMPLES:

Request: "Write a function to calculate Fibonacci"
→ 1 task: "Write a function that calculates the Fibonacci sequence up to n terms"

Request: "Research Python web frameworks and recommend one"
→ Task 0: "Research Python web frameworks (Django, Flask, FastAPI) and compare features"
→ Task 1: "Recommend the best framework based on ease of use and performance"

Request: "Build a web scraper for product prices"
→ Task 0: "Write a function to scrape product data from target website"
→ Task 1: "Write a function to store scraped data in a database"

Return JSON with this EXACT structure:
{{
    "request_id": "{request.request_id}",
    "tasks": [
        {{
            "plan_id": "placeholder",
            "task_order": 0,
            "goal": "Clear description of what needs to be done"
        }}
    ]
}}

CRITICAL RULES:
1. task_order starts at 0 and increments by 1
2. Each goal must clearly describe what needs to be accomplished
3. Include 1-3 tasks (prefer fewer for simple requests)
4. Use "placeholder" for plan_id
5. Return ONLY the JSON object

For this request, what tasks are needed?"""
    
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
        # Detect if this is server/daemon code
        code_lower = task.result.lower()
        is_server_code = any(keyword in code_lower for keyword in [
            'flask', 'fastapi', 'django', '.run(', 'uvicorn', 'serve', 'listen'
        ])
        
        if is_server_code:
            # Generate structural tests only (no execution)
            test_instructions = """
    IMPORTANT: This code defines a web server. Do NOT call .run() or start the server in tests.

    Instead, test the structure:
    1. Check that functions/routes are defined
    2. Check that the app object exists
    3. Verify function signatures
    4. Test route handlers directly (without starting server)

    Example for Flask:
    ```python
    # Test: Check app is defined
    assert hasattr(create_hello_world_server, '__call__')
    print("✓ Test passed: Function is callable")

    # Test: Check Flask app can be created
    app = Flask(__name__)
    assert app is not None
    print("✓ Test passed: Flask app created")
    ```

    Do NOT include:
    - app.run() calls
    - server.serve_forever() calls
    - Any code that would block execution
    """
        else:
            # Regular functional tests
            test_instructions = """
    Generate comprehensive test cases for this code. Your tests should:
    1. Test normal cases (typical inputs)
    2. Test edge cases (empty inputs, boundary values)
    3. Test error cases (invalid inputs that should raise exceptions)

    Return ONLY Python test code using assert statements.
    """
        
        prompt = f"""Given this Python code:

    {task.result}

    Task goal: {task.goal}

    {test_instructions}

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
    For web servers, test structure only - never start the server.
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
        Validate a task's result.
        For code tasks: run functional tests
        For other tasks: basic validation only
        """
        print(f"\n🔍 ValidationAgent processing task: {task.task_id}")
        print(f"📝 Goal: {task.goal[:80]}...")
        
        tokens_used = 0
        
        # Check if task has result
        if not task.result:
            task.status = "failed"
            task.error_message = "No result generated to validate"
            print(f"❌ Validation failed: No result to validate")
            return task, tokens_used
        
        # Update status to validating
        task.status = "validating"
        
        # Determine task type for validation
        task_type = TaskRouter.classify_task(task)
        
        if task_type == "code":
            # Full validation with tests
            return cls._validate_code_task(task)
        else:
            # Simple validation for non-code tasks
            return cls._validate_non_code_task(task)
        
    @classmethod
    def _validate_code_task(cls, task: TaskSchema) -> tuple[TaskSchema, int]:
        """Validate code task with full testing."""
        tokens_used = 0
        
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
        
        # Step 3: Detect server code
        code_lower = task.result.lower()
        is_server_code = any(keyword in code_lower for keyword in [
            'flask', 'fastapi', 'django', 'uvicorn',
            'httpserver', 'http.server', 'basehttprequesthandler',
            'serve_forever', '.run(', 'app.run'
        ])
        
        if is_server_code:
            # Server code - test with live request
            server_works, server_output = cls._test_server_code(task)
            
            if server_works:
                task.status = "validated"
                task.error_message = None
                task.test_code = f"# Server tested with live HTTP request\n# {server_output}"
                print(f"✅ Validation passed")
                return task, tokens_used
            else:
                task.status = "failed"
                task.error_message = f"Server test failed: {server_output}"
                print(f"❌ Validation failed")
                return task, tokens_used
        
        # Step 4: Regular code - generate and run tests
        print(f"   Generating tests...")
        test_code, test_tokens = cls.generate_tests(task)
        tokens_used += test_tokens
        
        task.test_code = test_code
        print(f"   ✓ Tests generated ({test_tokens} tokens)")
        
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

    @classmethod  
    def _validate_non_code_task(cls, task: TaskSchema) -> tuple[TaskSchema, int]:
        """Validate non-code task (research, analysis, etc)."""
        print(f"   Checking result...")
        
        # Basic checks
        if len(task.result) < 50:
            task.status = "failed"
            task.error_message = "Result too short (< 50 characters)"
            print(f"❌ Validation failed: Result too short")
            return task, 0
        
        # For research tasks, check if it looks like actual content
        if task.result.lower().startswith("error") or "failed" in task.result.lower()[:100]:
            task.status = "failed"
            task.error_message = "Result appears to be an error message"
            print(f"❌ Validation failed: Result is an error")
            return task, 0
        
        # Passed basic validation
        task.status = "validated"
        task.error_message = None
        print(f"✅ Validation passed (basic checks)")
        
        return task, 0

    @staticmethod
    def _find_free_port() -> int:
        """Find an available port for testing."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    @staticmethod
    def _kill_process_tree(pid: int):
        """
        Kill a process and all its children.
        Handles the case where Flask spawns child processes.
        """
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            
            # Kill children first
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            
            # Kill parent
            try:
                parent.terminate()
            except psutil.NoSuchProcess:
                pass
            
            # Wait for termination (up to 2 seconds)
            gone, alive = psutil.wait_procs(children + [parent], timeout=2)
            
            # Force kill if still alive
            for p in alive:
                try:
                    p.kill()
                except psutil.NoSuchProcess:
                    pass
                    
        except psutil.NoSuchProcess:
            # Process already dead
            pass
        except Exception as e:
            print(f"      ⚠️  Error during cleanup: {e}")
    
    @classmethod
    def _extract_expected_response(cls, task_goal: str, code: str) -> str:
        """
        Extract what response we expect from the server.
        Looks in task goal and code for clues.
        """
        goal_lower = task_goal.lower()
        code_lower = code.lower()
        
        # Check for explicit strings in task goal
        if "hello world" in goal_lower or "hello, world" in goal_lower:
            return "hello world"
        elif "hello" in goal_lower:
            return "hello"
        
        # Check code for return statements
        lines = code.split('\n')
        for line in lines:
            line_stripped = line.strip().lower()
            if "return" in line_stripped and ("'" in line or '"' in line):
                # Extract string from return statement
                # e.g., "return 'Hello World!'" -> "hello world!"
                start = line.find("'")
                if start == -1:
                    start = line.find('"')
                if start != -1:
                    end = line.find("'", start + 1)
                    if end == -1:
                        end = line.find('"', start + 1)
                    if end != -1:
                        return line[start + 1:end].lower().strip()
        
        # Default: just check for 200 status
        return None
    
    @classmethod
    def _detect_server_framework(cls, code: str) -> str:
        """Detect which framework is being used."""
        code_lower = code.lower()
        
        if 'flask' in code_lower:
            return 'flask'
        elif 'fastapi' in code_lower:
            return 'fastapi'
        elif 'django' in code_lower:
            return 'django'
        elif 'httpserver' in code_lower or 'http.server' in code_lower:
            return 'http.server'
        else:
            return 'unknown'
    
    @classmethod
    def _build_server_startup_code(cls, original_code: str, port: int, framework: str) -> str:
        """
        Build complete server code with startup logic.
        Handles different frameworks and edge cases.
        """
        # Ensure original code doesn't end with indentation issues
        original_code = original_code.rstrip()
        
        if framework == 'http.server':
            return f"""{original_code}

            
# === TESTING HARNESS ===
if __name__ == '__main__':
    import sys
    from http.server import HTTPServer
    
    # Get all objects in the module (snapshot to avoid dict mutation)
    namespace = dict(globals())
    
    # Find any run function
    run_func = None
    for name, obj in namespace.items():
        if 'run' in name.lower() and callable(obj) and not name.startswith('_'):
            run_func = obj
            break
    
    if run_func:
        try:
            import inspect
            sig = inspect.signature(run_func)
            
            if 'port' in sig.parameters:
                run_func(port={port})
            else:
                run_func()
        except Exception as e:
            print(f"ERROR: Server failed via {{run_func.__name__}}(): {{e}}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        handler = None
        for name, obj in namespace.items():
            if (isinstance(obj, type) and 
                hasattr(obj, 'do_GET') and 
                'Handler' in name):
                handler = obj
                break
        
        if handler:
            try:
                server_address = ('127.0.0.1', {port})
                httpd = HTTPServer(server_address, handler)
                httpd.serve_forever()
            except Exception as e:
                print(f"ERROR: Server failed: {{e}}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                sys.exit(1)
        else:
            print("ERROR: Cannot start http.server - no run function or handler found", file=sys.stderr)
            sys.exit(1)
"""
        
        elif framework == 'flask':
            return f"""{original_code}

            
# === TESTING HARNESS ===
if __name__ == '__main__':
    import sys
    from flask import Flask
    
    # [rest of Flask harness...]
"""
        
        else:
            return f"""{original_code}

            
# === TESTING HARNESS ===
if __name__ == '__main__':
    import sys
    print("ERROR: Unsupported framework for testing")
    sys.exit(1)
"""
    
    @classmethod
    def _test_server_code(cls, task: TaskSchema) -> tuple[bool, str]:
        """Test server code by starting it, making a request, and shutting down."""
        print(f"   🌐 Testing server with live HTTP request...")
        
        port = cls._find_free_port()
        print(f"      Using port: {port}")
        
        framework = cls._detect_server_framework(task.result)
        print(f"      Detected framework: {framework}")
        
        if framework == 'unknown':
            return False, "Cannot test: Unknown server framework"
        
        expected_response = cls._extract_expected_response(task.goal, task.result)
        print(f"      Expected response: {expected_response or 'Any 200 OK'}")
        
        # Build complete server code
        server_code = cls._build_server_startup_code(task.result, port, framework)
        
        # DEBUG: Log the generated server code
        print(f"      DEBUG: Generated harness code (first 500 chars):")
        print(f"      {server_code[:500]}")
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(server_code)
            temp_path = f.name
        
        print(f"      Temp file: {temp_path}")
        
        server_process = None
        
        try:
            # Start server in subprocess
            print(f"      Starting server...")
            server_process = subprocess.Popen(
                ['python', temp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            
            # Wait for server to start (with timeout)
            max_wait = 5
            wait_interval = 0.5
            waited = 0
            server_ready = False
            
            while waited < max_wait:
                # Check if process died
                if server_process.poll() is not None:
                    stdout, stderr = server_process.communicate()
                    print(f"      DEBUG: Process died")
                    print(f"      DEBUG: STDOUT: {stdout[:200]}")
                    print(f"      DEBUG: STDERR: {stderr[:200]}")
                    return False, f"Server failed to start:\nSTDOUT: {stdout}\nSTDERR: {stderr}"
                
                # Try to connect
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    result = sock.connect_ex(('127.0.0.1', port))
                    sock.close()
                    
                    if result == 0:
                        server_ready = True
                        break
                except:
                    pass
                
                time.sleep(wait_interval)
                waited += wait_interval
            
            if not server_ready:
                # Get output before killing
                server_process.terminate()
                stdout, stderr = server_process.communicate(timeout=2)
                print(f"      DEBUG: Timeout - no connection")
                print(f"      DEBUG: STDOUT: {stdout[:200]}")
                print(f"      DEBUG: STDERR: {stderr[:200]}")
                return False, f"Server did not start within {max_wait} seconds\nSTDOUT: {stdout}\nSTDERR: {stderr}"
            
            print(f"      ✓ Server started (took {waited:.1f}s)")
            
            # Make test request
            print(f"      Making test request to http://127.0.0.1:{port}/")
            
            try:
                response = requests.get(f'http://127.0.0.1:{port}/', timeout=3)
                
                print(f"      Status: {response.status_code}")
                print(f"      Response: {response.text[:100]}")
                
                # Validate response
                if response.status_code != 200:
                    return False, f"Server returned status {response.status_code} (expected 200)"
                
                # Check content if we have expected response
                if expected_response:
                    response_lower = response.text.lower().strip()
                    expected_lower = expected_response.lower().strip()
                    
                    if expected_lower not in response_lower:
                        return False, f"Response '{response.text}' does not contain expected '{expected_response}'"
                
                return True, f"Server test passed! Status: {response.status_code}, Response: {response.text[:100]}"
                
            except requests.exceptions.Timeout:
                return False, "Request timed out (server not responding)"
            except requests.exceptions.ConnectionError as e:
                return False, f"Connection error: {e}"
            except requests.exceptions.RequestException as e:
                return False, f"Request failed: {e}"
        
        except Exception as e:
            return False, f"Unexpected error during server test: {e}"
        
        finally:
            # === CRITICAL: CLEANUP ===
            print(f"      Cleaning up...")
            
            # Kill the server process tree
            if server_process:
                try:
                    cls._kill_process_tree(server_process.pid)
                    print(f"      ✓ Server process terminated")
                except Exception as e:
                    print(f"      ⚠️  Cleanup error: {e}")
            
            # Extra safety: kill anything on the test port
            try:
                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        # Get connections with proper kind parameter
                        connections = proc.connections(kind='inet')
                        for conn in connections:
                            if hasattr(conn, 'laddr') and conn.laddr.port == port:
                                print(f"      ⚠️  Killing leftover process on port {port}: {proc.pid}")
                                cls._kill_process_tree(proc.pid)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except Exception as e:
                print(f"      ⚠️  Port cleanup error: {e}")
            
            # Delete temp file
            try:
                os.unlink(temp_path)
                print(f"      ✓ Temp file deleted")
            except Exception as e:
                print(f"      ⚠️  File cleanup error: {e}")
            
            # Final safety wait
            time.sleep(0.5)

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
    
class ResearchAgent:
    """Research agent that searches the web and synthesizes findings."""
    
    model = "qwen2.5:7b-instruct"
    temperature = 0.3
    
    @classmethod
    def execute_task(cls, task: TaskSchema) -> tuple[TaskSchema, int]:
        """
        Execute research task using web search.
        
        Args:
            task: The research task to execute
            
        Returns:
            tuple: (updated_task, tokens_used)
        """
        print(f"\n🔍 ResearchAgent processing task: {task.task_id}")
        print(f"📝 Goal: {task.goal[:80]}...")
        
        task.status = "executing"
        tokens_used = 0
        
        # Step 1: Extract search query from task goal
        search_query = cls._extract_search_query(task.goal)
        print(f"   🔎 Search query: '{search_query}'")
        
        # Step 2: Perform web search
        print(f"   🌐 Searching the web...")
        search_results = cls._web_search(search_query, max_results=5)
        
        if not search_results:
            task.status = "failed"
            task.error_message = "No search results found"
            print(f"   ❌ No results found")
            return task, tokens_used
        
        print(f"   ✓ Found {len(search_results)} results")
        
        # Step 3: Use LLM to synthesize findings
        print(f"   🤖 Synthesizing findings...")
        synthesis, synth_tokens = cls._synthesize_findings(task.goal, search_results)
        tokens_used += synth_tokens
        
        # Step 4: Store results
        task.result = synthesis
        task.status = "complete"
        task.completed_at = datetime.utcnow()
        
        print(f"✅ Research complete ({tokens_used} tokens)")
        print(f"   Summary length: {len(synthesis)} characters")
        
        return task, tokens_used
    
    @staticmethod
    def _extract_search_query(goal: str) -> str:
        """
        Extract searchable query from task goal.
        
        Removes common research task words, keeps the core topic.
        """
        query = goal.lower()
        
        # Remove common research task words
        remove_phrases = [
            "research", "investigate", "find information about", "find", 
            "explore", "compare", "and compare their", "and compare", 
            "their features", "and identify", "look up"
        ]
        
        for phrase in remove_phrases:
            query = query.replace(phrase, "")
        
        # Clean up whitespace
        query = " ".join(query.split())
        query = query.strip()
        
        return query
    
    @staticmethod
    def _web_search(query: str, max_results: int = 5) -> list[dict]:
        """
        Perform web search using DuckDuckGo.
        
        Returns list of dicts with 'title', 'href', 'body' keys.
        """
        try:
            ddgs = DDGS()
            results = list(ddgs.text(query, max_results=max_results))
            return results
        except Exception as e:
            print(f"   ⚠️  Search error: {e}")
            return []
    
    @classmethod
    def _synthesize_findings(cls, goal: str, search_results: list[dict]) -> tuple[str, int]:
        """
        Use LLM to synthesize search results into coherent summary.
        
        Returns (synthesis, tokens_used).
        """
        # Build context from search results
        context = ""
        for i, result in enumerate(search_results, 1):
            context += f"\n--- Source {i}: {result.get('title', 'Untitled')} ---\n"
            context += f"{result.get('body', 'No description')}\n"
            context += f"URL: {result.get('href', 'No URL')}\n"
        
        prompt = f"""Research Task: {goal}

I searched the web and found these results:

{context}

Based on these search results, provide a comprehensive answer to the research task.

Requirements:
- Synthesize information from multiple sources
- Be factual and objective
- Cite sources when making specific claims (e.g., "According to Source 1...")
- Organize findings clearly with key points
- Focus on answering the specific research goal

Provide your research synthesis:"""

        system_prompt = """You are a research assistant that synthesizes web search results.
Provide clear, factual summaries based on the sources provided.
Cite sources when making specific claims.
Keep responses focused and relevant to the task.
Do not add information not found in the sources."""

        synthesis, tokens = call_ollama_code(
            model=cls.model,
            prompt=prompt,
            system=system_prompt,
            temperature=cls.temperature
        )
        
        return synthesis.strip(), tokens
    
