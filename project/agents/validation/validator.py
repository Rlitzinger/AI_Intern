from schemas import TaskSchema
from llm import call_ollama_code
import tempfile
import subprocess
import os
import time
import requests
import socket
import psutil
from ..orchestration.routing import TaskRouter


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

        # Detect if this is non-deterministic code
        is_nondeterministic = any(keyword in code_lower for keyword in [
            'random', 'uuid', 'datetime.now', 'time.time', 'randint', 'choice', 'shuffle'
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
        elif is_nondeterministic:
            test_instructions = """
Generate test cases that verify the code works correctly.

⚠️ IMPORTANT: This function appears to use randomness/timestamps - DO NOT test for consistency.

Your tests should verify:
1. Function runs without errors
2. Returns the correct data type
3. Handles edge cases appropriately
4. Error handling works

DO NOT create tests that call the function multiple times and expect the same result.

Example good tests:
```python
# Test: Function runs successfully
result = generate_random_tall_tree_name()
assert result is not None
assert isinstance(result, str)
print("✓ Test passed: Returns string")

# Test: Multiple calls all succeed
for i in range(5):
    result = generate_random_tall_tree_name()
    assert isinstance(result, str)
    assert len(result) > 0
print("✓ Test passed: Multiple calls succeed")
```

Example BAD test (NEVER DO THIS for random functions):
```python
result1 = generate_random_tall_tree_name()
result2 = generate_random_tall_tree_name()
assert result1 == result2  # ❌ WRONG - random functions return different values!
```

Return ONLY executable Python test code.
"""
        else:
            # Regular functional tests
            test_instructions = """
    Generate test cases that verify the code works correctly.

    DO NOT hardcode expected numeric values - you don't know what values the function uses internally.

    Example good test:
    ```python
    # Test: Function runs successfully
    result = calculate_macros(100)
    assert result is not None
    assert isinstance(result, (dict, float, int))
    print(f"✓ Test passed: Function returns {type(result).__name__}")

    # Test: Consistent results
    result1 = calculate_macros(100)
    result2 = calculate_macros(100)
    assert result1 == result2
    print("✓ Test passed: Consistent results")

    # Test: Zero input
    result = calculate_macros(0)
    assert result is not None
    print("✓ Test passed: Handles zero input")
    ```

    Example bad test (NEVER DO THIS):
    ```python
    result = calculate_macros(100)
    assert result == 26.0  # Never hardcode expected values!
    ```

    Your tests should:
    1. Test that the function runs without errors (normal cases)
    2. Test that it returns the correct data type
    3. Test edge cases (zero, negative, boundary values)
    4. Test error handling (invalid types, out of range values)

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
    assert result is not None  # Check it ran, NOT a hardcoded value
    assert isinstance(result, expected_type)  # Check return type
    print("✓ Test passed: description")

    Return ONLY the test code that can be executed immediately."""

        system_prompt = """You are a test generation expert.
    Generate executable test code that does NOT hardcode expected output values.
    Test that functions run correctly and return the right type, not specific numbers.
    Use concrete inputs, but assert on type/structure/consistency - not hardcoded outputs.
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
