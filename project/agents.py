from schemas import RequestSchema, PlanSchema, TaskSchema
from llm import call_ollama_structured
from llm import call_ollama_code
from datetime import datetime
import tempfile
import os

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
    """Validation agent that checks if generated code is valid Python."""
    
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
            # Other runtime errors are OK (the code might need inputs)
            return True, ""
    
    @classmethod
    def validate_task(cls, task: TaskSchema) -> TaskSchema:
        """
        Validate a task's generated code.
        Updates task status to 'validated' or 'failed'.
        Returns the updated task.
        """
        print(f"\n🔍 ValidationAgent processing task: {task.task_id}")
        print(f"📝 Goal: {task.goal[:80]}...")
        
        # Check if task has code to validate
        if not task.result:
            task.status = "failed"
            task.error_message = "No code generated to validate"
            print(f"❌ Validation failed: No code to validate")
            return task
        
        # Update status to validating
        task.status = "validating"
        
        # Step 1: Syntax validation
        print(f"   Checking syntax...")
        is_valid_syntax, syntax_error = cls.validate_syntax(task.result)
        
        if not is_valid_syntax:
            task.status = "failed"
            task.error_message = syntax_error
            print(f"❌ Validation failed: {syntax_error}")
            return task
        
        print(f"   ✓ Syntax valid")
        
        # Step 2: Import validation
        print(f"   Checking imports...")
        is_valid_imports, import_error = cls.validate_imports(task.result)
        
        if not is_valid_imports:
            task.status = "failed"
            task.error_message = import_error
            print(f"❌ Validation failed: {import_error}")
            return task
        
        print(f"   ✓ Imports valid")
        
        # All checks passed
        task.status = "validated"
        task.error_message = None
        print(f"✅ Validation passed")
        
        return task

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