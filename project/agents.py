from schemas import RequestSchema, PlanSchema, TaskSchema
from llm import call_ollama_structured
from llm import call_ollama_code
from datetime import datetime

class PlanningAgent:
    """Planning agent that converts user requests into structured plans."""
    
    model = "qwen2.5:7b-instruct"
    temperature = 0.3
    system_prompt = """You are a planning agent that breaks down user requests into actionable tasks.

Your job:
- Analyze the user's request
- Create 2-5 specific, ordered tasks
- Each task must be clear and measurable
- Tasks should build on each other logically

Return valid JSON matching the provided structure exactly."""
    
    @staticmethod
    def build_prompt(request: RequestSchema) -> str:
        """Build the planning prompt from a request."""
        return f"""User Request:
"{request.content}"

Create a plan to accomplish this request.

Return JSON with this EXACT structure:
{{
    "request_id": "{request.request_id}",
    "tasks": [
        {{
            "plan_id": "placeholder",
            "task_order": 0,
            "goal": "First specific task description"
        }},
        {{
            "plan_id": "placeholder",
            "task_order": 1,
            "goal": "Second specific task description"
        }}
    ]
}}

CRITICAL RULES:
1. task_order starts at 0 and increments by 1
2. Each goal must be specific and actionable
3. Include 2-5 tasks only
4. Use "placeholder" for plan_id (will be auto-generated)
5. Return ONLY the JSON object, no explanatory text

Focus on breaking down the request into concrete, implementable steps."""
    
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
        
        # Update status to executing
        task.status = "executing"
        
        prompt = cls.build_prompt(task)
        
        # Generate code using code-specific LLM call
        code, tokens = call_ollama_code(
            model=cls.model,
            prompt=prompt,
            system=cls.system_prompt,
            temperature=cls.temperature
        )
        
        # Store code in task
        task.result = code
        task.status = "complete"
        task.completed_at = datetime.utcnow()
        
        print(f"✅ Code generated ({len(code)} characters)")
        print(f"🎫 Tokens consumed: {tokens}")
        
        return task, tokens