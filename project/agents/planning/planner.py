from schemas import RequestSchema, PlanSchema
from llm import call_ollama_structured


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

MULTI-AGENT WORKFLOWS - When to create 2+ tasks:

1. RESEARCH + FILE OPERATION
   Request: "Research X and save to file Y"
   → Task 0: "Research [topic] and identify/compare/compile..."
   → Task 1: "Save the research results to outputs/[filename]"

2. READ + ANALYSIS
   Request: "Read file X and analyze Y"
   → Task 0: "Read file [filename]"
   → Task 1: "Analyze the file data and identify..."

3. CODE + FILE OPERATION
   Request: "Write code for X and save to Y"
   → Task 0: "Write a function/script that..."
   → Task 1: "Save the code to outputs/[filename]"

KEYWORDS THAT TRIGGER DECOMPOSITION:
- "and save to"
- "and write to"
- "then save"
- "then write"
- "and output to"

When you see these patterns, ALWAYS create separate tasks for:
1. The main work (research/code/read/analyze)
2. The file operation (save/write)

Task 1 should reference "previous" or "research results" or "from task 0" so FileAgent knows to use context.

EXAMPLES:

MULTI-AGENT EXAMPLES:

Request: "Research protein content in chicken breast and save to outputs/chicken.txt"
→ Task 0: "Research protein content in chicken breast and compile findings"
→ Task 1: "Save the previous research results to outputs/chicken.txt"

Request: "Find Python web frameworks and create a comparison table in outputs/frameworks.csv"
→ Task 0: "Research Python web frameworks (Django, Flask, FastAPI) and compare their features"
→ Task 1: "Write the comparison results to outputs/frameworks.csv"

Request: "Read Workouts.csv and calculate average calories"
→ Task 0: "Read file Workouts.csv"
→ Task 1: "Analyze the workout data and calculate average calories burned"

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
3. Include 1-3 tasks based on workflow complexity:
   - Single agent = 1 task
   - Multi-agent workflow (research + save, read + analyze) = 2 tasks
   - Complex multi-step = 3 tasks
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
