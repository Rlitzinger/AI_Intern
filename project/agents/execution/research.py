from schemas import TaskSchema
from llm import call_ollama_code
from datetime import datetime
from ddgs import DDGS


class ResearchAgent:
    """Research agent that searches the web and synthesizes findings."""

    model = "qwen2.5:7b-instruct"
    temperature = 0.3

    @classmethod
    def execute_task(cls, task: TaskSchema, context: dict = None) -> tuple[TaskSchema, int]:
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
