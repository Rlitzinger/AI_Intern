from ..schemas import TaskSchema, TaskOutput
from ..llm import call_ollama_code
from ..config import settings
from ..logging_config import get_logger
from datetime import datetime, timezone
from ddgs import DDGS
import json

logger = get_logger("research")


class ResearchAgent:
    """Research agent that searches the web and synthesizes findings."""

    model = settings.RESEARCH_MODEL
    temperature = settings.RESEARCH_TEMP

    @classmethod
    def execute_task(cls, task: TaskSchema, context: dict = None) -> tuple[TaskSchema, int]:
        """
        Execute research task using web search.

        Args:
            task: The research task to execute

        Returns:
            tuple: (updated_task, tokens_used)
        """
        logger.info(f"ResearchAgent processing task: {task.task_id}")
        logger.info(f"Goal: {task.goal[:80]}...")

        task.status = "executing"
        tokens_used = 0

        # Step 1: Extract search query from task goal
        search_query = cls._extract_search_query(task.goal)
        logger.debug(f"Search query: '{search_query}'")

        # Step 2: Perform web search
        logger.info("Searching the web...")
        search_results = cls._web_search(search_query, max_results=5)

        if not search_results:
            task.status = "failed"
            task.error_message = "No search results found"
            logger.error("No results found")
            return task, tokens_used

        logger.info(f"Found {len(search_results)} results")

        # Step 3: Use LLM to synthesize findings
        logger.info("Synthesizing findings...")
        synthesis, synth_tokens = cls._synthesize_findings(task.goal, search_results)
        tokens_used += synth_tokens

        # Step 4: Extract structured key-value data from synthesis
        key_values, kv_tokens = cls._extract_key_values(task.goal, synthesis)
        tokens_used += kv_tokens

        # Step 5: Store results
        task.result = synthesis
        task.status = "complete"
        task.completed_at = datetime.now(timezone.utc)

        # Build structured TaskOutput
        task.task_output = TaskOutput(
            output_type="data",
            raw_result=synthesis,
            data_summary=synthesis[:500],
        )
        if key_values:
            task.task_output.key_values = key_values
            logger.info(f"Extracted {len(key_values)} key-value pairs from research")

        logger.info(f"Research complete ({tokens_used} tokens)")
        logger.debug(f"Summary length: {len(synthesis)} characters")

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
            logger.warning(f"Search error: {e}")
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

    @classmethod
    def _extract_key_values(cls, goal: str, synthesis: str) -> tuple:
        """
        Extract numeric key-value pairs from research synthesis.

        Returns (dict, tokens_used). Empty dict on failure.
        """
        prompt = f"""Research goal: {goal}

Research findings:
{synthesis[:1500]}

Extract ALL numeric facts and measurements from the research findings above.
Return them as a flat JSON object with descriptive snake_case keys and numeric values.

Example output:
{{"protein_per_100g": 26, "fat_per_100g": 10, "calories_per_100g": 239}}

If there are no numeric facts, return an empty JSON object: {{}}

Return ONLY the JSON object, nothing else."""

        system_prompt = (
            "You are a data extraction assistant. "
            "Extract numeric facts as JSON. Return ONLY valid JSON."
        )

        try:
            response, tokens = call_ollama_code(
                model=cls.model,
                prompt=prompt,
                system=system_prompt,
                temperature=0.1,
            )

            # Try to parse JSON from response
            response = response.strip()
            # Strip markdown if present
            if response.startswith("```json"):
                response = response[len("```json"):].strip()
            if response.startswith("```"):
                response = response[3:].strip()
            if response.endswith("```"):
                response = response[:-3].strip()

            data = json.loads(response)
            if isinstance(data, dict):
                return data, tokens
            return {}, tokens

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Key-value extraction failed: {e}")
            return {}, 0
