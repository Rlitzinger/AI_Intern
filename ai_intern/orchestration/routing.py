from ..schemas import TaskSchema
from ..execution.coding import CodingAgent
from ..execution.research import ResearchAgent
from ..execution.file import FileAgent
from ..execution.analysis import AnalysisAgent
from ..logging_config import get_logger

logger = get_logger("routing")


class TaskRouter:
    """Routes tasks to appropriate agents based on task type using a scoring system."""

    @staticmethod
    def classify_task(task: TaskSchema) -> str:
        """
        Determine task type from goal using weighted scoring.

        Returns: "code", "research", "analysis", "file", or "unknown"
        """
        goal_lower = task.goal.lower()
        scores = {"file": 0, "code": 0, "research": 0, "analysis": 0}

        # === FILE signals ===
        # Strong: explicit file extensions
        if any(ext in goal_lower for ext in [".csv", ".json", ".txt", ".md", ".log"]):
            scores["file"] += 3

        # Strong: multi-word file phrases
        file_phrases = [
            "read file", "read csv", "read json", "load file", "load csv",
            "load data", "open file", "parse file", "list files",
            "save to file", "save to outputs", "write to outputs",
            "write to file", "output to file", "save as", "write as",
            "export to", "save results", "save the", "write the results",
            "summary file", "create a file", "output file",
        ]
        if any(phrase in goal_lower for phrase in file_phrases):
            scores["file"] += 5

        # Medium: single file keywords (only if goal is short/focused)
        file_keywords = ["read", "load", "save", "export"]
        for kw in file_keywords:
            if kw in goal_lower:
                scores["file"] += 1

        # ".py" extension with save/write/create = file operation
        if ".py" in goal_lower and any(kw in goal_lower for kw in ["save", "write", "create", "export"]):
            scores["file"] += 3  # save-as-py is a file operation

        # === RESEARCH signals ===
        research_phrases = [
            "research the", "research about", "find information",
            "investigate", "search for", "look up", "explore",
        ]
        if any(phrase in goal_lower for phrase in research_phrases):
            scores["research"] += 5

        research_keywords = ["research", "investigate", "search"]
        for kw in research_keywords:
            if kw in goal_lower:
                scores["research"] += 2

        # === ANALYSIS signals ===
        analysis_phrases = [
            "calculate average", "calculate total", "calculate sum",
            "find the average", "find the max", "find the min",
            "summarize the data", "analyze the data", "analyze my",
            "compute the", "aggregate", "statistics",
        ]
        if any(phrase in goal_lower for phrase in analysis_phrases):
            scores["analysis"] += 5

        analysis_keywords = [
            "analyze", "evaluate", "compare", "summarize",
            "calculate", "average", "total", "statistics",
        ]
        for kw in analysis_keywords:
            if kw in goal_lower:
                scores["analysis"] += 1

        # === CODE signals ===
        code_phrases = [
            "write a function", "write a script", "write a class",
            "write python", "implement a", "build a", "create a function",
            "create a script", "develop a", "code a",
        ]
        if any(phrase in goal_lower for phrase in code_phrases):
            scores["code"] += 5

        code_keywords = [
            "function", "script", "implement", "code",
            "class", "module", "program",
        ]
        for kw in code_keywords:
            if kw in goal_lower:
                scores["code"] += 2

        # Weak code signals (only +1 to avoid overriding file/analysis)
        weak_code = ["write", "create", "build", "develop"]
        for kw in weak_code:
            if kw in goal_lower:
                scores["code"] += 1

        # Get winner
        max_score = max(scores.values())
        if max_score == 0:
            return "unknown"

        winner = max(scores, key=scores.get)

        # Tie-breaking: if file and code are tied, prefer file for save operations
        if scores["file"] == scores["code"] and scores["file"] > 0:
            if any(kw in goal_lower for kw in ["save", "output", "export", "write to"]):
                winner = "file"

        logger.debug(f"Task scores: {scores} → {winner}")
        return winner

    @classmethod
    def route_task(cls, task: TaskSchema, context: dict = None) -> tuple[TaskSchema, int]:
        """Route task to the appropriate agent based on task type."""
        task_type = cls.classify_task(task)

        logger.info(f"Routing: Detected task type '{task_type}'")

        if task_type == "file":
            return FileAgent.execute_task(task, context)
        elif task_type == "code":
            return CodingAgent.execute_task(task, context)
        elif task_type == "research":
            return ResearchAgent.execute_task(task, context)
        elif task_type == "analysis":
            return AnalysisAgent.execute_task(task, context)
        else:
            logger.warning("Unknown task type, defaulting to CodingAgent")
            return CodingAgent.execute_task(task, context)
