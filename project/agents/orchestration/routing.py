from schemas import TaskSchema
from ..execution.coding import CodingAgent
from ..execution.research import ResearchAgent
from ..execution.file import FileAgent


class TaskRouter:
    """Routes tasks to appropriate agents based on task type."""

    @staticmethod
    def classify_task(task: TaskSchema) -> str:
        """
        Determine task type from goal.

        Returns: "code", "research", "analysis", "decision", "file", or "unknown"
        """
        goal_lower = task.goal.lower()

        # File operation tasks
        file_keywords = ["read file", "load file", "read csv", "list files", "write output", "save to file",
                         "save to outputs", "save the", "save results", "write to outputs", "write the results",
                         "write to file", "output to file"]
        if any(word in goal_lower for word in file_keywords):
            return "file"

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
    def route_task(cls, task: TaskSchema, context: dict = None) -> tuple[TaskSchema, int]:
        """Route task to the appropriate agent based on task type."""
        task_type = cls.classify_task(task)

        print(f"      🧭 Routing: Detected task type '{task_type}'")

        # Route to appropriate agent
        if task_type == "file":
            return FileAgent.execute_task(task, context)

        elif task_type == "code":
            return CodingAgent.execute_task(task, context)

        elif task_type == "research":
            return ResearchAgent.execute_task(task, context)

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
            return CodingAgent.execute_task(task, context)
