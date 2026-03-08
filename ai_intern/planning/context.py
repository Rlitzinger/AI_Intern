from dataclasses import dataclass, field
from ..config import settings
from typing import Optional


AGENT_ROSTER = {
    "code": {
        "name": "CodingAgent",
        "description": "Writes Python code. Use for: functions, scripts, calculations, data processing.",
        "keywords": ["write", "code", "function", "script", "implement", "calculate", "compute", "build"],
        "cannot_do": ["read files from disk", "search the web", "save files"]
    },
    "research": {
        "name": "ResearchAgent",
        "description": "Searches the web and synthesizes findings. Use for: facts, external data, comparisons.",
        "keywords": ["research", "find", "investigate", "search", "look up", "what is"],
        "cannot_do": ["write code", "read local files", "save files"]
    },
    "file": {
        "name": "FileAgent",
        "description": "Reads files from user_data/ and writes results to outputs/. Use for: reading CSVs, saving results.",
        "keywords": ["read file", "load file", "save", "write to", "output to", "store"],
        "cannot_do": ["write code", "search the web", "analyze data"]
    }
}


@dataclass
class PlanningContext:
    """
    Real-world context injected into every planning step.
    Built once per request, passed to classifier and decomposer.
    """
    available_files: list = field(default_factory=list)   # Paths relative to user_data/
    agent_roster: dict = field(default_factory=lambda: AGENT_ROSTER)

    @classmethod
    def build(cls) -> "PlanningContext":
        """
        Scan the filesystem and build planning context.
        Call this once at the start of create_plan().
        """
        context = cls()
        context.available_files = cls._scan_user_data()
        return context

    @staticmethod
    def _scan_user_data() -> list:
        """Return list of files in user_data/ relative to that directory."""
        user_data = settings.USER_DATA_DIR
        if not user_data.exists():
            return []
        files = []
        for f in sorted(user_data.rglob("*")):
            if f.is_file():
                files.append(str(f.relative_to(user_data)))
        return files

    def format_for_prompt(self) -> str:
        """
        Format context as a string block for injection into prompts.
        Keep it tight - every token here costs reasoning budget.
        """
        lines = []

        lines.append("AVAILABLE FILES (in user_data/):")
        if self.available_files:
            for f in self.available_files:
                lines.append(f"  - {f}")
        else:
            lines.append("  (none)")

        lines.append("")
        lines.append("AVAILABLE AGENTS:")
        for agent_type, info in self.agent_roster.items():
            lines.append(f"  [{agent_type}] {info['name']}: {info['description']}")

        return "\n".join(lines)


def format_context_for_prompt(previous_tasks: list[dict]) -> str:
    """Format previous task outputs for agent consumption.

    When key_values are present, renders them as named variables.
    Otherwise falls back to data_summary or raw result.
    """
    if not previous_tasks:
        return ""

    lines = ["INPUT FROM PREVIOUS TASKS:"]
    for t in previous_tasks:
        lines.append(f"\n--- Task {t['order']} ({t['goal'][:60]}) ---")

        if t.get('key_values'):
            for k, v in t['key_values'].items():
                if isinstance(v, list) and len(v) > 5:
                    lines.append(f"  {k} = [{len(v)} items, first 3: {v[:3]}]")
                elif isinstance(v, str) and len(v) > 200:
                    lines.append(f"  {k} = {v[:200]}...")
                else:
                    lines.append(f"  {k} = {v}")
        elif t.get('data_summary'):
            lines.append(f"  Summary: {t['data_summary'][:300]}")
        elif t.get('result'):
            lines.append(f"  Result: {t['result'][:300]}")
        elif t.get('file_path'):
            lines.append(f"  File: {t['file_path']}")

    return "\n".join(lines)
