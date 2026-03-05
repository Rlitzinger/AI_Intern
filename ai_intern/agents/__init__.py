"""
Compatibility shim: re-exports from the ai_intern package.
Allows test commands run from ai_intern/ to use `from agents import Orchestrator`.
"""
import sys
import os

# Ensure AI_Intern/ (project root) is on sys.path so ai_intern package is importable
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ai_intern.orchestration.orchestrator import Orchestrator
from ai_intern.orchestration.routing import TaskRouter
from ai_intern.planning.planner import PlanningAgent
from ai_intern.planning.classifier import TaskClassifier, TaskVerdict
from ai_intern.planning.hierarchical import HierarchicalPlanner
from ai_intern.planning.critic import CritiqueAgent

__all__ = [
    'Orchestrator',
    'TaskRouter',
    'PlanningAgent',
    'TaskClassifier',
    'TaskVerdict',
    'HierarchicalPlanner',
    'CritiqueAgent',
]
