"""
ConstraintVerifier
==================
Deterministically checks whether a replanned PlanSchema satisfies
the ConstraintManifest that triggered the replan.

Rules are pattern-based -- no LLM involved.
Returns a list of unsatisfied constraint strings (empty = all satisfied).
"""

from ...schemas import PlanSchema, ConstraintManifest
import re


# Keywords that indicate a task writes an entrypoint file
ENTRYPOINT_PATTERNS = [
    r"main\.py", r"app\.py", r"run\.py", r"server\.py",
    r"entrypoint", r"entry point", r"start the application",
    r"start_server", r"if __name__.*__main__",
]

# Keywords that indicate a task is a code-writing task
CODE_TASK_KEYWORDS = ["write", "implement", "create", "build", "generate code"]


class ConstraintVerifier:
    """
    Deterministic constraint satisfaction checker.

    Usage:
        violations = ConstraintVerifier.check(plan, manifest)
        if violations:
            # replan again or abort with clear error
    """

    @classmethod
    def check(cls, plan: PlanSchema, manifest: ConstraintManifest) -> list[str]:
        """
        Returns list of unsatisfied constraint descriptions.
        Empty list means all constraints are satisfied.
        """
        violations = []

        # 1. Check must_include_tasks
        for required_task_desc in manifest.must_include_tasks:
            if not cls._plan_satisfies_required_task(plan, required_task_desc):
                violations.append(
                    f"MISSING REQUIRED TASK: {required_task_desc}"
                )

        # 2. Check structural_constraints (pattern-match against task goals + output files)
        for constraint in manifest.structural_constraints:
            if not cls._plan_satisfies_structural(plan, constraint):
                violations.append(
                    f"STRUCTURAL CONSTRAINT UNSATISFIED: {constraint}"
                )

        # 3. Check must_not_combine
        for pair in manifest.must_not_combine:
            if len(pair) == 2:
                violation = cls._check_not_combined(plan, pair[0], pair[1])
                if violation:
                    violations.append(
                        f"ILLEGAL COMBINATION [{pair[0]}+{pair[1]}]: {violation}"
                    )

        return violations

    @classmethod
    def _plan_satisfies_required_task(cls, plan: PlanSchema, desc: str) -> bool:
        """
        Check if any task in the plan satisfies a must_include_tasks description.

        Strategy:
          - Detect the constraint type from the description string
          - Apply the appropriate check
          - Unknown constraint types: fall back to keyword overlap check
        """
        desc_lower = desc.lower()

        # Entrypoint check
        if "entrypoint" in desc_lower or "main.py" in desc_lower or "app.py" in desc_lower:
            return cls._plan_has_entrypoint(plan)

        # Dependency check (import chain)
        if "import" in desc_lower and "dependency" in desc_lower:
            return cls._plan_has_dependency_resolution(plan)

        # Fallback: keyword overlap between desc and any task goal
        desc_keywords = set(re.findall(r'\b\w{4,}\b', desc_lower))
        for task in plan.tasks:
            task_keywords = set(re.findall(r'\b\w{4,}\b', task.goal.lower()))
            overlap = desc_keywords & task_keywords
            # Require meaningful overlap (>= 2 shared content words)
            content_words = overlap - {
                "task", "file", "that", "this", "with", "from", "into",
                "write", "code", "python", "module", "function", "class"
            }
            if len(content_words) >= 2:
                return True

        return False

    @classmethod
    def _plan_has_entrypoint(cls, plan: PlanSchema) -> bool:
        """
        Returns True if the plan includes a task that writes a runnable entrypoint.

        Checks:
          1. Any task goal matches entrypoint patterns
          2. Any output_contract output_file is main.py, app.py, run.py, server.py
          3. Any task goal mentions 'if __name__' or 'start_server'
        """
        entrypoint_filenames = {"main.py", "app.py", "run.py", "server.py"}

        for task in plan.tasks:
            goal_lower = task.goal.lower()

            # Check goal text for entrypoint patterns
            for pattern in ENTRYPOINT_PATTERNS:
                if re.search(pattern, goal_lower):
                    return True

            # Check output_contract for entrypoint filenames
            if isinstance(task.output_contract, dict):
                output_file = task.output_contract.get("output_file", "")
                filename = output_file.split("/")[-1].lower()
                if filename in entrypoint_filenames:
                    # Additional check: task must not ONLY be a passive module
                    # (a "server.py" that starts the server counts;
                    #  a "server.py" that only defines a class may not)
                    if any(kw in goal_lower for kw in
                           ["start", "run", "launch", "entry", "main", "__main__"]):
                        return True
                    # If the component is named Server and has start_server interface, count it
                    component_name = task.output_contract.get("component_name", "")
                    interface = task.output_contract.get("public_interface", "")
                    if "server" in component_name.lower() and "start" in interface.lower():
                        return True

        return False

    @classmethod
    def _plan_has_dependency_resolution(cls, plan: PlanSchema) -> bool:
        """
        Check that all declared imports in output_contracts resolve to actual tasks.
        Returns True if no broken dependencies exist.
        """
        # Collect all produced component names
        produced = set()
        for task in plan.tasks:
            if isinstance(task.output_contract, dict):
                name = task.output_contract.get("component_name", "")
                if name:
                    produced.add(name)

        # Check all declared dependencies exist
        for task in plan.tasks:
            if isinstance(task.output_contract, dict):
                deps = task.output_contract.get("depends_on", [])
                for dep in deps:
                    if dep not in produced:
                        return False

        return True

    @classmethod
    def _plan_satisfies_structural(cls, plan: PlanSchema, constraint: str) -> bool:
        """
        Check a structural constraint string against the plan.
        Extracts the finding_type and delegates to specific checks.
        """
        constraint_lower = constraint.lower()

        # Route by finding_type token in constraint string
        if "missing_entrypoint" in constraint_lower:
            return cls._plan_has_entrypoint(plan)
        if "import_cycle" in constraint_lower:
            return not cls._plan_has_import_cycle(plan)
        if "broken_dependency" in constraint_lower:
            return cls._plan_has_dependency_resolution(plan)
        if "incomplete_artifact_set" in constraint_lower:
            return cls._plan_has_entrypoint(plan) and cls._plan_has_dependency_resolution(plan)

        # Generic: keyword overlap -- unknown structural constraint, assume satisfied
        return True

    @classmethod
    def _check_not_combined(cls, plan: PlanSchema, type_a: str, type_b: str) -> str | None:
        """
        Returns a description if any single task combines both type_a and type_b responsibilities.
        Returns None if no violation found.
        """
        for task in plan.tasks:
            goal_lower = task.goal.lower()
            agent = task.declared_agent or task.suggested_agent or ""

            has_a = type_a.lower() in goal_lower or type_a.lower() == agent.lower()
            has_b = type_b.lower() in goal_lower or type_b.lower() == agent.lower()

            if has_a and has_b:
                return f"Task {task.task_order}: '{task.goal[:80]}'"

        return None

    @classmethod
    def _plan_has_import_cycle(cls, plan: PlanSchema) -> bool:
        """
        Detect import cycles in the dependency graph.
        Uses DFS cycle detection on component depends_on graph.
        """
        # Build adjacency list
        graph: dict[str, list[str]] = {}
        for task in plan.tasks:
            if isinstance(task.output_contract, dict):
                name = task.output_contract.get("component_name", "")
                deps = task.output_contract.get("depends_on", [])
                if name:
                    graph[name] = deps

        # DFS cycle detection
        visited = set()
        rec_stack = set()

        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.discard(node)
            return False

        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    return True

        return False
