from ...schemas import PlanSchema


def format_plan_for_prompt(plan: PlanSchema) -> str:
    """Format all tasks with agent, goal, and contract for LLM prompts."""
    lines = []
    for t in plan.tasks:
        agent = t.declared_agent or t.suggested_agent or "unknown"
        lines.append(f"Task {t.task_order} [{agent}]: {t.goal[:120]}")
        if isinstance(t.output_contract, dict):
            lines.append(f"  output_file : {t.output_contract.get('output_file', 'none')}")
            lines.append(f"  interface   : {t.output_contract.get('public_interface', 'none')}")
            deps = t.output_contract.get("depends_on", [])
            if deps:
                lines.append(f"  depends_on  : {deps}")
        elif t.output_contract is not None:
            oc = t.output_contract
            lines.append(f"  output_type : {oc.output_type}")
            lines.append(f"  output_format: {oc.output_format}")
            lines.append(f"  consumed_by : {oc.required_by_tasks}")
    return "\n".join(lines)


def format_artifacts_for_prompt(plan: PlanSchema) -> str:
    """Format only spec-driven artifacts (component files + interfaces)."""
    lines = []
    for t in plan.tasks:
        if isinstance(t.output_contract, dict):
            name = t.output_contract.get("component_name", "?")
            file = t.output_contract.get("output_file", "?")
            iface = t.output_contract.get("public_interface", "?")
            deps = t.output_contract.get("depends_on", [])
            line = f"{file}  ({name}): {iface}"
            if deps:
                line += f"  -- imports: {deps}"
            lines.append(line)
    if not lines:
        return "(No spec-driven artifacts -- non-app-scale plan)"
    return "\n".join(lines)
