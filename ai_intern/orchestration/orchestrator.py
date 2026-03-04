from ..schemas import RequestSchema, PlanSchema, TaskSchema, TaskOutput
from ..storage import save_plan_to_sqlite
from .routing import TaskRouter
from ..planning.planner import PlanningAgent
from ..planning.hierarchical import HierarchicalPlanner
from ..validation.validator import ValidationAgent
from ..validation.error_classifier import ErrorClassifier, ErrorCategory
from ..workspace import ProjectWorkspace
from ..llm import call_ollama_code
from ..preprocessing import preprocess_request
from ..config import settings
from ..logging_config import get_logger
import subprocess
import tempfile
import os
import threading

logger = get_logger("orchestrator")


class Orchestrator:
    """Orchestrates the full pipeline with error recovery."""

    def __init__(self, max_retries: int = None):
        self.max_retries = max_retries if max_retries is not None else settings.MAX_RETRIES
        self._token_lock = threading.Lock()

    def execute_request(self, request: RequestSchema) -> PlanSchema:
        """Execute full pipeline: Planning → Execution → Validation (with retries)."""
        logger.info("ORCHESTRATOR: Starting Request Execution")

        # === STAGE 0: PREPROCESSING (#25) ===
        request.content = preprocess_request(request.content)

        # === STAGE 1: PLANNING ===
        logger.info("Stage 1: Planning")

        if settings.USE_HIERARCHICAL_PLANNING:
            logger.info("Using: HierarchicalPlanner (with classifier)")
            plan = HierarchicalPlanner.create_plan(request)
        else:
            logger.info("Using: PlanningAgent (flat planning)")
            plan = PlanningAgent.create_plan(request)

        logger.info(f"Generated {len(plan.tasks)} task(s)")

        # Log app spec summary if one was generated
        if plan.app_spec:
            spec = plan.app_spec
            logger.info("=== APP SPEC GENERATED ===")
            logger.info(f"Summary: {spec.get('summary', '')}")
            logger.info(f"Features: {'; '.join(spec.get('features', []))}")
            components = spec.get('components', [])
            logger.info(f"Components ({len(components)}):")
            for c in components:
                logger.info(
                    f"  - {c['name']} → {c['output_file']} | {c['public_interface']}"
                )
            logger.info(f"Done criteria: {'; '.join(spec.get('done_criteria', []))}")
            logger.info("==========================")

        # Auto-detect dependencies (#19)
        self._auto_detect_dependencies(plan)

        # Preserve original goals (#9)
        for task in plan.tasks:
            if not task.original_goal:
                task.original_goal = task.goal

        save_plan_to_sqlite(plan)
        logger.debug("Saved plan after planning")

        # === STAGE 2: EXECUTION + VALIDATION (with retries) ===
        logger.info("Stage 2: Execution & Validation")

        # Create project workspace if this is an app-scale plan
        workspace: ProjectWorkspace | None = None
        if plan.app_spec:
            workspace = ProjectWorkspace(plan.plan_id)
            plan.workspace_root = str(workspace.root)
            logger.info(f"Workspace created: {workspace.root}")

        for i, task in enumerate(plan.tasks):
            logger.info(f"Task {i}: {task.goal[:60]}...")

            # Check dependencies (#19) - skip if any dependency failed
            if task.depends_on:
                deps_met = all(
                    plan.tasks[dep].status in ("validated", "complete")
                    for dep in task.depends_on
                    if dep < len(plan.tasks)
                )
                if not deps_met:
                    task.status = "skipped"
                    task.error_message = "Skipped: dependency failed"
                    logger.warning(f"Task {i} skipped (dependency failed)")
                    continue

            success = self._execute_task_with_retry(task, plan, workspace=workspace)

            if success:
                logger.info(f"Task {i} completed successfully")
            else:
                logger.error(f"Task {i} failed after {task.retry_count} attempts")

            # Save after each task (#21)
            save_plan_to_sqlite(plan)

        logger.debug("Saved plan after execution & validation")

        # === STAGE 3: FINAL ANSWER SYNTHESIS (#6) ===
        self._synthesize_final_answer(plan)

        # === STAGE 4: FINAL STATUS ===
        failed_count = sum(1 for t in plan.tasks if t.status == "failed")
        skipped_count = sum(1 for t in plan.tasks if t.status == "skipped")

        if all(t.status in ("validated", "complete") for t in plan.tasks):
            plan.status = "complete"
            logger.info("All tasks validated!")
        elif skipped_count > 0 and failed_count > 0:
            plan.status = "failed"
            logger.warning(f"{failed_count} failed, {skipped_count} skipped out of {len(plan.tasks)} tasks")
        else:
            plan.status = "failed"
            logger.warning(f"{failed_count}/{len(plan.tasks)} tasks failed")

        logger.info(f"Total tokens: {plan.token_usage}")

        if plan.final_answer:
            logger.info(f"Final Answer:\n{plan.final_answer[:200]}")

        save_plan_to_sqlite(plan)
        logger.debug("Saved final plan")

        logger.info(f"ORCHESTRATOR: Request {'Complete' if plan.status == 'complete' else 'Failed'}")

        return plan

    def _execute_task_with_retry(
        self,
        task: TaskSchema,
        plan: PlanSchema,
        workspace: ProjectWorkspace | None = None,
    ) -> bool:
        """Execute a task with retry logic."""
        attempt = 0

        while attempt <= self.max_retries:
            logger.info(f"Attempt {attempt + 1}/{self.max_retries + 1}")

            # Reset goal to original on retry (#9)
            if attempt > 0 and task.original_goal:
                task.goal = task.original_goal

            # Build context from previous tasks (#5 - use data_summary when available)
            context = {
                'previous_tasks': [
                    {
                        'order': t.task_order,
                        'goal': t.goal,
                        'result': t.result,
                        'status': t.status,
                        'data_summary': (
                            t.task_output.data_summary
                            if t.task_output and t.task_output.data_summary
                            else None
                        ),
                        'file_path': (
                            t.task_output.file_path
                            if t.task_output and t.task_output.file_path
                            else None
                        ),
                        'key_values': (
                            t.task_output.key_values
                            if t.task_output and t.task_output.key_values
                            else None
                        ),
                    }
                    for t in plan.tasks
                    if t.task_order < task.task_order and t.result is not None
                ]
            }

            # Inject workspace context for tasks that depend on prior components
            if workspace and task.output_contract:
                depends_on_names = task.output_contract.get("depends_on", [])
                # Fall back to spec component depends_on if not in contract
                if not depends_on_names and plan.app_spec:
                    for comp in plan.app_spec.get("components", []):
                        if comp["name"] == task.output_contract.get("component_name"):
                            depends_on_names = comp.get("depends_on", [])
                            break

                if depends_on_names:
                    workspace_ctx = workspace.get_context_for_task(depends_on_names)
                    import_lines = workspace.resolve_imports(depends_on_names)
                    if workspace_ctx:
                        context['workspace_context'] = workspace_ctx
                    if import_lines:
                        context['workspace_imports'] = import_lines

            # Pass error history to context for retry (#9)
            if task.error_history:
                context['error_history'] = task.error_history

            # Execute: Route to appropriate agent (with context)
            updated_task, exec_tokens = TaskRouter.route_task(task, context)
            plan.token_usage += exec_tokens

            # Update task reference in plan
            task.status = updated_task.status
            task.result = updated_task.result
            task.completed_at = updated_task.completed_at
            if updated_task.task_output:
                task.task_output = updated_task.task_output

            # Check if task was routed to unsupported agent
            if task.status == "failed" and "not yet implemented" in (task.error_message or ""):
                logger.error("Task type not supported, cannot retry")
                return False

            logger.debug(f"Code generated ({exec_tokens} tokens)")

            # Validate: Test the code (only for code tasks)
            validated_task, val_tokens = ValidationAgent.validate_task(task)
            plan.token_usage += val_tokens

            # Update task reference in plan
            task.status = validated_task.status
            task.test_code = validated_task.test_code
            task.error_message = validated_task.error_message

            logger.debug(f"Validation complete ({val_tokens} tokens)")

            # Check if successful
            if task.status == "validated":
                # For code tasks that compute results, execute and capture output (#22)
                task_type = TaskRouter.classify_task(task)
                if task_type == "code":
                    self._try_execute_code(task)

                # Register component in workspace manifest and write file
                if workspace and task.output_contract:
                    contract = task.output_contract
                    raw_output_file = contract.get("output_file", "")
                    # Derive the filename within the workspace
                    rel_file = os.path.basename(raw_output_file) if raw_output_file else ""
                    if not rel_file:
                        rel_file = raw_output_file.removeprefix("outputs/").lstrip("/")

                    # Write code to workspace/{plan_id}/storage.py etc.
                    if rel_file and task.result:
                        from pathlib import Path as _Path
                        ws_file_path = _Path(workspace.root) / rel_file
                        ws_file_path.write_text(task.result, encoding="utf-8")
                        logger.info(f"Wrote component to workspace: {rel_file}")
                        # Record the path on task_output so it's inspectable
                        if not task.task_output:
                            task.task_output = TaskOutput(
                                output_type="code", raw_result=task.result
                            )
                        task.task_output.file_path = str(ws_file_path)

                    workspace.register_component(
                        name=contract["component_name"],
                        file_path=rel_file,
                        interface=contract["public_interface"],
                        depends_on=contract.get("depends_on", []),
                    )

                return True

            # Classify the error before recording
            category = ErrorClassifier.classify(task)
            task.error_category = category.value

            # Failed - record to error history (#9)
            task.error_history.append({
                'attempt': attempt + 1,
                'error': task.error_message[:300] if task.error_message else 'Unknown error',
                'test_code': task.test_code[:300] if task.test_code else None,
                'category': category.value,
            })

            # Early abort on unrecoverable errors
            if category.value.startswith("unrecoverable"):
                logger.error(f"Unrecoverable error ({category.value}), skipping retries")
                task.retry_count = attempt + 1
                return False

            task.retry_count = attempt + 1

            if attempt < self.max_retries:
                logger.warning("Validation failed, preparing retry...")
            else:
                logger.error("Max retries reached")
                return False

            attempt += 1

        return False

    def _synthesize_final_answer(self, plan: PlanSchema):
        """Synthesize a final answer from all task results (#6)."""
        completed_tasks = [t for t in plan.tasks if t.status in ("validated", "complete")]

        if not completed_tasks:
            plan.final_answer = "No tasks completed successfully."
            return

        # Workspace-aware final answer for app-scale plans
        if plan.workspace_root:
            plan.final_answer = self._synthesize_workspace_answer(plan, completed_tasks)
            return

        # If the last completed task produced a file path, report it
        last_task = completed_tasks[-1]
        if last_task.task_output and last_task.task_output.file_path:
            plan.final_answer = (
                f"File output: {last_task.task_output.file_path}\n"
                f"{last_task.result or ''}"
            )
            return

        # If the last task has a direct result, use it
        if last_task.result:
            # For file writes, use the result directly
            if "wrote output to" in (last_task.result or "").lower():
                plan.final_answer = last_task.result
                return

            # For multi-task plans, try LLM synthesis
            if len(completed_tasks) > 1:
                plan.final_answer = self._llm_synthesize(plan, completed_tasks)
                return

            # Single task - just use the result
            plan.final_answer = last_task.result
            return

        plan.final_answer = "Plan completed but no result was generated."

    def _synthesize_workspace_answer(
        self, plan: PlanSchema, completed_tasks: list[TaskSchema]
    ) -> str:
        """Build the final answer for an app-scale plan with a workspace."""
        workspace_root = plan.workspace_root
        manifest_path = os.path.join(workspace_root, "project.json")

        # Read the manifest to list components and their files
        component_lines = []
        if os.path.exists(manifest_path):
            try:
                import json as _json
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = _json.load(f)
                for name, entry in manifest.items():
                    fp = entry.get("file_path", "")
                    iface = entry.get("interface", "")
                    deps = entry.get("depends_on", [])
                    line = f"  - {fp:<20} → {name}: {iface}"
                    if deps:
                        line += f"  (depends on: {', '.join(deps)})"
                    component_lines.append(line)
            except Exception as e:
                logger.warning(f"Failed to read manifest for final answer: {e}")

        # Detect an entry point
        entry_point = ""
        for t in completed_tasks:
            goal_lower = t.goal.lower()
            if "cli" in goal_lower or "main" in goal_lower or "entry" in goal_lower:
                if t.task_output and t.task_output.file_path:
                    entry_point = t.task_output.file_path

        lines = [f"Application generated in: {workspace_root}", "", "Files:"]
        if component_lines:
            lines.extend(component_lines)
        else:
            lines.append("  (no components registered)")

        if entry_point:
            lines.append("")
            lines.append(f"To run: python {entry_point}")

        return "\n".join(lines)

    def _llm_synthesize(self, plan: PlanSchema, completed_tasks: list[TaskSchema]) -> str:
        """Use LLM to synthesize final answer from multiple task results."""
        task_summaries = []
        for t in completed_tasks:
            summary = f"Task {t.task_order} ({t.goal[:80]}): "
            if t.task_output and t.task_output.data_summary:
                summary += t.task_output.data_summary
            elif t.result:
                summary += t.result[:300]
            else:
                summary += "Completed"
            task_summaries.append(summary)

        prompt = (
            f"Original request: {plan.tasks[0].original_goal or plan.tasks[0].goal}\n\n"
            f"Task results:\n" + "\n".join(task_summaries) + "\n\n"
            "Synthesize a clear, concise final answer for the user. "
            "Focus on the computed values, outcomes, or key findings. "
            "Keep it under 200 words."
        )

        try:
            answer, tokens = call_ollama_code(
                model=settings.PLANNING_MODEL,
                prompt=prompt,
                system="You are a helpful assistant that synthesizes task results into clear answers.",
                temperature=0.2
            )
            plan.token_usage += tokens
            return answer.strip()
        except Exception as e:
            logger.warning(f"Final answer synthesis failed: {e}")
            # Fallback: return last task result
            return completed_tasks[-1].result or "Plan completed."

    @staticmethod
    def _try_execute_code(task: TaskSchema):
        """Try to execute validated code and capture stdout as execution result (#22)."""
        if not task.result:
            return

        goal_lower = task.goal.lower()
        # Only execute for compute/calculate/analyze tasks, not for code-as-deliverable
        exec_keywords = ["calculate", "compute", "analyze", "average", "sum", "total", "count", "find"]
        if not any(kw in goal_lower for kw in exec_keywords):
            return

        code = task.result
        # Add a main call if the code defines functions but doesn't call them
        lines = code.strip().split("\n")
        has_function = any(l.strip().startswith("def ") for l in lines)
        has_call = any(not l.startswith(" ") and not l.startswith("def ") and not l.startswith("import ")
                       and not l.startswith("from ") and not l.startswith("#") and l.strip()
                       for l in lines)

        if has_function and not has_call:
            # Try to find the main function and call it
            for l in lines:
                if l.startswith("def "):
                    func_name = l.split("(")[0].replace("def ", "").strip()
                    code += f"\n\nresult = {func_name}()\nif result is not None:\n    print(result)\n"
                    break

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            temp_path = f.name

        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            result = subprocess.run(
                ["python", temp_path],
                capture_output=True, text=True,
                timeout=settings.CODE_EXEC_TIMEOUT,
                encoding="utf-8", env=env,
            )

            if result.returncode == 0 and result.stdout.strip():
                execution_output = result.stdout.strip()
                logger.info(f"Code execution output: {execution_output[:200]}")
                # Store execution result in task_output
                if not task.task_output:
                    task.task_output = TaskOutput(output_type="code", raw_result=task.result)
                task.task_output.data_summary = execution_output
        except (subprocess.TimeoutExpired, Exception) as e:
            logger.debug(f"Code execution skipped: {e}")
        finally:
            os.unlink(temp_path)

    @staticmethod
    def _auto_detect_dependencies(plan: PlanSchema):
        """Auto-detect task dependencies based on goal text (#19)."""
        dependency_keywords = [
            "previous", "results", "from task", "above", "earlier",
            "using the", "from the", "the research", "the data",
            "the code", "the analysis", "the summary",
        ]

        for task in plan.tasks:
            if task.task_order == 0:
                continue  # First task has no dependencies

            goal_lower = task.goal.lower()

            # Check if goal references previous task output
            if any(kw in goal_lower for kw in dependency_keywords):
                task.depends_on = [task.task_order - 1]
                logger.debug(f"Task {task.task_order} depends on Task {task.task_order - 1}")
            elif task.task_order > 0:
                # For sequential plans, assume each task depends on the previous
                task.depends_on = [task.task_order - 1]
                logger.debug(f"Task {task.task_order} sequential dependency on Task {task.task_order - 1}")
