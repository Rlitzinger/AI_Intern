from ..schemas import RequestSchema, PlanSchema, TaskSchema, TaskOutput, OutputContract, CritiqueResult, ConstraintManifest
from ..storage import save_plan_to_sqlite, mark_findings_survived
from .routing import TaskRouter
from ..planning.planner import PlanningAgent
from ..planning.hierarchical import HierarchicalPlanner
from ..planning.red_team.council import RedTeamCouncil
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
        print(f"\nStage 1: Planning")
        logger.info("Stage 1: Planning")

        if settings.USE_HIERARCHICAL_PLANNING:
            print(f"   Using: HierarchicalPlanner (with classifier)")
            logger.info("Using: HierarchicalPlanner (with classifier)")
            plan = HierarchicalPlanner.create_plan(request)
        else:
            print(f"   Using: PlanningAgent (flat planning)")
            logger.info("Using: PlanningAgent (flat planning)")
            plan = PlanningAgent.create_plan(request)

        print(f"   Generated {len(plan.tasks)} task(s)")
        logger.info(f"Generated {len(plan.tasks)} task(s)")
        save_plan_to_sqlite(plan)
        print(f"   Saved plan after planning")

        # === STAGE 1.5: RED TEAM COUNCIL ===
        print(f"\nStage 1.5: Red Team Council")

        from ..planning.red_team.constraint_verifier import ConstraintVerifier
        from ..storage import save_constraint_audit

        import uuid as _uuid
        council_session_id = str(_uuid.uuid4())
        MAX_REPLAN_CYCLES = 3  # Hard cap. Prevents infinite loops on stubborn constraints.

        manifest = None
        for replan_cycle in range(MAX_REPLAN_CYCLES + 1):
            # First iteration: review the initial plan
            # Subsequent iterations: review the replanned plan
            council_verdict, council_tokens = RedTeamCouncil.review(
                plan, request,
                cycle=replan_cycle,
                session_id=council_session_id
            )
            plan.token_usage += council_tokens

            if council_verdict.approved:
                if replan_cycle > 0:
                    print(f"   Plan approved after {replan_cycle} replan cycle(s)")
                    if manifest:
                        # Verify deterministically even when council approves after replan
                        # (council can be fooled; verifier cannot)
                        violations = ConstraintVerifier.check(plan, manifest)
                        satisfied_count = len(manifest.must_include_tasks) - len(violations)
                        save_constraint_audit(
                            council_session_id, plan.plan_id, replan_cycle,
                            violations, max(satisfied_count, 0)
                        )
                        if violations:
                            print(f"   WARNING: Council approved but {len(violations)} constraint(s) still unmet:")
                            for v in violations:
                                print(f"     - {v}")
                            # Log to storage for inspection but don't block execution
                            # (we've already replanned; proceeding is better than infinite loop)
                break

            # Council rejected the plan
            if not council_verdict.constraint_manifest:
                # No manifest means council found issues but couldn't generate constraints
                print(f"   Council rejected plan (cycle {replan_cycle}) but no manifest generated -- proceeding")
                break

            manifest = council_verdict.constraint_manifest
            n = len(council_verdict.high_confidence_findings)

            # Mark whether findings survived from previous cycle
            if replan_cycle > 0:
                mark_findings_survived(council_session_id, cycle=replan_cycle - 1, survived=True)

            if replan_cycle >= MAX_REPLAN_CYCLES:
                # Exhausted replan budget
                violations = ConstraintVerifier.check(plan, manifest)
                save_constraint_audit(
                    council_session_id, plan.plan_id, replan_cycle,
                    violations, len(manifest.must_include_tasks) - len(violations)
                )
                print(f"   Max replan cycles ({MAX_REPLAN_CYCLES}) reached.")
                print(f"   {len(violations)} constraint(s) remain unmet after {MAX_REPLAN_CYCLES} attempts:")
                for v in violations:
                    print(f"     - {v}")
                print(f"   Proceeding with best available plan.")
                break

            print(f"\n   Cycle {replan_cycle + 1}/{MAX_REPLAN_CYCLES}: {n} blocking issue(s) -- replanning...")

            # Replan
            plan = self._replan_with_manifest(request, manifest)
            plan.token_usage += council_tokens  # approximate carry-forward

            # DETERMINISTIC VERIFICATION: Check before burning another council review
            violations = ConstraintVerifier.check(plan, manifest)
            save_constraint_audit(
                council_session_id, plan.plan_id, replan_cycle,
                violations, len(manifest.must_include_tasks) - len(violations)
            )
            if violations:
                print(f"   Post-replan verifier: {len(violations)} constraint(s) still unmet after replan:")
                for v in violations:
                    print(f"     - {v}")
                # Don't break -- let council review run. Council may catch different issues.
                # But this surfaces the problem immediately in the terminal output.
            else:
                print(f"   Post-replan verifier: all {len(manifest.must_include_tasks)} required tasks present \u2713")

        save_plan_to_sqlite(plan)
        print(f"   Saved plan after council review")

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

        # Preserve original goals (#9)
        for task in plan.tasks:
            if not task.original_goal:
                task.original_goal = task.goal

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
            # When depends_on is set, only include declared dependencies (focused context).
            # When depends_on is empty, include all previous tasks (backward compat).
            if task.depends_on:
                relevant_tasks = [
                    t for t in plan.tasks
                    if t.task_order in task.depends_on and t.result is not None
                ]
            else:
                relevant_tasks = [
                    t for t in plan.tasks
                    if t.task_order < task.task_order and t.result is not None
                ]

            context = {
                'previous_tasks': [
                    {
                        'order': t.task_order,
                        'goal': t.goal,
                        # Prefer key_values over raw result — suppress result when key_values present
                        'key_values': (
                            t.task_output.key_values
                            if t.task_output and t.task_output.key_values
                            else None
                        ),
                        'result': (
                            t.result
                            if not (t.task_output and t.task_output.key_values)
                            else None
                        ),
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
                    }
                    for t in relevant_tasks
                ]
            }

            # Inject workspace context for tasks that depend on prior components
            # Only for spec-driven (dict) contracts, not planning OutputContract models
            if workspace and isinstance(task.output_contract, dict):
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
                # Contract validation — check output has required keys
                missing_keys = self._validate_contract(task)
                if missing_keys and attempt < self.max_retries:
                    logger.warning(f"Contract violation: missing keys {missing_keys}")
                    task.error_history.append({
                        'attempt': attempt + 1,
                        'error': f"Output missing required keys: {missing_keys}",
                        'test_code': None,
                        'category': 'contract_violation',
                    })
                    task.status = "pending"
                    attempt += 1
                    continue

                # For code tasks that compute results, execute and capture output (#22)
                task_type = TaskRouter.classify_task(task)
                if task_type == "code":
                    self._try_execute_code(task)

                # Register component in workspace manifest and write file
                if workspace and isinstance(task.output_contract, dict):
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
    def _validate_contract(task: TaskSchema) -> list[str]:
        """
        Check that task output matches its OutputContract.
        Returns list of missing keys (empty = valid).
        """
        contract = task.output_contract
        if not isinstance(contract, OutputContract):
            return []  # No contract to validate (spec-driven dict or no contract)
        if not contract.expected_keys:
            return []  # Contract doesn't specify keys

        if not task.task_output or not task.task_output.key_values:
            return contract.expected_keys  # All keys missing

        produced_keys = set(task.task_output.key_values.keys())
        expected_keys = set(contract.expected_keys)
        missing = expected_keys - produced_keys

        return list(missing)

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

                # Try parsing stdout as JSON for key_values
                import json as _json
                try:
                    parsed = _json.loads(execution_output)
                    if isinstance(parsed, dict):
                        task.task_output.key_values = parsed
                        logger.info(f"Parsed {len(parsed)} key-value pairs from code output")
                except _json.JSONDecodeError:
                    pass  # stdout wasn't JSON — data_summary already set
        except (subprocess.TimeoutExpired, Exception) as e:
            logger.debug(f"Code execution skipped: {e}")
        finally:
            os.unlink(temp_path)

    def _replan_with_manifest(
        self, request: RequestSchema, manifest: ConstraintManifest
    ) -> PlanSchema:
        """Re-plan with structured constraints. Not freeform feedback."""
        lines = []
        for task_idx, c in manifest.task_constraints.items():
            lines.append(f"- Task {task_idx} must satisfy: {c}")
        for c in manifest.structural_constraints:
            lines.append(f"- STRUCTURAL: {c}")
        for goal in manifest.must_include_tasks:
            lines.append(f"- MUST INCLUDE A TASK FOR: {goal}")
        for pair in manifest.must_not_combine:
            if len(pair) == 2:
                lines.append(f"- NEVER combine [{pair[0]}] and [{pair[1]}] in one task")

        augmented_content = (
            f"{request.content}\n\n"
            "HARD PLANNING CONSTRAINTS (non-negotiable, from adversarial review):\n"
            + "\n".join(lines)
        )
        augmented_request = RequestSchema(
            request_id=request.request_id,
            content=augmented_content
        )
        if settings.USE_HIERARCHICAL_PLANNING:
            return HierarchicalPlanner.create_plan(augmented_request)
        else:
            return PlanningAgent.create_plan(augmented_request)

