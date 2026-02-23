from schemas import RequestSchema, PlanSchema, TaskSchema
from storage import save_plan_to_sqlite
from .routing import TaskRouter
from ..planning.planner import PlanningAgent
from ..planning.hierarchical import HierarchicalPlanner  # NEW
from ..validation.validator import ValidationAgent
from config import USE_HIERARCHICAL_PLANNING  # NEW


class Orchestrator:
    """Orchestrates the full pipeline with error recovery."""

    def __init__(self, max_retries: int = 2):
        """
        Initialize orchestrator.

        Args:
            max_retries: Maximum retry attempts per task (default: 2)
                        Total attempts = 1 + max_retries (so 3 total with default)
        """
        self.max_retries = max_retries

    def execute_request(self, request: RequestSchema) -> PlanSchema:
        """
        Execute full pipeline: Planning → Execution → Validation (with retries).

        Args:
            request: The user's request

        Returns:
            PlanSchema: Completed plan with all tasks executed and validated
        """
        print("\n" + "=" * 80)
        print("🎯 ORCHESTRATOR: Starting Request Execution")
        print("=" * 80)

        # === STAGE 1: PLANNING ===
        print(f"\n📋 Stage 1: Planning")

        if USE_HIERARCHICAL_PLANNING:
            print(f"   Using: HierarchicalPlanner (with classifier)")
            plan = HierarchicalPlanner.create_plan(request)
        else:
            print(f"   Using: PlanningAgent (flat planning)")
            plan = PlanningAgent.create_plan(request)

        print(f"   Generated {len(plan.tasks)} task(s)")

        save_plan_to_sqlite(plan)
        print(f"   💾 Saved plan after planning")

        # === STAGE 2: EXECUTION + VALIDATION (with retries) ===
        print(f"\n💻 Stage 2: Execution & Validation")

        for i, task in enumerate(plan.tasks):
            print(f"\n{'─' * 80}")
            print(f"Task {i}: {task.goal[:60]}...")
            print(f"{'─' * 80}")

            success = self._execute_task_with_retry(task, plan)

            if success:
                print(f"✅ Task {i} completed successfully")
            else:
                print(f"❌ Task {i} failed after {task.retry_count} attempts")

        # Save after all execution/validation
        save_plan_to_sqlite(plan)
        print(f"\n💾 Saved plan after execution & validation")

        # === STAGE 3: FINAL STATUS ===
        print(f"\n📊 Stage 3: Final Status")

        validated_count = sum(1 for t in plan.tasks if t.status == "validated")
        failed_count = sum(1 for t in plan.tasks if t.status == "failed")

        if all(t.status == "validated" for t in plan.tasks):
            plan.status = "complete"
            print(f"   🎉 All tasks validated!")
        else:
            plan.status = "failed"
            print(f"   ⚠️  {failed_count}/{len(plan.tasks)} tasks failed")

        print(f"   Total tokens: {plan.token_usage}")

        # Final save with status
        save_plan_to_sqlite(plan)
        print(f"   💾 Saved final plan")

        print("\n" + "=" * 80)
        print(f"🎯 ORCHESTRATOR: Request {'Complete' if plan.status == 'complete' else 'Failed'}")
        print("=" * 80)

        return plan

    def _execute_task_with_retry(self, task: TaskSchema, plan: PlanSchema) -> bool:
        """
        Execute a task with retry logic.

        Args:
            task: The task to execute
            plan: The parent plan (for token tracking)

        Returns:
            bool: True if task succeeded, False if failed after all retries
        """
        attempt = 0

        while attempt <= self.max_retries:
            print(f"\n   Attempt {attempt + 1}/{self.max_retries + 1}")

            # Build context from previous tasks in this plan
            context = {
                'previous_tasks': [
                    {
                        'order': t.task_order,
                        'goal': t.goal,
                        'result': t.result,
                        'status': t.status
                    }
                    for t in plan.tasks
                    if t.task_order < task.task_order and t.result is not None
                ]
            }

            # Execute: Route to appropriate agent (with context)
            updated_task, exec_tokens = TaskRouter.route_task(task, context)
            plan.token_usage += exec_tokens

            # Update task reference in plan
            task.status = updated_task.status
            task.result = updated_task.result
            task.completed_at = updated_task.completed_at

            # Check if task was routed to unsupported agent
            if task.status == "failed" and "not yet implemented" in (task.error_message or ""):
                print(f"      ❌ Task type not supported, cannot retry")
                return False

            print(f"      Code generated ({exec_tokens} tokens)")

            # Validate: Test the code (only for code tasks)
            validated_task, val_tokens = ValidationAgent.validate_task(task)
            plan.token_usage += val_tokens

            # Update task reference in plan
            task.status = validated_task.status
            task.test_code = validated_task.test_code
            task.error_message = validated_task.error_message

            print(f"      Validation complete ({val_tokens} tokens)")

            # Check if successful
            if task.status == "validated":
                return True

            # Failed - check if we should retry
            task.retry_count = attempt + 1

            if attempt < self.max_retries:
                print(f"      ⚠️  Validation failed, preparing retry...")
                self._add_retry_feedback(task)
            else:
                print(f"      ❌ Max retries reached")
                return False

            attempt += 1

        return False


    def _add_retry_feedback(self, task: TaskSchema):
        """
        Add feedback to task goal to help CodingAgent fix issues on retry.

        Appends error information to the task goal so the next attempt
        has context about what went wrong.
        """
        if task.error_message:
            # Extract the key error info (first 200 chars)
            error_summary = task.error_message[:200]

            feedback = f"""

PREVIOUS ATTEMPT FAILED:
{error_summary}

Please fix the issue and regenerate the code.

Failed test code for reference:
{task.test_code[:300] if task.test_code else 'No test code available'}
"""

            # Append feedback to goal
            task.goal = task.goal + feedback

            print(f"      📝 Added error feedback to task goal")
