"""Tests for Pydantic schemas."""
import pytest
from ai_intern.schemas import TaskSchema, PlanSchema, RequestSchema, TaskOutput


class TestRequestSchema:
    def test_create_request(self):
        req = RequestSchema(content="test query")
        assert req.content == "test query"
        assert req.request_id  # auto-generated
        assert req.schema_version == "0.0.1"

    def test_unique_ids(self):
        r1 = RequestSchema(content="a")
        r2 = RequestSchema(content="b")
        assert r1.request_id != r2.request_id


class TestTaskSchema:
    def test_create_task(self):
        task = TaskSchema(plan_id="p1", task_order=0, goal="test")
        assert task.status == "pending"
        assert task.retry_count == 0
        assert task.depends_on == []
        assert task.error_history == []
        assert task.original_goal == ""

    def test_status_values(self):
        for status in ["pending", "executing", "validating", "complete", "validated", "failed", "skipped"]:
            task = TaskSchema(plan_id="p", task_order=0, goal="t", status=status)
            assert task.status == status

    def test_invalid_status_rejected(self):
        with pytest.raises(Exception):
            TaskSchema(plan_id="p", task_order=0, goal="t", status="invalid")

    def test_task_output_assignment(self):
        task = TaskSchema(plan_id="p", task_order=0, goal="t")
        output = TaskOutput(output_type="data", raw_result="test data")
        task.task_output = output
        assert task.task_output.output_type == "data"

    def test_error_history_tracking(self):
        task = TaskSchema(plan_id="p", task_order=0, goal="t")
        task.error_history.append({"attempt": 1, "error": "fail"})
        assert len(task.error_history) == 1


class TestTaskOutput:
    def test_default_values(self):
        output = TaskOutput()
        assert output.output_type == "text"
        assert output.raw_result == ""
        assert output.column_names is None

    def test_data_output(self):
        output = TaskOutput(
            output_type="data",
            raw_result="csv content",
            column_names=["Date", "Calories"],
            row_count=45,
            data_summary="CSV with 45 rows",
        )
        assert output.row_count == 45
        assert len(output.column_names) == 2


class TestPlanSchema:
    def test_create_plan(self):
        plan = PlanSchema(request_id="r1")
        assert plan.status == "planning"
        assert plan.tasks == []
        assert plan.token_usage == 0
        assert plan.final_answer is None

    def test_plan_with_tasks(self):
        plan = PlanSchema(request_id="r1")
        task = TaskSchema(plan_id=plan.plan_id, task_order=0, goal="test")
        plan.tasks.append(task)
        assert len(plan.tasks) == 1

    def test_serialization_roundtrip(self):
        plan = PlanSchema(request_id="r1")
        plan.tasks.append(TaskSchema(plan_id=plan.plan_id, task_order=0, goal="g"))
        data = plan.model_dump(mode="json")
        plan2 = PlanSchema(**data)
        assert plan2.plan_id == plan.plan_id
        assert len(plan2.tasks) == 1
