from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal, Optional
from uuid import uuid4

class RequestSchema(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    schema_version: str = "0.0.1"
    created_at: datetime = Field(default_factory=datetime.utcnow)

class TaskSchema(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    plan_id: str
    task_order: int
    goal: str
    status: Literal["pending", "executing", "validating", "complete", "validated", "failed"] = "pending"
    retry_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    schema_version: str = "0.0.1"
    result: Optional[str] = None  # ← Generated code goes here
    error_message: Optional[str] = None

class PlanSchema(BaseModel):
    plan_id: str = Field(default_factory=lambda: str(uuid4()))
    request_id: str
    tasks: list[TaskSchema] = []
    status: Literal["planning", "critiquing", "executing", "validating", "complete", "failed"] = "planning"
    retry_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    schema_version: str = "0.0.1"
    token_usage: int = 0