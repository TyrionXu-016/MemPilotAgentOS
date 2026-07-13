from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


MemoryType = Literal["profile", "preference", "task", "scene", "feedback"]
ExecutionStatus = Literal["success", "failed", "skipped"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryRecord(BaseModel):
    id: int | None = None
    user_id: str
    memory_type: MemoryType
    content: str
    source: str
    importance: float = Field(default=0.5, ge=0, le=1)
    confidence: float = Field(default=0.8, ge=0, le=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    is_active: bool = True
    tags: list[str] = Field(default_factory=list)


class RetrievedMemory(BaseModel):
    memory: MemoryRecord
    score: float
    reason: str


class SkillSpec(BaseModel):
    name: str
    description: str
    required_params: list[str]
    output_type: str
    safe: bool = True


class PlanStep(BaseModel):
    skill: str
    params: dict[str, Any] = Field(default_factory=dict)


class PlanContext(BaseModel):
    scenario_group: str = "learning"
    interaction_style: str = "neutral"
    task_items: list[str] = Field(default_factory=lambda: ["review"])


class MemoryEvidence(BaseModel):
    memory_id: int | None
    memory_type: MemoryType
    content: str
    score: float


class Plan(BaseModel):
    goal: str
    basis: list[str]
    steps: list[PlanStep]
    planner_name: str = "layered_memory"
    context: PlanContext = Field(default_factory=PlanContext)
    evidence: list[MemoryEvidence] = Field(default_factory=list)
    used_memory_ids: list[int] = Field(default_factory=list)
    generation_metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionEvent(BaseModel):
    skill: str
    status: ExecutionStatus
    output: dict[str, Any]
    message: str


class EvaluationResult(BaseModel):
    planner_name: str
    scenario_count: int
    memory_hit_rate: float
    preference_match_rate: float
    executable_plan_rate: float
    task_completion_rate: float
