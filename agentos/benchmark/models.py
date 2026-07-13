from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from agentos.models import MemoryRecord


ScenarioGroup = Literal["learning", "family_education", "home_service"]
Explicitness = Literal["full", "style_only", "item_only", "implicit"]


class BenchmarkMemory(MemoryRecord):
    memory_key: str


class BenchmarkCase(BaseModel):
    case_id: str
    scenario_group: ScenarioGroup
    user_id: str
    goal: str
    explicitness: Explicitness
    memories: list[BenchmarkMemory]
    relevant_memory_keys: list[str]
    expected_style: str
    expected_items: list[str] = Field(min_length=1)
    required_skills: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_memory_references(self) -> "BenchmarkCase":
        known = {memory.memory_key for memory in self.memories}
        missing = set(self.relevant_memory_keys) - known
        if missing:
            raise ValueError(f"unknown relevant memory keys: {sorted(missing)}")
        return self


class BenchmarkSuite(BaseModel):
    version: str
    cases: list[BenchmarkCase]

    @model_validator(mode="after")
    def validate_unique_case_ids(self) -> "BenchmarkSuite":
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("benchmark case_id values must be unique")
        return self
