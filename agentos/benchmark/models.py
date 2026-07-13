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


class CaseRunResult(BaseModel):
    run_key: str
    experiment: Literal["clean", "robustness", "feedback_loop", "deepseek"]
    case_id: str
    scenario_group: ScenarioGroup
    explicitness: Explicitness
    condition: str
    planner_name: str
    repeat_index: int = 0
    success: bool
    error: str | None = None
    memory_retrieval_precision: float
    memory_retrieval_recall: float
    preference_match: bool
    task_coverage: bool
    executable_plan: bool
    skill_chain_correct: bool
    conflict_selection_correct: bool
    latency_ms: float
    model: str | None = None
    system_fingerprint: str | None = None
    prompt_hash: str | None = None
    response_id: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class MetricSummary(BaseModel):
    planner_name: str
    condition: str
    scenario_group: str
    metric: str
    sample_count: int
    value: float
    standard_deviation: float
    ci_low: float
    ci_high: float


class SignificanceResult(BaseModel):
    baseline_planner: str
    comparison_planner: str
    metric: str
    sample_count: int
    baseline_only_successes: int
    comparison_only_successes: int
    p_value: float
    adjusted_p_value: float


class ExperimentManifest(BaseModel):
    dataset_version: str
    dataset_sha256: str
    git_commit: str
    python_version: str
    generated_at: str
    deepseek_model: str | None = None
    deepseek_repeats: int = 0
    system_fingerprints: list[str] = Field(default_factory=list)
    prompt_hashes: list[str] = Field(default_factory=list)


class BenchmarkExperimentResult(BaseModel):
    clean_runs: list[CaseRunResult]
    robustness_runs: list[CaseRunResult]
    feedback_runs: list[CaseRunResult]
    deepseek_runs: list[CaseRunResult] = Field(default_factory=list)
    statistics: list[MetricSummary]
    significance: list[SignificanceResult]
    manifest: ExperimentManifest | None = None
