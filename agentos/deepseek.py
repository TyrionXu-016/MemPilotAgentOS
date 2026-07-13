from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class DeepSeekConfigurationError(RuntimeError):
    pass


class DeepSeekOutputError(RuntimeError):
    pass


class DeepSeekRawResponse(BaseModel):
    content: str
    response_id: str
    model: str
    system_fingerprint: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class DeepSeekStepDraft(BaseModel):
    skill: str
    params: dict[str, Any] = Field(default_factory=dict)


class DeepSeekPlanDraft(BaseModel):
    scenario_group: str
    interaction_style: str
    task_items: list[str] = Field(min_length=1)
    memory_ids: list[int] = Field(default_factory=list)
    steps: list[DeepSeekStepDraft] = Field(min_length=1)


class DeepSeekGeneration(BaseModel):
    draft: DeepSeekPlanDraft
    response_id: str
    model: str
    system_fingerprint: str | None
    prompt_hash: str
    prompt_tokens: int
    completion_tokens: int


RequestFunction = Callable[[dict[str, Any]], DeepSeekRawResponse]


class DeepSeekPlanGenerator:
    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        request: RequestFunction | None = None,
        max_attempts: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.model = model
        self.base_url = base_url
        self.request = request
        self.max_attempts = max_attempts
        self.sleep = sleep

    def generate(
        self,
        user_id: str,
        goal: str,
        skills: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> DeepSeekGeneration:
        payload, prompt_hash = self._build_payload(user_id, goal, skills, evidence)
        for attempt in range(1, self.max_attempts + 1):
            try:
                raw = (self.request or self._request_official)(payload)
                if not raw.content.strip():
                    raise _EmptyResponse("DeepSeek returned empty content")
                try:
                    draft = DeepSeekPlanDraft.model_validate(json.loads(raw.content))
                except (json.JSONDecodeError, ValidationError) as exc:
                    raise DeepSeekOutputError(f"invalid DeepSeek plan JSON: {exc}") from exc
                return DeepSeekGeneration(
                    draft=draft,
                    response_id=raw.response_id,
                    model=raw.model,
                    system_fingerprint=raw.system_fingerprint,
                    prompt_hash=prompt_hash,
                    prompt_tokens=raw.prompt_tokens,
                    completion_tokens=raw.completion_tokens,
                )
            except DeepSeekOutputError:
                raise
            except Exception as exc:
                if attempt >= self.max_attempts or not _is_retryable(exc):
                    raise
                self.sleep(float(2 ** (attempt - 1)))
        raise RuntimeError("unreachable")

    def _build_payload(
        self,
        user_id: str,
        goal: str,
        skills: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], str]:
        example = {
            "scenario_group": "learning",
            "interaction_style": "space",
            "task_items": ["gravity"],
            "memory_ids": [1],
            "steps": [
                {"skill": "retrieve_learning_history", "params": {"user_id": "u001"}},
                {"skill": "generate_quiz", "params": {"theme": "space", "words": ["gravity"]}},
                {"skill": "ask_question", "params": {"mode": "interactive"}},
                {"skill": "evaluate_answer", "params": {"expected_words": ["gravity"]}},
                {"skill": "update_memory", "params": {"feedback": "reviewed gravity"}},
            ],
        }
        system = (
            "You are an AgentOS task planner. Return one JSON object only. "
            "Use only the provided skills, include every required parameter, and do not invent memory IDs. "
            f"Example JSON: {json.dumps(example, ensure_ascii=False)}"
        )
        user = json.dumps(
            {"user_id": user_id, "goal": goal, "skills": skills, "memories": evidence},
            ensure_ascii=False,
            sort_keys=True,
        )
        prompt_hash = hashlib.sha256(f"{system}\n{user}".encode("utf-8")).hexdigest()
        return (
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0,
                "max_tokens": 2048,
                "stream": False,
                "extra_body": {"thinking": {"type": "disabled"}},
            },
            prompt_hash,
        )

    def _request_official(self, payload: dict[str, Any]) -> DeepSeekRawResponse:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise DeepSeekConfigurationError("DEEPSEEK_API_KEY is required")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise DeepSeekConfigurationError("install the llm extra with: pip install -e '.[llm]'") from exc
        client = OpenAI(api_key=api_key, base_url=self.base_url)
        response = client.chat.completions.create(**payload)
        usage = response.usage
        return DeepSeekRawResponse(
            content=response.choices[0].message.content or "",
            response_id=response.id,
            model=response.model,
            system_fingerprint=getattr(response, "system_fingerprint", None),
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
        )


class _EmptyResponse(RuntimeError):
    pass


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, _EmptyResponse):
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code == 429 or isinstance(status_code, int) and status_code >= 500:
        return True
    name = type(exc).__name__.lower()
    return "timeout" in name or "connection" in name
