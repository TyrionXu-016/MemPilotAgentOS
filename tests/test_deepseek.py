import json

import pytest

from agentos.deepseek import (
    DeepSeekConfigurationError,
    DeepSeekOutputError,
    DeepSeekPlanGenerator,
    DeepSeekRawResponse,
)
from agentos.deepseek_planner import DeepSeekPlanner
from agentos.skills import build_default_registry


def _valid_content():
    return json.dumps(
        {
            "scenario_group": "learning",
            "interaction_style": "space",
            "task_items": ["gravity"],
            "memory_ids": [],
            "steps": [
                {"skill": "retrieve_learning_history", "params": {"user_id": "u001"}},
                {"skill": "generate_quiz", "params": {"theme": "space", "words": ["gravity"]}},
                {"skill": "ask_question", "params": {"mode": "interactive"}},
                {"skill": "evaluate_answer", "params": {"expected_words": ["gravity"]}},
                {"skill": "update_memory", "params": {"feedback": "reviewed gravity"}},
            ],
        }
    )


def _response(content):
    return DeepSeekRawResponse(
        content=content,
        response_id="response-1",
        model="deepseek-v4-flash",
        system_fingerprint="fp-test",
        prompt_tokens=100,
        completion_tokens=50,
    )


def test_deepseek_planner_builds_valid_plan_from_json():
    generator = DeepSeekPlanGenerator(request=lambda payload: _response(_valid_content()))
    registry = build_default_registry()
    planner = DeepSeekPlanner(registry, generator, planner_name="deepseek_no_memory")

    plan = planner.plan("u001", "请用 space 风格复习 gravity")

    assert plan.planner_name == "deepseek_no_memory"
    assert plan.context.interaction_style == "space"
    assert plan.context.task_items == ["gravity"]
    assert plan.generation_metadata["system_fingerprint"] == "fp-test"
    registry.validate_plan(plan)


def test_deepseek_planner_rejects_unavailable_memory_ids():
    content = json.loads(_valid_content())
    content["memory_ids"] = [999]
    generator = DeepSeekPlanGenerator(request=lambda payload: _response(json.dumps(content)))
    planner = DeepSeekPlanner(
        build_default_registry(), generator, planner_name="deepseek_no_memory"
    )

    with pytest.raises(ValueError, match="unavailable memory IDs"):
        planner.plan("u001", "复习 gravity")


def test_deepseek_generator_retries_empty_content_and_retryable_errors():
    calls = []

    class RateLimitError(Exception):
        status_code = 429

    responses = [RateLimitError("limited"), _response(""), _response(_valid_content())]

    def request(payload):
        calls.append(payload)
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    generator = DeepSeekPlanGenerator(
        request=request,
        max_attempts=3,
        sleep=lambda _: None,
    )

    result = generator.generate("u001", "复习 gravity", [], [])

    assert result.draft.task_items == ["gravity"]
    assert len(calls) == 3
    assert calls[0]["model"] == "deepseek-v4-flash"
    assert calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}


def test_deepseek_generator_retries_server_errors():
    class ServerError(Exception):
        status_code = 503

    responses = [ServerError("unavailable"), _response(_valid_content())]

    def request(payload):
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    generator = DeepSeekPlanGenerator(request=request, sleep=lambda _: None)

    assert generator.generate("u001", "复习", [], []).draft.task_items == ["gravity"]


def test_deepseek_generator_does_not_repair_invalid_json():
    generator = DeepSeekPlanGenerator(request=lambda payload: _response("not-json"))

    with pytest.raises(DeepSeekOutputError):
        generator.generate("u001", "复习", [], [])


def test_deepseek_generator_rejects_valid_json_with_wrong_schema():
    generator = DeepSeekPlanGenerator(request=lambda payload: _response('{"answer": "gravity"}'))

    with pytest.raises(DeepSeekOutputError):
        generator.generate("u001", "复习", [], [])


def test_deepseek_generator_requires_api_key_without_injected_client(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    generator = DeepSeekPlanGenerator()

    with pytest.raises(DeepSeekConfigurationError, match="DEEPSEEK_API_KEY"):
        generator.generate("u001", "复习", [], [])
