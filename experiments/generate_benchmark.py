"""Generate the checked-in AgentOS benchmark from its balanced profile matrix."""

from __future__ import annotations

import json
from pathlib import Path


PROFILES = {
    "learning": [
        ("space", "gravity"),
        ("ocean", "pronunciation"),
        ("sports", "spelling"),
        ("music", "vocabulary"),
        ("adventure", "grammar"),
        ("science", "orbit"),
    ],
    "family_education": [
        ("story", "fraction"),
        ("visual", "geometry"),
        ("game", "multiplication"),
        ("stepwise", "word_problem"),
        ("encouraging", "division"),
        ("hands_on", "measurement"),
    ],
    "home_service": [
        ("gentle", "warm_water"),
        ("quiet", "medicine_reminder"),
        ("concise", "find_keys"),
        ("home", "bedtime_light"),
        ("voice", "door_check"),
        ("minimal", "room_temperature"),
    ],
}

LEARNING_SKILLS = [
    "retrieve_learning_history",
    "generate_quiz",
    "ask_question",
    "evaluate_answer",
    "update_memory",
]
HOME_SKILLS = [
    "retrieve_home_context",
    "prepare_home_assistance",
    "notify_user",
    "verify_home_task",
    "update_memory",
]


def build_cases() -> list[dict]:
    cases: list[dict] = []
    for group, profiles in PROFILES.items():
        for index, (style, item) in enumerate(profiles, start=1):
            user_id = f"{group[:4]}{index:02d}"
            memories = [
                {
                    "memory_key": f"{user_id}.preference",
                    "user_id": user_id,
                    "memory_type": "preference",
                    "content": f"用户偏好 {style} 交互风格",
                    "source": "benchmark-v1",
                    "importance": 0.9,
                    "confidence": 0.95,
                    "tags": [f"domain:{group}", f"style:{style}", style],
                },
                {
                    "memory_key": f"{user_id}.feedback",
                    "user_id": user_id,
                    "memory_type": "feedback",
                    "content": f"历史反馈要求继续处理 {item}",
                    "source": "benchmark-v1",
                    "importance": 0.85,
                    "confidence": 0.9,
                    "tags": [f"domain:{group}", f"item:{item}", item],
                },
            ]
            relevant = [memory["memory_key"] for memory in memories]
            if group == "home_service":
                memories.append(
                    {
                        "memory_key": f"{user_id}.scene",
                        "user_id": user_id,
                        "memory_type": "scene",
                        "content": f"家庭场景线索与 {item} 任务相关",
                        "source": "benchmark-v1",
                        "importance": 0.75,
                        "confidence": 0.85,
                        "tags": [f"domain:{group}", f"item:{item}", "scene"],
                    }
                )
                relevant.append(memories[-1]["memory_key"])
            goals = {
                "full": f"请用 {style} 风格完成 {item} 任务",
                "style_only": f"请保持 {style} 风格继续之前的任务",
                "item_only": f"请继续处理 {item}，沿用之前的交互方式",
                "implicit": "请按照我的习惯继续昨天未完成的任务",
            }
            required_skills = HOME_SKILLS if group == "home_service" else LEARNING_SKILLS
            for explicitness, goal in goals.items():
                cases.append(
                    {
                        "case_id": f"{group}-{index:02d}-{explicitness}",
                        "scenario_group": group,
                        "user_id": user_id,
                        "goal": goal,
                        "explicitness": explicitness,
                        "memories": memories,
                        "relevant_memory_keys": relevant,
                        "expected_style": style,
                        "expected_items": [item],
                        "required_skills": required_skills,
                    }
                )
    return cases


def main() -> None:
    output = Path("experiments/data/agentos-benchmark-v1.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"version": "1.0", "cases": build_cases()}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
