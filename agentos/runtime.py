from __future__ import annotations

from agentos.adapter import SimulatedRobotAdapter
from agentos.memory import MemoryManager
from agentos.models import ExecutionEvent, Plan
from agentos.skills import SkillRegistry


class AgentOSRuntime:
    def __init__(
        self,
        registry: SkillRegistry,
        memory: MemoryManager,
        adapter: SimulatedRobotAdapter | None = None,
    ):
        self.registry = registry
        self.memory = memory
        self.adapter = adapter or SimulatedRobotAdapter()

    def run(self, user_id: str, plan: Plan) -> list[ExecutionEvent]:
        self.registry.validate_plan(plan)
        events: list[ExecutionEvent] = []
        for step in plan.steps:
            try:
                output = self.registry.execute(step.skill, step.params)
                delivered = self.adapter.emit(channel=step.skill, payload=output)
                events.append(
                    ExecutionEvent(
                        skill=step.skill,
                        status="success",
                        output=delivered,
                        message="executed",
                    )
                )
                if step.skill == "update_memory":
                    self.memory.record_feedback(
                        user_id=user_id,
                        goal=plan.goal,
                        feedback=str(step.params["feedback"]),
                    )
            except Exception as exc:
                events.append(
                    ExecutionEvent(
                        skill=step.skill,
                        status="failed",
                        output={"error": str(exc)},
                        message="execution failed",
                    )
                )
                break
        return events
