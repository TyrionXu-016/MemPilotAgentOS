# AgentOS Prototype Architecture

The prototype implements a closed loop for embodied AgentOS research:

1. `MemoryManager` stores user, preference, task, scene, and feedback memories.
2. `MemoryRetriever` selects memories relevant to the current goal.
3. `LayeredMemoryPlanner` converts selected memories into an executable skill plan.
4. `SkillRegistry` validates every plan against available skill contracts.
5. `AgentOSRuntime` executes skills through `SimulatedRobotAdapter`.
6. Runtime feedback is written back as long-term memory.
7. `evaluation.py` compares no-memory, preference-only, feedback-only, and layered-memory planners.

The first experiments use learning-companion, family-education, and home-service embodied scenarios because they verify personalization, repeated task continuity, skill execution, robustness under memory noise, and memory update without physical robot hardware.
