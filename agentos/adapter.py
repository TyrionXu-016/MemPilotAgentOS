from typing import Any


class SimulatedRobotAdapter:
    def emit(self, channel: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "channel": channel,
            "payload": payload,
            "delivered": True,
        }
