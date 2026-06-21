from pathlib import Path

import pytest

from agentos.storage import AgentOSStore


@pytest.fixture()
def store(tmp_path: Path) -> AgentOSStore:
    db_path = tmp_path / "agentos.sqlite"
    created = AgentOSStore(db_path)
    created.migrate()
    return created
