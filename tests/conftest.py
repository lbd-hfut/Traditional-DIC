from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def f0_data_root(repository_root: Path) -> Path:
    return repository_root / "tests" / "data" / "f0"
