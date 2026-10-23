from pathlib import Path

import pytest

from rag.collect.config import HttpConfig
from rag.collect.http import PoliteClient


class FakeClock:
    """Stands in for time.monotonic/time.sleep: sleeping advances time instantly."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def http_config() -> HttpConfig:
    return HttpConfig(
        user_agent="test-agent/0.0",
        contact_email_env="TEST_CONTACT_EMAIL",
        timeout_s=5,
        max_attempts=3,
        backoff_base_s=2,
        default_min_interval_s=0.5,
        min_interval_s={"api.example.org": 3.0},
    )


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def make_client(http_config: HttpConfig, fake_clock: FakeClock, tmp_path: Path):
    """Build a PoliteClient on a fake clock with its cache in tmp_path."""

    def _make(**kwargs) -> PoliteClient:
        return PoliteClient(
            http_config,
            cache_dir=tmp_path / "http_cache",
            clock=fake_clock.clock,
            sleep=fake_clock.sleep,
            **kwargs,
        )

    return _make
