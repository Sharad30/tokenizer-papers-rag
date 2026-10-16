"""Pydantic base models shared by every stage."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base for every schema in the project.

    `extra="forbid"` turns a misspelled key (in config.yaml or in a stage's
    output file) into a validation error instead of a silently ignored field.
    """

    model_config = ConfigDict(extra="forbid")


class StageFailure(StrictModel):
    """One item a stage could not process, kept so failures are counted, not hidden."""

    item_id: str
    reason: str


class StageStats(StrictModel):
    """Contents of the stats.json every stage writes next to its outputs."""

    stage: str
    started_at: datetime
    finished_at: datetime
    limit: int | None
    counts: dict[str, int]
    failures: list[StageFailure]
    git_commit: str
    config_hash: str
