"""Pydantic base models shared by every stage."""

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base for every schema in the project.

    `extra="forbid"` turns a misspelled key (in config.yaml or in a stage's
    output file) into a validation error instead of a silently ignored field.
    """

    model_config = ConfigDict(extra="forbid")
