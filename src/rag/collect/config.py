"""Config models for the collect stage (the `collect:` section of config.yaml)."""

from typing import Literal

from pydantic import Field, field_validator

from rag.common.schemas import StrictModel


class HttpConfig(StrictModel):
    user_agent: str
    # Name of the env var holding a contact email for the User-Agent, so no
    # address is committed. Sources ask for contact details from crawlers.
    contact_email_env: str
    timeout_s: float = Field(gt=0)
    max_attempts: int = Field(ge=1)
    # Wait before retry n (1-based) is backoff_base_s * 2**(n-1), unless the
    # server sends a Retry-After header.
    backoff_base_s: float = Field(ge=0)
    # Minimum seconds between requests to the same host. Hosts not listed get
    # the default (e.g. open-access PDF links on arbitrary publisher sites).
    default_min_interval_s: float = Field(ge=0)
    min_interval_s: dict[str, float]


class ArxivQuery(StrictModel):
    name: str  # recorded in each record's found_via, e.g. "arxiv_query:bpe"
    query: str  # arXiv search_query syntax, without the category and date filters


class ArxivConfig(StrictModel):
    categories: list[str]
    # arXiv API limits: slices of at most 2000, at most 30000 results per query.
    page_size: int = Field(ge=1, le=2000)
    max_results_per_query: int = Field(ge=1, le=30000)
    queries: list[ArxivQuery] = Field(min_length=1)

    @field_validator("queries")
    @classmethod
    def _unique_names(cls, queries: list[ArxivQuery]) -> list[ArxivQuery]:
        names = [q.name for q in queries]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate query names: {duplicates}")
        return queries


class S2Config(StrictModel):
    # Optional: without a key, requests share Semantic Scholar's public pool.
    api_key_env: str
    batch_size: int = Field(ge=1, le=500)  # API maximum: 500 ids per batch call
    page_size: int = Field(ge=1, le=1000)  # API maximum: 1000 per citations page
    max_items_per_anchor: int = Field(ge=1)
    directions: list[Literal["references", "citations"]] = Field(min_length=1)
    # Seed papers for the one-hop snowball, as S2 lookup ids ("ARXIV:1508.07909").
    anchors: list[str] = Field(min_length=1)


class AclConfig(StrictModel):
    # The scan clones the Anthology data repo (~120 MB) into paths.raw.
    enabled: bool


class CollectConfig(StrictModel):
    http: HttpConfig
    arxiv: ArxivConfig
    s2: S2Config
    acl: AclConfig
