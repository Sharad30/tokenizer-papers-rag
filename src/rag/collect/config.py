"""Config models for the collect stage (the `collect:` section of config.yaml)."""

from pydantic import Field

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


class CollectConfig(StrictModel):
    http: HttpConfig
