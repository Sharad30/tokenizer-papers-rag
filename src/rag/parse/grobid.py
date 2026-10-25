"""Client for a local GROBID service: PDF in, TEI XML out.

GROBID runs in Docker (`make grobid-up`). It is local, so there is no rate
limit to respect, but it has a fixed pool of worker threads and answers 503
when they are all busy; the docs say to wait 5-10 s and resend.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import httpx

from rag.parse.config import GrobidConfig

logger = logging.getLogger(__name__)

FULLTEXT_PATH = "/api/processFulltextDocument"
ALIVE_PATH = "/api/isalive"
BUSY = 503
NO_CONTENT = 204


class GrobidOutcome(StrEnum):
    OK = "ok"
    NO_CONTENT = "no_content"  # 204: GROBID could not extract or structure anything
    ERROR = "error"  # 4xx/5xx other than busy, or retries exhausted


@dataclass(frozen=True)
class GrobidResult:
    outcome: GrobidOutcome
    tei: bytes | None = None
    detail: str | None = None


class GrobidClient:
    def __init__(
        self,
        config: GrobidConfig,
        http: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._http = http or httpx.Client(base_url=config.url, timeout=config.timeout_s)
        self._sleep = sleep

    def __enter__(self) -> "GrobidClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self._http.close()

    def is_alive(self) -> bool:
        try:
            return self._http.get(ALIVE_PATH).status_code == httpx.codes.OK
        except httpx.TransportError:
            return False

    def form_fields(self) -> dict[str, str | list[str]]:
        c = self._config
        return {
            "consolidateHeader": str(c.consolidate_header),
            "consolidateCitations": str(c.consolidate_citations),
            "includeRawCitations": "1",
            # <s> per sentence: sentence-level chunk splits and citation markers.
            "segmentSentences": "1",
            "teiCoordinates": c.tei_coordinates,  # a repeated form field
        }

    def process(self, pdf: Path) -> GrobidResult:
        """Send one PDF; retry while GROBID is busy or the connection fails."""
        detail = ""
        for attempt in range(1, self._config.max_attempts + 1):
            try:
                with pdf.open("rb") as handle:
                    response = self._http.post(
                        FULLTEXT_PATH,
                        data=self.form_fields(),
                        files={"input": (pdf.name, handle, "application/pdf")},
                    )
            except httpx.TransportError as exc:
                detail = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == httpx.codes.OK:
                    return GrobidResult(GrobidOutcome.OK, tei=response.content)
                if response.status_code == NO_CONTENT:
                    return GrobidResult(GrobidOutcome.NO_CONTENT, detail="grobid 204")
                if response.status_code != BUSY:
                    # A 500 on a malformed PDF repeats on every attempt: don't retry.
                    message = response.text[:200].strip()
                    return GrobidResult(
                        GrobidOutcome.ERROR, detail=f"grobid {response.status_code}: {message}"
                    )
                detail = "grobid 503: busy"
            if attempt < self._config.max_attempts:
                logger.warning(
                    "%s on %s (attempt %d/%d); retrying in %ss",
                    detail,
                    pdf.name,
                    attempt,
                    self._config.max_attempts,
                    self._config.busy_retry_s,
                )
                self._sleep(self._config.busy_retry_s)
        return GrobidResult(GrobidOutcome.ERROR, detail=f"{detail} (retries exhausted)")
