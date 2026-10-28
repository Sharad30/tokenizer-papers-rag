"""What the Pick an embedding model page shows (D-038).

The models come from config (`index.dense_models`) with each index's state
read from its folder: built (and fresh), stale, or not built yet, plus its
build time, dimensions, size and truncations. The comparison table comes
from the latest committed results/<date>-embeddings-ablation.json. On a
fresh clone the indexes are gitignored, so the snapshot keeps the model
table.
"""

import json
from pathlib import Path
from typing import Literal

from rag.common.config import Config
from rag.common.io import file_sha256
from rag.common.schemas import StrictModel
from rag.eval.ablation import AblationResult
from rag.parse.pipeline import CHUNKS, step_dir
from rag.retrieve.retriever import IndexUnavailable, bm25_dir, dense_dir, read_meta
from rag.showcase.sources import Snapshot, read_snapshot, snapshot_file

STAGE = "embeddings"
RESULTS_GLOB = "*-embeddings-ablation.json"


class ModelInfo(StrictModel):
    name: str  # key in index.dense_models, or "bm25"
    model: str
    kind: str
    deployed: bool
    query_prompt: str
    status: Literal["built", "stale", "missing"]
    dims: int | None = None
    minutes: float | None = None
    size_mb: float | None = None
    truncated: int | None = None
    built_commit: str | None = None


class EmbeddingsData(StrictModel):
    origin: Literal["live", "snapshot"]
    snapshot: Snapshot | None = None
    models: list[ModelInfo]
    results: AblationResult | None = None
    results_file: str | None = None


def _info(name: str, folder: Path, chunks_sha: str | None, settings, **fields) -> ModelInfo:
    status, stats = "missing", {}
    try:
        meta = read_meta(folder)
    except IndexUnavailable:
        meta = None
    if meta is not None:
        fresh = meta.chunks_sha256 == chunks_sha and (settings is None or meta.settings == settings)
        status = "built" if fresh else "stale"
        stats_path = folder / "stats.json"
        stats = json.loads(stats_path.read_text())["counts"] if stats_path.is_file() else {}
        fields["built_commit"] = meta.git_commit[:8]
        fields["size_mb"] = round(
            sum(f.stat().st_size for f in folder.iterdir() if f.is_file()) / 1e6, 1
        )
    return ModelInfo(
        name=name,
        status=status,
        dims=stats.get("dimensions"),
        minutes=round(stats["seconds"] / 60, 1) if "seconds" in stats else None,
        truncated=stats.get("truncated"),
        **fields,
    )


def models(config: Config) -> list[ModelInfo]:
    chunks = step_dir(config, "chunk") / CHUNKS
    sha = file_sha256(chunks) if chunks.is_file() else None
    out = [
        _info(
            name,
            dense_dir(config, name),
            sha,
            settings.settings(),
            model=settings.model,
            kind=settings.kind,
            deployed=name == config.index.dense_model,
            query_prompt=settings.query_prompt,
        )
        for name, settings in config.index.dense_models.items()
    ]
    out.append(
        _info(
            "bm25",
            bm25_dir(config),
            sha,
            config.index.bm25.model_dump(mode="json"),
            model="bm25s (Lucene BM25)",
            kind="keyword",
            deployed=True,
            query_prompt="",
        )
    )
    return out


def latest_results(results_dir: Path) -> tuple[AblationResult | None, str | None]:
    files = sorted(results_dir.glob(RESULTS_GLOB))
    if not files:
        return None, None
    return AblationResult.model_validate_json(files[-1].read_text(encoding="utf-8")), files[-1].name


def load(config: Config) -> EmbeddingsData:
    results, name = latest_results(config.paths.results)
    live = models(config)
    if any(m.status != "missing" for m in live):
        return EmbeddingsData(origin="live", models=live, results=results, results_file=name)
    snapshot = read_snapshot(snapshot_file(config.paths.snapshot, STAGE))
    if snapshot is None:
        return EmbeddingsData(origin="live", models=live, results=results, results_file=name)
    return EmbeddingsData(
        origin="snapshot",
        snapshot=snapshot,
        models=[ModelInfo.model_validate(m) for m in snapshot.data["models"]],
        results=results,
        results_file=name,
    )


def source_files(config: Config) -> dict[str, Path]:
    files = {name: dense_dir(config, name) / "meta.json" for name in config.index.dense_models}
    files["bm25"] = bm25_dir(config) / "meta.json"
    return files


def snapshot_data(data: EmbeddingsData) -> dict:
    return {"models": [m.model_dump(mode="json") for m in data.models]}
