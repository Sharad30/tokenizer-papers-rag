"""SPECTER2 runs in another process; these tests fake that process."""

import json
from pathlib import Path

import numpy as np
import pytest

from rag.index.config import DenseConfig, IndexConfig
from rag.retrieve.dense import DenseIndex, Specter2Encoder, make_encoder
from tests.retrieve.conftest import make_chunk

SPECTER2 = DenseConfig(
    kind="specter2",
    model="allenai/specter2_base",
    query_adapter="allenai/specter2_adhoc_query",
    document_adapter="allenai/specter2",
    query_prompt="",
    prepend_header=True,
    header_separator="[SEP]",
    max_length=512,
    batch_size=8,
    device="cpu",
    isolated_with=["adapters==1.3.0"],
)


class FakeWorker:
    """Records each call and writes vectors [len(text), 1] and a report, like the worker."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, packages: list[str], args: list[str]) -> None:
        opts = dict(zip(args[::2], args[1::2], strict=False))
        texts = [
            json.loads(line)["text"] for line in Path(opts["--input"]).read_text().splitlines()
        ]
        self.calls.append(
            {
                "packages": packages,
                "adapter": opts["--adapter"],
                "texts": texts,
                "one_at_a_time": "--one-at-a-time" in args,
            }
        )
        np.save(opts["--output"], np.array([[len(t), 1.0] for t in texts], dtype=np.float32))
        report = {"n": len(texts), "truncated": 1}
        if "--one-at-a-time" in args:
            report["encode_ms"] = [10.0] * len(texts)
        Path(opts["--report"]).write_text(json.dumps(report))


def test_documents_use_the_proximity_adapter_and_title_sep_text() -> None:
    worker = FakeWorker()
    encoder = Specter2Encoder(SPECTER2, run=worker)
    index = DenseIndex.build(
        [make_chunk("p:a", 0, "BPE merges", header="BPE > Method")], SPECTER2, encoder
    )
    (call,) = worker.calls
    assert call["adapter"] == "allenai/specter2" and not call["one_at_a_time"]
    assert call["texts"] == ["BPE > Method[SEP]BPE merges"]
    assert call["packages"] == ["adapters==1.3.0"]
    assert np.allclose(np.linalg.norm(index.embeddings, axis=1), 1.0)  # normalised for dot product
    assert encoder.truncated == 1


def test_queries_use_the_query_adapter_once_and_are_cached_with_timings() -> None:
    worker = FakeWorker()
    encoder = Specter2Encoder(SPECTER2, run=worker)
    encoder.encode_queries(["what is BPE", "what is BPE", "fertility"])
    encoder.encode_query("what is BPE")  # cached: no new process
    assert len(worker.calls) == 1
    assert worker.calls[0]["adapter"] == "allenai/specter2_adhoc_query"
    assert worker.calls[0]["texts"] == ["what is BPE", "fertility"]  # no prompt, deduplicated
    assert worker.calls[0]["one_at_a_time"]
    assert encoder.encode_ms == {"what is BPE": 10.0, "fertility": 10.0}


def test_make_encoder_picks_by_kind() -> None:
    assert isinstance(make_encoder(SPECTER2), Specter2Encoder)


def test_settings_leave_defaults_out_so_old_indexes_stay_valid() -> None:
    plain = SPECTER2.model_copy(
        update={
            "kind": "sentence_transformers",
            "header_separator": "\n",
            "query_adapter": None,
            "document_adapter": None,
            "max_length": None,
            "isolated_with": [],
        }
    )
    assert set(plain.settings()) == {
        "model",
        "query_prompt",
        "prepend_header",
        "batch_size",
        "device",
    }


def test_deployed_dense_model_must_be_one_of_the_models() -> None:
    bm25 = {
        "method": "lucene",
        "k1": 1.5,
        "b": 0.75,
        "include_header": True,
        "stopwords": "en",
        "split_compounds": True,
    }
    with pytest.raises(ValueError, match="not in dense_models"):
        IndexConfig(bm25=bm25, dense_model="nope", dense_models={"s2": SPECTER2})
    assert IndexConfig(bm25=bm25, dense_model="s2", dense_models={"s2": SPECTER2}).dense is SPECTER2
