from rag.chat.context import build_context, render_context, source_line
from rag.collect.schemas import ManifestRecord, Source
from rag.parse.schemas import Chunk
from rag.retrieve.schemas import Hit


def hit(record_id: str, n: int, text: str, **kw) -> Hit:
    chunk = Chunk(
        chunk_id=f"{record_id}#{n:04d}",
        record_id=record_id,
        element_type="paragraph",
        header="Header Title > Sec",
        text=text,
        token_count=3,
        header_token_count=2,
        parser="grobid",
        **kw,
    )
    return Hit(chunk=chunk, rank=n + 1, score=0.03)


RECORDS = {"p:a": ManifestRecord(record_id="p:a", source=Source.ACL, title="BPE Paper", year=2016)}


def test_labels_follow_rank_order_and_titles_come_from_the_manifest() -> None:
    ctx = build_context(
        [
            hit("p:a", 0, "merges", section_path=["Method", "BPE"], section_number="3.2", page=3),
            hit("p:z", 1, "other"),  # not in the manifest: header title, no year
        ],
        RECORDS,
    )
    assert [c.label for c in ctx] == ["C1", "C2"]
    assert render_context(ctx) == (
        "[C1] BPE Paper (2016) > §3.2 BPE, p.3\nmerges\n\n[C2] Header Title > paragraph\nother"
    )
    assert source_line(ctx[0]) == "BPE Paper (2016), §3.2 BPE, p.3 · p:a#0000"
