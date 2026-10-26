from rag.collect.schemas import ManifestRecord, Source
from rag.eval.identity import identity_groups


def rec(record_id: str, **ids) -> ManifestRecord:
    return ManifestRecord(record_id=record_id, source=Source.ARXIV, title="t", **ids)


def test_links_by_shared_ids_and_derivation() -> None:
    groups = identity_groups(
        [
            rec("arxiv:1508.07909v5", arxiv_id="1508.07909", s2_id="S1"),
            rec("arxiv:1508.07909v1", arxiv_id="1508.07909"),
            rec("acl:P16-1162", acl_id="P16-1162", s2_id="S1"),
            rec("acl:copy", derived_from="acl:P16-1162"),
            rec("arxiv:other", arxiv_id="9999.00001", doi="10.1/X"),
            rec("s2:other", doi="10.1/x"),  # DOIs compare case-insensitively
            rec("s2:alone"),
        ]
    )
    bpe = {
        groups[r] for r in ["arxiv:1508.07909v5", "arxiv:1508.07909v1", "acl:P16-1162", "acl:copy"]
    }
    assert bpe == {"acl:P16-1162"}  # smallest record id names the group
    assert groups["arxiv:other"] == groups["s2:other"]
    assert groups["s2:alone"] == "s2:alone"
