from rag.collect.schemas import ManifestRecord, Source
from rag.common.identity import UnionFind, id_links


def test_union_find_names_groups_by_smallest_item() -> None:
    uf = UnionFind()
    uf.union("c", "b")
    uf.union("b", "a")
    uf.find("z")
    assert uf.groups() == {"c": "a", "b": "a", "a": "a", "z": "z"}


def test_id_links_report_the_shared_key() -> None:
    records = [
        ManifestRecord(record_id="acl:x", source=Source.ACL, title="t", doi="10.1/A"),
        ManifestRecord(record_id="s2:y", source=Source.S2, title="t", doi="10.1/a"),
        ManifestRecord(record_id="arxiv:z", source=Source.ARXIV, title="t", derived_from="acl:x"),
    ]
    assert list(id_links(records)) == [
        ("acl:x", "s2:y", "doi"),
        ("acl:x", "arxiv:z", "derived_from"),
    ]
