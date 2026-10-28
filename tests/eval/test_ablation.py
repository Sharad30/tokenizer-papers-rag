from rag.eval.ablation import read_queries


def test_queries_skip_comments_and_blank_lines(tmp_path) -> None:
    path = tmp_path / "q.txt"
    path.write_text("# header\nWhat is BPE?\n\n  How does WordPiece work?  \n# end\n")
    assert read_queries(path) == ["What is BPE?", "How does WordPiece work?"]


def test_shipped_ablation_conditions_are_valid(repo_config_path) -> None:
    from rag.common.config import load_config

    ablation = load_config(repo_config_path).eval.ablations["dedup"]
    names = [c.name for c in ablation.conditions]
    assert names[0] == "none" and not ablation.conditions[0].dedup.enabled
    assert len(set(names)) == len(names)
