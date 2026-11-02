"""A scripted conversation and its Markdown transcript."""

from pathlib import Path

from rag.chat import script
from tests.chat.test_engine import config, make_engine  # noqa: F401 (fixture)


def test_turns_file_skips_comments_and_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "turns.txt"
    path.write_text("# comment\nFirst question?\n\n  Second one?  \n", encoding="utf-8")
    assert script.read_turns(path) == ["First question?", "Second one?"]


def test_transcript_shows_rewrites_sources_and_abstentions(config) -> None:  # noqa: F811
    def reply(system, messages):
        text = messages[0].content
        if system == "Rewrite.":
            return "byte level models"  # the rewriter's standalone query
        return "NOT_SUPPORTED" if "byte level" in text.split("Question:")[-1] else "Merges [C1]."

    engine, _ = make_engine(config, reply)
    results = script.run_script(engine, ["merge operations", "and the other one?"])
    text = script.markdown(results, ["header line"])
    assert "## Turn 1" in text and "**You:** merge operations" in text
    assert "Merges [1]." in text and "- [1] BPE Paper (2016)" in text
    assert "*Rewritten for retrieval:* byte level models" in text
    assert "the model found the passages don't support an answer" in text
