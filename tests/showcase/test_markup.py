from rag.showcase.markup import escape, highlight


def test_escape_neutralises_markdown_and_latex() -> None:
    assert escape("costs $5 and *bold*") == r"costs \$5 and \*bold\*"
    assert escape("a_b [x](y)") == r"a\_b \[x\]\(y\)"


def test_highlight_marks_whole_words_case_insensitively() -> None:
    marked = highlight("BPE and bpe-dropout, not abpe.", ["bpe"])
    assert marked == r":orange-background[BPE] and :orange-background[bpe]\-dropout, not abpe."


def test_highlight_prefers_the_longest_term() -> None:
    marked = highlight("byte-pair encoding", ["byte-pair encoding", "encoding"])
    assert marked == r":orange-background[byte\-pair encoding]"


def test_highlight_without_terms_only_escapes() -> None:
    assert highlight("$x$", []) == r"\$x\$"
