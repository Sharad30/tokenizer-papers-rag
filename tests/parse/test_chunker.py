from rag.parse.chunker import SPECIAL_TOKENS, Chunker
from rag.parse.schemas import (
    ElementType,
    Figure,
    Paragraph,
    Parser,
    Section,
    Sentence,
    StructuredPaper,
    Table,
)


def words(n: int, tag: str) -> str:
    """A sentence of exactly n tokens (whitespace counter), ending with a full stop."""
    return " ".join(f"{tag}{i}" for i in range(n - 1)) + f" {tag}end."


def para(*sentences: str, page: int | None = 1, refs: list[str] | None = None) -> Paragraph:
    return Paragraph(
        sentences=[Sentence(text=s, cited_refs=refs or []) for s in sentences], page=page
    )


def section(i: int, title: str, *paragraphs: Paragraph, number: str | None = None) -> Section:
    return Section(
        id=f"s{i}", number=number, title=title, path=[title], level=1, paragraphs=list(paragraphs)
    )


def paper(**kwargs) -> StructuredPaper:
    return StructuredPaper(record_id="arxiv:1v1", parser=Parser.GROBID, title="T", **kwargs)


def body_chunks(chunks):
    return [c for c in chunks if c.element_type is ElementType.PARAGRAPH]


def test_short_paragraphs_are_packed_up_to_target(chunk_config, counter) -> None:
    p = paper(
        sections=[
            section(0, "Intro", para(words(8, "a")), para(words(8, "b")), para(words(8, "c")))
        ]
    )
    chunks = Chunker(chunk_config, counter).chunk(p).chunks
    # 8 + 8 = 16 <= 20; another 8 would pass the target, so a new chunk starts.
    # No overlap: the last sentence (8 tokens) is over the 5-token budget.
    assert [c.token_count for c in chunks] == [16, 8]
    assert "a0" in chunks[0].text and "b0" in chunks[0].text
    assert chunks[0].text.count("\n\n") == 1  # paragraphs rejoined with a blank line


def test_overlap_is_whole_sentences_within_budget(chunk_config, counter) -> None:
    p = paper(
        sections=[section(0, "Intro", para(words(12, "a"), words(4, "b")), para(words(15, "c")))]
    )
    first, second = Chunker(chunk_config, counter).chunk(p).chunks
    # The last sentence of chunk 1 (4 tokens <= overlap budget 5) opens chunk 2.
    assert second.text.startswith(words(4, "b"))
    assert second.token_count == 4 + 15
    assert first.token_count == 16


def test_no_chunk_crosses_a_section_and_overlap_stays_inside(chunk_config, counter) -> None:
    p = paper(
        sections=[
            section(0, "Intro", para(words(10, "a"), words(3, "x"))),
            section(1, "Method", para(words(10, "b"))),
        ]
    )
    chunks = body_chunks(Chunker(chunk_config, counter).chunk(p).chunks)
    assert [c.section_path for c in chunks] == [["Intro"], ["Method"]]
    assert "x0" not in chunks[1].text  # no overlap carried into the next section


def test_long_paragraph_is_split_on_sentences(chunk_config, counter) -> None:
    long_para = para(*(words(9, f"s{i}") for i in range(6)))  # 54 tokens, over max 30
    chunks = (
        Chunker(chunk_config, counter).chunk(paper(sections=[section(0, "S", long_para)])).chunks
    )
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= chunk_config.max_tokens
        assert chunk.text.endswith(".")  # never cut mid-sentence


def test_over_long_sentence_is_split_on_tokens(chunk_config, counter) -> None:
    result = Chunker(chunk_config, counter).chunk(
        paper(sections=[section(0, "S", para(words(70, "w")))])
    )
    assert result.split_sentences == 1
    assert all(c.token_count <= chunk_config.max_tokens for c in result.chunks)
    assert sum(c.token_count for c in result.chunks) >= 70


def test_every_chunk_fits_the_model_with_its_header(chunk_config, counter) -> None:
    long_title = " ".join(["Title"] * 20)
    p = StructuredPaper(
        record_id="r",
        parser=Parser.GROBID,
        title=long_title,
        abstract=[para(words(25, "abs"))],
        sections=[section(0, "Intro", para(words(29, "a")), para(words(29, "b")))],
    )
    chunks = Chunker(chunk_config, counter).chunk(p).chunks
    for c in chunks:
        assert (
            c.header_token_count + c.token_count + SPECIAL_TOKENS
            <= chunk_config.embedding_max_tokens
        )
    assert chunks[0].header_token_count == 8  # capped at 40 - 2 - 30


def test_abstract_is_its_own_chunk(chunk_config, counter) -> None:
    p = paper(abstract=[para(words(5, "abs"))], sections=[section(0, "Intro", para(words(5, "a")))])
    first, second = Chunker(chunk_config, counter).chunk(p).chunks
    assert (first.element_type, first.section_path, first.header) == (
        ElementType.ABSTRACT,
        ["Abstract"],
        "T > Abstract",
    )
    assert "a0" not in first.text and second.element_type is ElementType.PARAGRAPH


def test_header_number_page_and_cited_refs(chunk_config, counter) -> None:
    p = paper(
        sections=[section(0, "Method", para(words(5, "a"), page=3, refs=["b2", "b0"]), number="2")]
    )
    [chunk] = Chunker(chunk_config, counter).chunk(p, arxiv_id="1", arxiv_version=1).chunks
    assert (chunk.header, chunk.section_number, chunk.page, chunk.cited_refs) == (
        "T > Method",
        "2",
        3,
        ["b2", "b0"],
    )
    assert (chunk.chunk_id, chunk.arxiv_id) == ("arxiv:1v1#0000", "1")


def test_long_formula_becomes_placeholder(chunk_config, counter) -> None:
    short = Paragraph(sentences=[Sentence(text="f(x) = x")], formula_id="formula_0")
    long = Paragraph(
        sentences=[Sentence(text="g(x) = " + "x + " * 10 + "1")], formula_id="formula_1"
    )
    result = Chunker(chunk_config, counter).chunk(paper(sections=[section(0, "S", short, long)]))
    [chunk] = result.chunks
    assert "f(x) = x" in chunk.text and "[equation]" in chunk.text and "g(x)" not in chunk.text
    assert result.formula_placeholders == 1


def test_table_chunks_repeat_caption_and_header(chunk_config, counter) -> None:
    table = Table(
        id="tab_0",
        label="Table 1",
        caption="Sizes.",
        rows=[["Model", "Vocab"], ["A", "1"], ["B", "2"], ["C", "3"]],
        page=4,
    )
    chunks = Chunker(chunk_config, counter).chunk(paper(tables=[table])).chunks
    assert [c.element_type for c in chunks] == [
        ElementType.TABLE,
        ElementType.TABLE,
    ]  # 2 rows per chunk
    for c in chunks:
        assert c.text.startswith("Table 1: Sizes.\n\n| Model | Vocab |")
        assert c.page == 4 and c.header == "T > Table 1"
    assert "| C | 3 |" in chunks[1].text


def test_table_without_rows_is_caption_only(chunk_config, counter) -> None:
    table = Table(id="tab_0", label="Table 2", caption="Results.")
    [chunk] = Chunker(chunk_config, counter).chunk(paper(tables=[table])).chunks
    assert chunk.text == "Table 2: Results."


def test_figure_captions_are_own_chunks_or_dropped(chunk_config, counter) -> None:
    p = paper(figures=[Figure(id="fig_0", label="Figure 1", caption="Merges.", page=2)])
    [chunk] = Chunker(chunk_config, counter).chunk(p).chunks
    assert (chunk.element_type, chunk.text) == (ElementType.FIGURE_CAPTION, "Figure 1: Merges.")
    dropped = chunk_config.model_copy(update={"captions": "drop"})
    assert Chunker(dropped, counter).chunk(p).chunks == []


def test_chunk_ids_are_stable(chunk_config, counter) -> None:
    p = paper(
        abstract=[para(words(5, "x"))],
        sections=[section(0, "S", para(words(25, "a")), para(words(25, "b")))],
    )
    first = Chunker(chunk_config, counter).chunk(p).chunks
    again = Chunker(chunk_config, counter).chunk(p).chunks
    assert first == again
    assert [c.chunk_id for c in first] == [f"arxiv:1v1#{i:04d}" for i in range(len(first))]
