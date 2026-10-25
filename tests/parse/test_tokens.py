import re

from rag.parse.tokens import HFTokenCounter


class FakeHFTokenizer:
    """Mimics a fast HF tokenizer's call: one token per word, with character offsets."""

    def __call__(self, text: str, add_special_tokens: bool, verbose: bool, **kwargs) -> dict:
        assert add_special_tokens is False  # counts must exclude [CLS] / [SEP]
        spans = [m.span() for m in re.finditer(r"\S+", text)]
        out = {"input_ids": list(range(len(spans)))}
        if kwargs.get("return_offsets_mapping"):
            out["offset_mapping"] = spans
        return out


def make_counter() -> HFTokenCounter:
    counter = HFTokenCounter.__new__(HFTokenCounter)  # skip loading a real model
    counter._tokenizer = FakeHFTokenizer()
    return counter


def test_count_excludes_special_tokens() -> None:
    assert make_counter().count("one two  three") == 3


def test_split_cuts_original_text_at_token_boundaries() -> None:
    text = "Byte-pair  Encoding merges\nfrequent pairs."
    assert make_counter().split(text, 2) == ["Byte-pair  Encoding", "merges\nfrequent", "pairs."]


def test_split_short_text_is_one_piece() -> None:
    assert make_counter().split("a b", 5) == ["a b"]
