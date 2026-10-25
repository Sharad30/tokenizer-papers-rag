"""Token counting with the embedding model's own tokenizer (D-016).

A chunk limit only means something in the tokens of the model that will
embed the chunk: past its limit (512 for bge-small) the rest of the input is
silently dropped. Only the tokenizer is loaded, not the model.
"""

from typing import Protocol


class TokenCounter(Protocol):
    def count(self, text: str) -> int:
        """Tokens in text, without the model's special tokens ([CLS], [SEP])."""
        ...

    def split(self, text: str, max_tokens: int) -> list[str]:
        """Cut text into consecutive pieces of at most max_tokens tokens each."""
        ...


class HFTokenCounter:
    def __init__(self, model_name: str) -> None:
        from transformers import AutoTokenizer  # slow import; only when really counting

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def count(self, text: str) -> int:
        # verbose=False: counting a long text is the point, not a mistake to warn about.
        return len(self._tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"])

    def split(self, text: str, max_tokens: int) -> list[str]:
        # Offsets map each token back to its characters, so pieces are cut
        # from the original text (no detokenising, which loses case and spacing).
        offsets = self._tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True, verbose=False
        )["offset_mapping"]
        pieces = []
        for start in range(0, len(offsets), max_tokens):
            window = offsets[start : start + max_tokens]
            end = offsets[start + max_tokens][0] if start + max_tokens < len(offsets) else len(text)
            pieces.append(text[window[0][0] : end].strip())
        return [p for p in pieces if p]
