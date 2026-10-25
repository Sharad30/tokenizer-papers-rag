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
        return split_at_word_starts(text, [start for start, _ in offsets], max_tokens)


def split_at_word_starts(text: str, starts: list[int], max_tokens: int) -> list[str]:
    """Cut text into pieces of at most max_tokens tokens, only where a word begins.

    `starts` is each token's first character. Cutting inside a word ("token|izer")
    would put a space into the word when the pieces are rejoined, and change
    both the text and its token count. A single "word" longer than max_tokens
    (a URL, a run of symbols) is the one case cut mid-word.
    """
    pieces: list[str] = []
    first = 0  # index of the first token of the current piece
    while first < len(starts):
        end = min(first + max_tokens, len(starts))  # token index the next piece starts at
        if end < len(starts):
            word_starts = [
                i
                for i in range(first + 1, end + 1)
                # Several tokens can share one start character (BERT's normaliser
                # decomposes a Hangul syllable into jamo): cut only before the first.
                if starts[i] != starts[i - 1] and _begins_word(text, starts[i])
            ]
            end = word_starts[-1] if word_starts else end
        stop = starts[end] if end < len(starts) else len(text)
        pieces.append(text[starts[first] : stop].strip())
        first = end
    return [p for p in pieces if p]


def _begins_word(text: str, char: int) -> bool:
    return char == 0 or text[char - 1].isspace()
