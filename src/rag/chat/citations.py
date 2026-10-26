"""Citations in the model's answer: parse, check, renumber (D-035).

The model is told to cite context labels like [C3] or [C1, C4]. Code then:
- keeps labels that were in the context and renumbers them [1], [2], ...
  in order of first use, so the Sources list reads top to bottom;
- removes labels that weren't in the context, and reports them: an invalid
  citation is a grounding failure, not a typo to fix quietly;
- counts sentences with no citation at all (Phase 8 material).
"""

import re
from dataclasses import dataclass, field

# "[C3]", "[C1, C4]", "[C1; C4]", case-insensitive. Adjacent "[C1][C4]" are two matches.
_CITE = re.compile(r"\[\s*([Cc]\d+(?:\s*[,;]\s*[Cc]\d+)*)\s*\]")
_LABEL = re.compile(r"[Cc]\d+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
MIN_CLAIM_WORDS = 4  # shorter "sentences" (list headers, "Yes.") aren't counted as uncited claims


@dataclass
class CitedAnswer:
    text: str  # with [1], [2] ... in place of [Cn]
    used: list[str] = field(default_factory=list)  # context labels in order of first use
    invalid: list[str] = field(default_factory=list)  # cited labels not in the context
    uncited_sentences: int = 0


def resolve(answer: str, valid: set[str]) -> CitedAnswer:
    result = CitedAnswer(text="")

    def replace(match: re.Match) -> str:
        numbers = []
        for label in (m.upper() for m in _LABEL.findall(match.group(1))):
            if label not in valid:
                if label not in result.invalid:
                    result.invalid.append(label)
                continue
            if label not in result.used:
                result.used.append(label)
            number = result.used.index(label) + 1
            if number not in numbers:
                numbers.append(number)
        return f"[{', '.join(map(str, numbers))}]" if numbers else ""

    text = _CITE.sub(replace, answer)
    # Removing an invalid citation can leave "word ." or double spaces behind.
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)
    result.text = re.sub(r"[ \t]{2,}", " ", text).strip()
    result.uncited_sentences = count_uncited(answer)
    return result


def count_uncited(answer: str) -> int:
    sentences = [s for line in answer.splitlines() for s in _SENTENCE_END.split(line)]
    return sum(1 for s in sentences if len(s.split()) >= MIN_CLAIM_WORDS and not _CITE.search(s))
