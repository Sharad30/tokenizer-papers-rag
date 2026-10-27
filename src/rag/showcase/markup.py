"""Text for Streamlit markdown, built without importing streamlit."""

import re

# Characters Streamlit's markdown would read as formatting: $ starts LaTeX,
# * and _ emphasis, ` code, [ ] links, ~ strikethrough, : colour directives.
_SPECIAL = re.compile(r"([\\`*_{}\[\]<>()#+\-!|$~:])")


def escape(text: str) -> str:
    return _SPECIAL.sub(r"\\\1", text)


def highlight(text: str, terms: list[str], colour: str = "orange") -> str:
    """Escape text and mark each whole-word, case-insensitive match of a term.

    Uses the same word-boundary rule as the keyword rules, so what's marked
    is exactly what matched.
    """
    if not terms:
        return escape(text)
    alternatives = sorted((re.escape(t) for t in terms), key=len, reverse=True)
    pattern = re.compile(r"\b(?:" + "|".join(alternatives) + r")\b", re.IGNORECASE)
    parts, last = [], 0
    for match in pattern.finditer(text):
        parts.append(escape(text[last : match.start()]))
        parts.append(f":{colour}-background[{escape(match.group(0))}]")
        last = match.end()
    parts.append(escape(text[last:]))
    return "".join(parts)
