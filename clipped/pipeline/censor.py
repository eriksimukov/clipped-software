"""Maskiert Beleidigungen und auf Wunsch Flueche in Texten, die sichtbar im Clip landen.

Die Muster stehen nur hier, damit sie gezielt gepflegt werden koennen. Der Ton bleibt unveraendert;
Clips mit Beleidigungen im Ton werden deshalb in der Auswahl markiert und abgewertet.
"""

from __future__ import annotations

import re

SLURS = re.compile(
    r"\b(?:n+[i1]+g+(?:a+h?|e+r+|u+h)s?z?"
    r"|fags?|f+a+g+o+t+s?"
    r"|retards?|retarded"
    r"|trann(?:y|ies)"
    r"|kikes?|chinks?|spics?|wetbacks?|beaners?|coons?)\b",
    re.IGNORECASE,
)
PROFANITY = re.compile(
    r"\b(?:\w*fuck\w*|motherf\w+|shit\w*|bullshit|bitch\w*|asshole\w*|cunts?|dicks?|dickhead\w*"
    r"|puss(?:y|ies)|bastards?)\b",
    re.IGNORECASE,
)


def _mask(match: re.Match) -> str:
    word = match.group(0)
    return word[0] + "*" * (len(word) - 1)


def censor_text(text: str, level: str = "slurs") -> str:
    """Stufen: none laesst alles stehen, slurs maskiert Beleidigungen, all auch Flueche."""
    if level == "none":
        return text
    text = SLURS.sub(_mask, text)
    if level == "all":
        text = PROFANITY.sub(_mask, text)
    return text


def contains_slur(text: str) -> bool:
    return bool(SLURS.search(text))
