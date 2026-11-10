"""Urdu text normalisation (kept in its own module so the pickled model can be loaded anywhere)."""
import re

# Arabic diacritics + tatweel
_DIACRITICS = re.compile(r"[ً-ٰٟـ]")
_NON_TEXT = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_urdu(text: str) -> str:
    """Light Urdu normalisation so spelling variants map to the same token."""
    text = str(text)
    text = _DIACRITICS.sub("", text)
    text = (
        text.replace("ي", "ی").replace("ك", "ک").replace("ە", "ہ")
        .replace("ۃ", "ہ").replace("ھ", "ہ")
    )
    text = _NON_TEXT.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip().lower()
