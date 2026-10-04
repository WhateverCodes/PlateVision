"""Conservative text cleanup and configurable Indian format plausibility."""
import re
from difflib import SequenceMatcher

DEFAULT_PATTERNS = (r"[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}", r"\d{2}BH\d{4}[A-Z]{1,2}")

def normalize(text: str) -> str:
    return re.sub(r"[\s\-]", "", text.upper())

def plausible(text: str, patterns: tuple[str, ...] = DEFAULT_PATTERNS) -> bool:
    return any(re.fullmatch(pattern, normalize(text)) for pattern in patterns)

def plausible_ocr(text: str) -> bool:
    # Also support the no-series-letter layout in the supplied HP 88 5801
    # photograph. This is a shape check, not official registration verification.
    return plausible(text) or bool(re.fullmatch(r"[A-Z]{2}\d{6}",normalize(text)))

def same_plate(left: str, right: str, threshold: float = 1.0) -> bool:
    a, b = normalize(left), normalize(right)
    if a == b:
        return True
    if threshold >= 1 or len(a) != len(b) or a[:4] != b[:4]:
        return False
    # Do not merge plates whose numeric portion differs; that is a common real distinction.
    if "".join(filter(str.isdigit, a)) != "".join(filter(str.isdigit, b)):
        return False
    return SequenceMatcher(None, a, b).ratio() >= threshold
