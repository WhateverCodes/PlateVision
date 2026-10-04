"""Bounded model-supported Indian plate decoding, not registration verification.

Historical prefixes are retained. Unknown/special formats fall back unchanged.
No label lookups, inferred missing glyphs, or confidence boosting are used.
"""
import math
import re

STATE_CODES = frozenset('AN AP AR AS BR CG CH DD DL DN GA GJ HP HR JH JK KA KL LA LD MH ML MN MP MZ NL OD OR PB PY RJ SK TN TR TS TG UK UP UT WB'.split())
CONFUSIONS = ('0OQ', '1IL', '2Z', '5S', '6G', '8B')


def supported(text):
    """A reading prior covering common, no-series and BH layouts."""
    return bool(re.fullmatch(r'\d{2}BH\d{4}[A-Z]{1,2}', text) or
                (text[:2] in STATE_CODES and
                 re.fullmatch(r'[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{1,4}', text)))


def decode(options, max_changes=2):
    """Return a supported alternative and its original per-token probabilities.

    options contains one character->probability mapping per emitted glyph.
    A supported greedy reading is never rewritten to a different valid plate.
    Alternatives must have >=2% probability and >=5% of the best token score.
    Only visual confusion groups may change outside the two-letter state prefix.
    """
    if not options or any(not row for row in options):
        return None
    raw = ''.join(max(row, key=row.get) for row in options)
    if supported(raw):
        return None
    n = len(options)
    layouts = []
    for district in (1, 2):
        for series in range(4):
            digits = n - 2 - district - series
            if 1 <= digits <= 4:
                layouts.append('AA' + 'D'*district + 'A'*series + 'D'*digits)
    for letters in (1, 2):
        if n == 8 + letters:
            layouts.append('DDBHDDDD' + 'A'*letters)
    choices = {}
    for layout in layouts:
        beams = [('', [], 0., 0)]
        for pos, kind in enumerate(layout):
            top = raw[pos]; top_score = options[pos][top]
            allowed = []
            for char, score in options[pos].items():
                if not ((kind == 'A' and char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ') or
                        (kind == 'D' and char in '0123456789') or
                        (kind in 'BH' and char == kind)):
                    continue
                if char != top and (score < .02 or score < top_score*.05):
                    continue
                if char != top and pos >= 2 and not any(top in g and char in g for g in CONFUSIONS):
                    continue
                allowed.append((char, score))
            expanded = []
            for text, scores, quality, changes in beams:
                for char, score in allowed:
                    count = changes + (char != top)
                    value = text + char
                    if count > max_changes:
                        continue
                    if pos == 1 and layout.startswith('AA') and value not in STATE_CODES:
                        continue
                    expanded.append((value, scores+[score], quality+math.log(max(score, 1e-8)), count))
            beams = sorted(expanded, key=lambda v: v[2], reverse=True)[:24]
            if not beams:
                break
        for text, scores, quality, _ in beams:
            if len(text) == n and supported(text):
                choices[text] = (text, scores, quality)
    return max(choices.values(), key=lambda v: v[2])[:2] if choices else None
