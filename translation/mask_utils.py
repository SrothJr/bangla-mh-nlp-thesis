"""
Masking/unmasking logic for protected terms (medicines, acronyms, Reddit slang, etc.).
These terms are replaced with placeholder tokens (⟦T1⟧, ⟦T2⟧, ...) before passing
text to any model, then restored after generation, ensuring they are never
mistranslated or altered.
"""

import re
from protected_terms import ALL_PROTECTED

# Pattern for placeholders like ⟦T1⟧, ⟦T2⟧, etc.
PLACEHOLDER_PATTERN = re.compile(r"⟦T(\d+)⟧")


def mask_text(text):
    """
    Replaces protected terms in text with placeholder tokens like ⟦T1⟧.
    Returns (masked_text, lookup_dict) where lookup_dict maps placeholder
    numbers back to the original protected terms.
    """
    # Make a copy we can modify
    current = text
    lut = {}
    idx = 1

    for term in ALL_PROTECTED:
        # Match term as whole word only (boundaries on both sides)
        # Use word boundaries but allow for common punctuation around terms
        pattern = re.compile(rf"(?<![a-zA-Z0-9_]){re.escape(term)}(?![a-zA-Z0-9_])")
        # Replace all occurrences with the current placeholder
        placeholder = f"⟦T{idx}⟧"
        if pattern.search(current):
            current = pattern.sub(placeholder, current)
            lut[idx] = term
            idx += 1

    return current, lut


def unmask_text(text, lut):
    """
    Restores protected terms from placeholders using the lookup dict.
    """
    result = text
    # Iterate in reverse order (longer placeholders first, though we use unique nums)
    for idx in sorted(lut.keys(), reverse=True):
        placeholder = f"⟦T{idx}⟧"
        result = result.replace(placeholder, lut[idx])
    return result
