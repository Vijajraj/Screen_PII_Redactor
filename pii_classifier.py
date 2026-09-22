"""
pii_classifier.py — Portable PII Classification Layer for Screen PII Redactor
Phase 1 Deliverable — Snapdragon AI Lab Challenge

Constraints:
- Portable: stdlib + re ONLY (no external dependencies).
- Order of evaluation (Section 3.3):
  1. Aadhaar (Regex + Verhoeff checksum)
  2. PAN (Regex)
  3. UPI ID (Regex + known bank handle suffix list)
  4. Phone (India) (Regex)
  5. IFSC (Regex)
  6. Email (Standard email regex)
  7. Card Number (Regex + Luhn checksum)
- Confidence: 1.0 for checksum-validated types, 0.8 for regex-only types.
"""

import re
from typing import Dict, List, Optional, Tuple, Any


# =====================================================================
# 1. VERHOEFF CHECKSUM ALGORITHM (Aadhaar Verification)
# =====================================================================

_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
]

_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
]

_VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def validate_verhoeff(num_str: str) -> bool:
    """Validate numeric string using Verhoeff algorithm (returns True if valid)."""
    clean_digits = re.sub(r'\D', '', str(num_str))
    if not clean_digits:
        return False
    c = 0
    for i, item in enumerate(reversed(clean_digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(item)]]
    return c == 0


def generate_verhoeff(base_digits: str) -> str:
    """Generate valid Verhoeff-appended number from numeric string."""
    clean_digits = re.sub(r'\D', '', str(base_digits))
    c = 0
    for i, item in enumerate(reversed(clean_digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[(i + 1) % 8][int(item)]]
    check_digit = _VERHOEFF_INV[c]
    return f"{clean_digits}{check_digit}"


# =====================================================================
# 2. LUHN CHECKSUM ALGORITHM (Card Verification)
# =====================================================================

def validate_luhn(card_str: str) -> bool:
    """Validate payment card number using Luhn checksum algorithm."""
    digits = [int(c) for c in card_str if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    reverse_digits = digits[::-1]
    for i, d in enumerate(reverse_digits):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def generate_luhn(prefix: str, length: int = 16) -> str:
    """Generate a synthetically valid card number with specified prefix and length."""
    import random
    clean_prefix = re.sub(r'\D', '', str(prefix))
    digits = [int(c) for c in clean_prefix]
    while len(digits) < length - 1:
        digits.append(random.randint(0, 9))
    checksum = 0
    for i, d in enumerate(digits[::-1]):
        val = d * 2 if i % 2 == 0 else d
        if val > 9:
            val -= 9
        checksum += val
    check_digit = (10 - (checksum % 10)) % 10
    digits.append(check_digit)
    return ''.join(map(str, digits))


# =====================================================================
# 3. KNOWN UPI BANK HANDLES (Prevents over-matching generic emails)
# =====================================================================

KNOWN_UPI_HANDLES = {
    # Google Pay
    "okaxis", "okhdfcbank", "okicici", "oksbi",
    # PhonePe
    "ybl", "ibl", "axl",
    # Paytm
    "paytm", "ptyes", "pthdfc", "ptaxis", "ptsbi",
    # Amazon Pay
    "apl", "rapl",
    # BHIM & Banks
    "upi", "sbi", "hdfcbank", "icici", "axisbank", "kotak", "indus",
    "barodampay", "pnb", "cnrb", "boi", "iob", "cub", "federal",
    "rbl", "idfcbank", "postbank", "airtel", "freecharge", "mobikwik",
    "jupiteraxis", "sliceaxis", "naviaxis", "fbl"
}


# =====================================================================
# 4. REGEX PATTERNS
# =====================================================================

# Aadhaar: 12 digits, optional space/hyphen/period/comma after each 4 digits; first digit 2-9
RE_AADHAAR = re.compile(r'\b[2-9]\d{3}[\s\-,\.]?[0-9]{4}[\s\-,\.]?[0-9]{4}\b')

# PAN: 5 uppercase letters, 4 digits, 1 uppercase letter
RE_PAN = re.compile(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b')

# UPI ID: username@bank_handle
RE_UPI_CANDIDATE = re.compile(r'\b([A-Za-z0-9._\-]+)@([A-Za-z0-9]+)\b')

# Indian Phone: optional +91 prefix, starts with 6, 7, 8, or 9, followed by 9 digits
RE_PHONE_IN = re.compile(r'(?:\+91[\-\s,\.]?)?[6-9]\d{9}\b')

# IFSC: 4 uppercase letters, 0, 6 alphanumeric characters
RE_IFSC = re.compile(r'\b[A-Z]{4}0[A-Z0-9]{6}\b')

# Standard Email
RE_EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')

# Payment Card candidate (13-19 digits, optional group separators)
RE_CARD_CANDIDATE = re.compile(r'\b(?:\d{4}[\s\-,\.]?){3}\d{4}\b|\b\d{13,19}\b')


# =====================================================================
# 5. CORE CLASSIFIER FUNCTION
# =====================================================================

def classify_text(text: str, bbox: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """
    Classifies PII entities inside `text` according to Section 3.3 priority order.

    Order:
      1. Aadhaar (Regex + Verhoeff checksum) -> confidence = 1.0
      2. PAN (Regex) -> confidence = 0.8
      3. UPI ID (Regex + Known Handle) -> confidence = 0.8
      4. Phone (India) (Regex) -> confidence = 0.8
      5. IFSC (Regex) -> confidence = 0.8
      6. Email (Standard Email Regex) -> confidence = 0.8
      7. Card Number (Regex + Luhn checksum) -> confidence = 1.0

    Returns list of matches:
      [{'bbox': bbox, 'pii_type': ..., 'matched_text': ..., 'confidence': ...}]
    """
    if not text or not isinstance(text, str):
        return []

    matches: List[Dict[str, Any]] = []
    matched_spans: List[Tuple[int, int]] = []

    def span_overlaps(start: int, end: int) -> bool:
        for s, e in matched_spans:
            if not (end <= s or start >= e):
                return True
        return False

    # 1. Aadhaar (Verhoeff checksum validated)
    for m in RE_AADHAAR.finditer(text):
        candidate = m.group(0)
        digits_only = re.sub(r'\D', '', candidate)
        if len(digits_only) == 12 and validate_verhoeff(digits_only):
            start, end = m.span()
            if not span_overlaps(start, end):
                matches.append({
                    "bbox": bbox,
                    "pii_type": "AADHAAR",
                    "matched_text": candidate,
                    "confidence": 1.0,
                    "span": [start, end]
                })
                matched_spans.append((start, end))

    # 2. PAN
    for m in RE_PAN.finditer(text):
        start, end = m.span()
        if not span_overlaps(start, end):
            candidate = m.group(0)
            matches.append({
                "bbox": bbox,
                "pii_type": "PAN",
                "matched_text": candidate,
                "confidence": 0.8,
                "span": [start, end]
            })
            matched_spans.append((start, end))

    # 3. UPI ID (Restricted to known bank handles)
    for m in RE_UPI_CANDIDATE.finditer(text):
        handle = m.group(2).lower()
        if handle in KNOWN_UPI_HANDLES:
            start, end = m.span()
            if not span_overlaps(start, end):
                candidate = m.group(0)
                matches.append({
                    "bbox": bbox,
                    "pii_type": "UPI_ID",
                    "matched_text": candidate,
                    "confidence": 0.8,
                    "span": [start, end]
                })
                matched_spans.append((start, end))

    # 4. Indian Phone
    for m in RE_PHONE_IN.finditer(text):
        start, end = m.span()
        if not span_overlaps(start, end):
            candidate = m.group(0)
            matches.append({
                "bbox": bbox,
                "pii_type": "PHONE_IN",
                "matched_text": candidate,
                "confidence": 0.8,
                "span": [start, end]
            })
            matched_spans.append((start, end))

    # 5. IFSC Code
    for m in RE_IFSC.finditer(text):
        start, end = m.span()
        if not span_overlaps(start, end):
            candidate = m.group(0)
            matches.append({
                "bbox": bbox,
                "pii_type": "IFSC",
                "matched_text": candidate,
                "confidence": 0.8,
                "span": [start, end]
            })
            matched_spans.append((start, end))

    # 6. Email (Generic, not India-specific)
    for m in RE_EMAIL.finditer(text):
        start, end = m.span()
        if not span_overlaps(start, end):
            candidate = m.group(0)
            matches.append({
                "bbox": bbox,
                "pii_type": "EMAIL",
                "matched_text": candidate,
                "confidence": 0.8,
                "span": [start, end]
            })
            matched_spans.append((start, end))

    # 7. Card Number (Luhn checksum validated)
    for m in RE_CARD_CANDIDATE.finditer(text):
        candidate = m.group(0)
        digits_only = re.sub(r'\D', '', candidate)
        if 13 <= len(digits_only) <= 19 and validate_luhn(digits_only):
            start, end = m.span()
            if not span_overlaps(start, end):
                matches.append({
                    "bbox": bbox,
                    "pii_type": "CARD_NUMBER",
                    "matched_text": candidate,
                    "confidence": 1.0,
                    "span": [start, end]
                })
                matched_spans.append((start, end))

    return matches


# Self-test when run directly
if __name__ == "__main__":
    test_strings = [
        "Aadhaar test: 9876 5432 1012 and invalid 9876 5432 1019",
        "PAN test: ABCDE1234F",
        "UPI test: user@okhdfcbank and generic email user@gmail.com",
        "Phone: +91 9876543210 and 8765432109",
        "IFSC: SBIN0001234 and HDFC0004567",
        "Card: 4532 1957 3372 8189"
    ]
    print("Running PII Classifier self-test...")
    for s in test_strings:
        res = classify_text(s)
        print(f"\nInput: {s}")
        for r in res:
            print(f"  -> Found {r['pii_type']} ({r['confidence']}): '{r['matched_text']}'")
