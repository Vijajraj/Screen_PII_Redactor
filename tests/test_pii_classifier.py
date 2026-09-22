"""
test_pii_classifier.py — Unit tests for pii_classifier.py
Phase 1 — Screen PII Redactor (Snapdragon AI Lab Challenge)
"""

import pytest
from pii_classifier import (
    validate_verhoeff,
    generate_verhoeff,
    validate_luhn,
    generate_luhn,
    classify_text,
    KNOWN_UPI_HANDLES
)


# =====================================================================
# 1. VERHOEFF ALGORITHM TESTS
# =====================================================================

def test_verhoeff_checksum_generation_and_validation():
    # Test generation from arbitrary 11 digits
    test_uids = ["98765432101", "54321678901", "34567890123", "89012345678"]
    for base in test_uids:
        aadhaar = generate_verhoeff(base)
        assert len(aadhaar) == 12
        assert validate_verhoeff(aadhaar) is True

    # Test that single digit alteration fails
    corrupt_aadhaar = aadhaar[:-1] + str((int(aadhaar[-1]) + 1) % 10)
    assert validate_verhoeff(corrupt_aadhaar) is False

    # Test transposition error (adjacent digit swap)
    swapped = aadhaar[:-2] + aadhaar[-1] + aadhaar[-2]
    assert validate_verhoeff(swapped) is False


# =====================================================================
# 2. LUHN ALGORITHM TESTS
# =====================================================================

def test_luhn_checksum_generation_and_validation():
    prefixes = ["4532", "5241", "4111", "6011", "3782"]
    for prefix in prefixes:
        card = generate_luhn(prefix, 16)
        assert len(card) == 16
        assert validate_luhn(card) is True

    # Corrupted card
    corrupted_card = card[:-1] + str((int(card[-1]) + 1) % 10)
    assert validate_luhn(corrupted_card) is False


# =====================================================================
# 3. PII CLASSIFICATION TESTS
# =====================================================================

def test_classify_aadhaar():
    valid_aadhaar = generate_verhoeff("98765432101")
    text = f"Customer UIDAI Aadhaar: {valid_aadhaar[:4]} {valid_aadhaar[4:8]} {valid_aadhaar[8:]}"
    results = classify_text(text)
    assert len(results) == 1
    assert results[0]["pii_type"] == "AADHAAR"
    assert results[0]["confidence"] == 1.0


def test_classify_pan():
    text = "Permanent Account Number: ABCDE1234F verified."
    results = classify_text(text)
    assert len(results) == 1
    assert results[0]["pii_type"] == "PAN"
    assert results[0]["matched_text"] == "ABCDE1234F"
    assert results[0]["confidence"] == 0.8


def test_classify_upi_vs_email():
    # Known bank handle -> UPI_ID
    upi_text = "Pay via rahul@okhdfcbank or deepak@paytm"
    results = classify_text(upi_text)
    assert len(results) == 2
    assert results[0]["pii_type"] == "UPI_ID"
    assert results[1]["pii_type"] == "UPI_ID"

    # Generic domain -> EMAIL (not UPI)
    email_text = "Contact support@github.com or billing@enterprise.org"
    results = classify_text(email_text)
    assert len(results) == 2
    assert results[0]["pii_type"] == "EMAIL"
    assert results[1]["pii_type"] == "EMAIL"


def test_classify_indian_phone():
    text = "Call direct line +91 9845123456 or alternate 8765432109"
    results = classify_text(text)
    assert len(results) == 2
    assert results[0]["pii_type"] == "PHONE_IN"
    assert results[1]["pii_type"] == "PHONE_IN"


def test_classify_ifsc():
    text = "Branch IFSC code HDFC0001234 and SBIN0004321"
    results = classify_text(text)
    assert len(results) == 2
    assert results[0]["pii_type"] == "IFSC"
    assert results[1]["pii_type"] == "IFSC"


def test_classify_card_number():
    card = generate_luhn("4532", 16)
    formatted = f"{card[:4]} {card[4:8]} {card[8:12]} {card[12:]}"
    text = f"Credit card on file: {formatted}"
    results = classify_text(text)
    assert len(results) == 1
    assert results[0]["pii_type"] == "CARD_NUMBER"
    assert results[0]["confidence"] == 1.0


def test_clean_text_no_false_positives():
    clean_text = """
    Cluster Latency: P99 12.4 ms, P50 3.1 ms
    Memory Footprint: 6.4 GB allocated across 8 CPU worker cores.
    GET /v1/indexes/vector_store/query status 200 OK.
    """
    results = classify_text(clean_text)
    assert len(results) == 0
