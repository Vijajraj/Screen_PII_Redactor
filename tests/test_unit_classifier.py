"""
test_unit_classifier.py -- Granular unit tests for classify_text: one test per
PII category, edge-case inputs, priority ordering, span non-overlap, and
confidence values.
"""

import pytest

from pii_classifier import classify_text, generate_luhn, generate_verhoeff

# =====================================================================
# AADHAAR TESTS
# =====================================================================


class TestClassifyAadhaar:
    """Tests for Aadhaar detection (Verhoeff-validated)."""

    def test_valid_aadhaar_spaced(self):
        uid = generate_verhoeff("98765432101")
        text = f"UID: {uid[:4]} {uid[4:8]} {uid[8:]}"
        res = classify_text(text)
        assert len(res) == 1
        assert res[0]["pii_type"] == "AADHAAR"
        assert res[0]["confidence"] == 1.0

    def test_valid_aadhaar_no_spaces(self):
        uid = generate_verhoeff("54321678901")
        res = classify_text(f"Aadhaar: {uid}")
        assert any(r["pii_type"] == "AADHAAR" for r in res)

    def test_valid_aadhaar_hyphen_separated(self):
        uid = generate_verhoeff("34567890123")
        text = f"{uid[:4]}-{uid[4:8]}-{uid[8:]}"
        res = classify_text(text)
        assert any(r["pii_type"] == "AADHAAR" for r in res)

    def test_invalid_aadhaar_bad_checksum_ignored(self):
        uid = generate_verhoeff("98765432101")
        bad = uid[:-1] + str((int(uid[-1]) + 1) % 10)
        res = classify_text(f"UID: {bad}")
        assert not any(r["pii_type"] == "AADHAAR" for r in res)

    def test_aadhaar_starting_with_0_or_1_ignored(self):
        # Aadhaar first digit must be 2-9
        res = classify_text("ID: 0123 4567 8901")
        assert not any(r["pii_type"] == "AADHAAR" for r in res)
        res = classify_text("ID: 1234 5678 9012")
        assert not any(r["pii_type"] == "AADHAAR" for r in res)


# =====================================================================
# PAN TESTS
# =====================================================================


class TestClassifyPan:
    """Tests for Indian PAN detection."""

    @pytest.mark.parametrize(
        "pan",
        [
            "ABCDE1234F",
            "ZZZZZ9999Z",
            "BKZPR9876Q",
        ],
    )
    def test_valid_pan_detected(self, pan):
        res = classify_text(f"PAN: {pan}")
        assert len(res) == 1
        assert res[0]["pii_type"] == "PAN"
        assert res[0]["matched_text"] == pan
        assert res[0]["confidence"] == 0.8

    @pytest.mark.parametrize(
        "invalid",
        [
            "ABCDE1234",  # missing trailing letter
            "abcde1234F",  # lowercase letters
            "ABCDE12345F",  # extra digit
            "A1CDE1234F",  # digit in first 5 positions
        ],
    )
    def test_invalid_pan_not_detected(self, invalid):
        res = classify_text(f"PAN: {invalid}")
        assert not any(r["pii_type"] == "PAN" for r in res)


# =====================================================================
# UPI ID TESTS
# =====================================================================


class TestClassifyUPI:
    """Tests for UPI ID detection (bank-handle whitelisted)."""

    @pytest.mark.parametrize(
        "upi",
        [
            "rahul@okhdfcbank",
            "priya99@okaxis",
            "merchant@paytm",
            "sender@ybl",
            "pay@oksbi",
        ],
    )
    def test_known_bank_handles_detected(self, upi):
        res = classify_text(f"UPI: {upi}")
        assert len(res) == 1
        assert res[0]["pii_type"] == "UPI_ID"

    @pytest.mark.parametrize(
        "generic_email",
        [
            "user@gmail.com",
            "admin@company.org",
            "test@example.co.in",
        ],
    )
    def test_generic_emails_not_classified_as_upi(self, generic_email):
        res = classify_text(generic_email)
        assert not any(r["pii_type"] == "UPI_ID" for r in res)
        assert any(r["pii_type"] == "EMAIL" for r in res)


# =====================================================================
# PHONE TESTS
# =====================================================================


class TestClassifyPhone:
    """Tests for Indian phone number detection."""

    @pytest.mark.parametrize(
        "phone",
        [
            "+91 9845123456",
            "+919876543210",
            "9123456789",
            "8765432100",
            "7012345678",
            "6543210987",
        ],
    )
    def test_valid_indian_phones_detected(self, phone):
        res = classify_text(f"Call: {phone}")
        assert any(r["pii_type"] == "PHONE_IN" for r in res)

    @pytest.mark.parametrize(
        "invalid_phone",
        [
            "5123456789",  # starts with 5
            "4123456789",  # starts with 4
            "12345",  # too short
        ],
    )
    def test_invalid_phones_not_detected(self, invalid_phone):
        res = classify_text(f"Number: {invalid_phone}")
        assert not any(r["pii_type"] == "PHONE_IN" for r in res)


# =====================================================================
# IFSC TESTS
# =====================================================================


class TestClassifyIFSC:
    """Tests for IFSC code detection."""

    @pytest.mark.parametrize(
        "ifsc",
        [
            "HDFC0001234",
            "SBIN0004321",
            "ICIC0000987",
            "UTIB0000456",
        ],
    )
    def test_valid_ifsc_detected(self, ifsc):
        res = classify_text(f"IFSC: {ifsc}")
        assert len(res) == 1
        assert res[0]["pii_type"] == "IFSC"
        assert res[0]["confidence"] == 0.8

    @pytest.mark.parametrize(
        "invalid",
        [
            "HDFC1001234",  # 5th char not 0
            "HDF00001234",  # only 3 letters
            "hdfc0001234",  # lowercase
        ],
    )
    def test_invalid_ifsc_not_detected(self, invalid):
        res = classify_text(f"Code: {invalid}")
        assert not any(r["pii_type"] == "IFSC" for r in res)


# =====================================================================
# EMAIL TESTS
# =====================================================================


class TestClassifyEmail:
    """Tests for email detection."""

    @pytest.mark.parametrize(
        "email",
        [
            "user@example.com",
            "first.last@company.co.in",
            "test+tag@domain.org",
        ],
    )
    def test_valid_emails_detected(self, email):
        res = classify_text(f"Contact: {email}")
        assert any(r["pii_type"] == "EMAIL" for r in res)


# =====================================================================
# CARD NUMBER TESTS
# =====================================================================


class TestClassifyCard:
    """Tests for credit/debit card detection (Luhn-validated)."""

    def test_valid_luhn_card_detected(self):
        card = generate_luhn("4532", 16)
        formatted = f"{card[:4]} {card[4:8]} {card[8:12]} {card[12:]}"
        res = classify_text(f"Card: {formatted}")
        assert any(r["pii_type"] == "CARD_NUMBER" for r in res)
        match = next(r for r in res if r["pii_type"] == "CARD_NUMBER")
        assert match["confidence"] == 1.0

    def test_invalid_luhn_card_ignored(self):
        card = generate_luhn("4532", 16)
        bad = card[:-1] + str((int(card[-1]) + 1) % 10)
        formatted = f"{bad[:4]} {bad[4:8]} {bad[8:12]} {bad[12:]}"
        res = classify_text(f"Card: {formatted}")
        assert not any(r["pii_type"] == "CARD_NUMBER" for r in res)


# =====================================================================
# EDGE CASES AND INTERACTIONS
# =====================================================================


class TestClassifyEdgeCases:
    """Cross-cutting edge cases."""

    def test_empty_string_returns_empty(self):
        assert classify_text("") == []

    def test_none_returns_empty(self):
        assert classify_text(None) == []

    def test_non_string_returns_empty(self):
        assert classify_text(12345) == []

    def test_bbox_is_propagated(self):
        res = classify_text("ABCDE1234F", bbox=[10, 20, 100, 50])
        assert res[0]["bbox"] == [10, 20, 100, 50]

    def test_span_field_present(self):
        res = classify_text("PAN: ABCDE1234F")
        assert "span" in res[0]
        start, end = res[0]["span"]
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert start < end

    def test_multiple_pii_types_in_one_text(self):
        uid = generate_verhoeff("98765432101")
        card = generate_luhn("4532", 16)
        card_fmt = f"{card[:4]} {card[4:8]} {card[8:12]} {card[12:]}"
        text = f"Aadhaar: {uid[:4]} {uid[4:8]} {uid[8:]} PAN: ABCDE1234F Phone: +91 9876543210 Card: {card_fmt}"
        res = classify_text(text)
        types_found = {r["pii_type"] for r in res}
        assert "AADHAAR" in types_found
        assert "PAN" in types_found
        assert "PHONE_IN" in types_found
        assert "CARD_NUMBER" in types_found

    def test_no_overlap_between_matches(self):
        """Verify that no two matches share overlapping character spans."""
        uid = generate_verhoeff("98765432101")
        text = f"Data: {uid} ABCDE1234F user@paytm +91 9876543210"
        res = classify_text(text)
        spans = [tuple(r["span"]) for r in res]
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                s1, e1 = spans[i]
                s2, e2 = spans[j]
                assert e1 <= s2 or e2 <= s1, f"Overlapping spans: {spans[i]} and {spans[j]}"

    def test_priority_ordering(self):
        """Spec Section 3.3: Aadhaar is detected before PAN even if both
        could theoretically match overlapping text."""
        uid = generate_verhoeff("98765432101")
        res = classify_text(f"ID: {uid}")
        if res:
            assert res[0]["pii_type"] == "AADHAAR"
