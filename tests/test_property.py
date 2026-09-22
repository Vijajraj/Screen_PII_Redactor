"""
test_property.py -- Property-based tests using Hypothesis.

Verifies invariants:
  1. Any Verhoeff-generated 12-digit number is detected as AADHAAR
  2. Any Luhn-valid card number is detected as CARD_NUMBER
  3. Any valid PAN format string is detected as PAN
  4. Any valid Indian phone number is detected as PHONE_IN
  5. Any valid IFSC code is detected as IFSC
  6. Any valid UPI ID with known handle is detected as UPI_ID
  7. Any valid email is detected as EMAIL
  8. Random garbage text never triggers false positives
  9. Verhoeff roundtrip: generate + validate = True
 10. Luhn roundtrip: generate + validate = True
"""

import re
import string

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from pii_classifier import (
    KNOWN_UPI_HANDLES,
    classify_text,
    generate_luhn,
    generate_verhoeff,
    validate_luhn,
    validate_verhoeff,
)

# =====================================================================
# STRATEGY HELPERS
# =====================================================================

# Generate 11-digit base for Aadhaar (first digit 2-9)
aadhaar_base = st.builds(
    lambda first, rest: str(first) + "".join(str(d) for d in rest),
    first=st.integers(min_value=2, max_value=9),
    rest=st.lists(st.integers(min_value=0, max_value=9), min_size=10, max_size=10),
)

# Generate card prefix (4-6 digits) and length (13-19)
card_prefix = st.text(alphabet=string.digits, min_size=1, max_size=4).filter(lambda s: len(s) >= 1)
card_length = st.integers(min_value=13, max_value=19)

# Generate PAN: exactly AAAAA9999A format
pan_strategy = st.builds(
    lambda letters, digits, last: letters + digits + last,
    letters=st.text(alphabet=string.ascii_uppercase, min_size=5, max_size=5),
    digits=st.text(alphabet=string.digits, min_size=4, max_size=4),
    last=st.text(alphabet=string.ascii_uppercase, min_size=1, max_size=1),
)

# Indian phone numbers (starting with 6, 7, 8, or 9)
phone_strategy = st.builds(
    lambda first, rest: str(first) + "".join(str(d) for d in rest),
    first=st.sampled_from([6, 7, 8, 9]),
    rest=st.lists(st.integers(min_value=0, max_value=9), min_size=9, max_size=9),
)

# IFSC codes: 4 uppercase letters + '0' + 6 alphanumeric chars
ifsc_strategy = st.builds(
    lambda bank, branch: bank + "0" + branch,
    bank=st.text(alphabet=string.ascii_uppercase, min_size=4, max_size=4),
    branch=st.text(alphabet=string.ascii_uppercase + string.digits, min_size=6, max_size=6),
)

# UPI IDs with known bank handles
upi_strategy = st.builds(
    lambda user, handle: f"{user}@{handle}",
    user=st.text(
        alphabet=string.ascii_lowercase + string.digits + "._-",
        min_size=3,
        max_size=15,
    ).filter(lambda s: s[0].isalnum()),
    handle=st.sampled_from(sorted(KNOWN_UPI_HANDLES)),
)

# Email strategy
email_strategy = st.builds(
    lambda local, domain, tld: f"{local}@{domain}.{tld}",
    local=st.text(
        alphabet=string.ascii_lowercase + string.digits + "._%+-",
        min_size=1,
        max_size=20,
    ).filter(lambda s: s[0].isalnum()),
    domain=st.text(
        alphabet=string.ascii_lowercase + string.digits + ".-",
        min_size=2,
        max_size=15,
    ).filter(lambda s: s[0].isalnum() and s[-1].isalnum()),
    tld=st.text(alphabet=string.ascii_lowercase, min_size=2, max_size=5),
)

# Safe garbage: alphabetic text that can't accidentally be PII
safe_garbage = st.text(
    alphabet="abcdefghjklmnopqrstuvwxyz ",
    min_size=5,
    max_size=50,
).filter(lambda s: not re.search(r"[A-Z]{5}\d{4}[A-Z]", s.upper()))


# =====================================================================
# VERHOEFF PROPERTY TESTS
# =====================================================================


class TestPropertyVerhoeff:
    """Verhoeff invariants via Hypothesis."""

    @given(base=aadhaar_base)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_generate_then_validate_roundtrip(self, base):
        """Invariant: generate_verhoeff(base) always validates."""
        aadhaar = generate_verhoeff(base)
        assert validate_verhoeff(aadhaar) is True

    @given(base=aadhaar_base)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_valid_aadhaar_is_classified(self, base):
        """Invariant: any Verhoeff-valid 12-digit number (first digit 2-9) is
        detected as AADHAAR by classify_text."""
        aadhaar = generate_verhoeff(base)
        spaced = f"{aadhaar[:4]} {aadhaar[4:8]} {aadhaar[8:]}"
        results = classify_text(f"UID: {spaced}")
        aadhaar_results = [r for r in results if r["pii_type"] == "AADHAAR"]
        assert len(aadhaar_results) >= 1

    @given(base=aadhaar_base, corrupt_pos=st.integers(min_value=0, max_value=11))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_corrupted_aadhaar_fails_validation(self, base, corrupt_pos):
        """Invariant: single-digit corruption breaks Verhoeff."""
        aadhaar = generate_verhoeff(base)
        original_digit = int(aadhaar[corrupt_pos])
        corrupt_digit = (original_digit + 1) % 10
        corrupted = aadhaar[:corrupt_pos] + str(corrupt_digit) + aadhaar[corrupt_pos + 1 :]
        assert validate_verhoeff(corrupted) is False


# =====================================================================
# LUHN PROPERTY TESTS
# =====================================================================


class TestPropertyLuhn:
    """Luhn invariants via Hypothesis."""

    @given(prefix=card_prefix, length=card_length)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_generate_then_validate_roundtrip(self, prefix, length):
        """Invariant: generate_luhn(prefix, length) always validates."""
        assume(len(prefix) < length)
        card = generate_luhn(prefix, length)
        assert validate_luhn(card) is True

    @given(prefix=card_prefix)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_valid_card_is_classified(self, prefix):
        """Invariant: any Luhn-valid 16-digit card is detected as CARD_NUMBER."""
        assume(len(prefix) < 16)
        card = generate_luhn(prefix, 16)
        formatted = f"{card[:4]} {card[4:8]} {card[8:12]} {card[12:]}"
        results = classify_text(f"Card: {formatted}")
        card_results = [r for r in results if r["pii_type"] == "CARD_NUMBER"]
        assert len(card_results) >= 1


# =====================================================================
# PAN PROPERTY TESTS
# =====================================================================


class TestPropertyPAN:
    """PAN format invariants."""

    @given(pan=pan_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_valid_pan_detected(self, pan):
        """Invariant: any AAAAA9999A string is detected as PAN."""
        results = classify_text(f"PAN: {pan}")
        pan_results = [r for r in results if r["pii_type"] == "PAN"]
        assert len(pan_results) >= 1


# =====================================================================
# PHONE PROPERTY TESTS
# =====================================================================


class TestPropertyPhone:
    """Indian phone invariants."""

    @given(phone=phone_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_valid_phone_detected(self, phone):
        """Invariant: 10-digit number starting with 6-9 is detected as PHONE_IN."""
        results = classify_text(f"Call: {phone}")
        phone_results = [r for r in results if r["pii_type"] == "PHONE_IN"]
        assert len(phone_results) >= 1


# =====================================================================
# IFSC PROPERTY TESTS
# =====================================================================


class TestPropertyIFSC:
    """IFSC code invariants."""

    @given(ifsc=ifsc_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_valid_ifsc_detected(self, ifsc):
        """Invariant: XXXX0YYYYYY is detected as IFSC."""
        results = classify_text(f"IFSC: {ifsc}")
        ifsc_results = [r for r in results if r["pii_type"] == "IFSC"]
        assert len(ifsc_results) >= 1


# =====================================================================
# UPI PROPERTY TESTS
# =====================================================================


class TestPropertyUPI:
    """UPI ID invariants."""

    @given(upi=upi_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_valid_upi_detected(self, upi):
        """Invariant: user@known_bank_handle is detected as UPI_ID."""
        results = classify_text(f"Pay: {upi}")
        upi_results = [r for r in results if r["pii_type"] == "UPI_ID"]
        assert len(upi_results) >= 1


# =====================================================================
# EMAIL PROPERTY TESTS
# =====================================================================


class TestPropertyEmail:
    """Email invariants."""

    @given(email=email_strategy)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_valid_email_detected(self, email):
        """Invariant: user@domain.tld is detected as EMAIL or UPI_ID
        (if handle happens to match a UPI bank)."""
        results = classify_text(f"Contact: {email}")
        email_or_upi = [r for r in results if r["pii_type"] in ("EMAIL", "UPI_ID")]
        assert len(email_or_upi) >= 1


# =====================================================================
# NEGATIVE / FALSE POSITIVE PROPERTY TESTS
# =====================================================================


class TestPropertyNegative:
    """No false positives on random safe text."""

    @given(text=safe_garbage)
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_garbage_text_no_pii(self, text):
        """Invariant: random lowercase alphabetic text never triggers PII."""
        results = classify_text(text)
        assert len(results) == 0, f"False positive on garbage: {text!r} -> {results}"
