"""
test_unit_verhoeff.py -- Exhaustive unit tests for the Verhoeff checksum algorithm.
Covers: generation, validation, single-digit corruption, transposition errors,
        empty / non-numeric input, and leading-zero handling.
"""

import pytest

from pii_classifier import generate_verhoeff, validate_verhoeff


class TestVerhoeffGeneration:
    """Tests for generate_verhoeff: appending a valid check digit."""

    @pytest.mark.parametrize(
        "base",
        [
            "98765432101",
            "54321678901",
            "34567890123",
            "89012345678",
            "20000000000",  # minimum valid first digit for Aadhaar
            "99999999999",  # all nines
            "11111111111",  # all ones
        ],
    )
    def test_generated_number_has_correct_length(self, base):
        result = generate_verhoeff(base)
        assert len(result) == len(base) + 1

    @pytest.mark.parametrize(
        "base",
        [
            "98765432101",
            "54321678901",
            "34567890123",
        ],
    )
    def test_generated_number_passes_validation(self, base):
        result = generate_verhoeff(base)
        assert validate_verhoeff(result) is True

    def test_generated_check_digit_is_single_digit(self):
        result = generate_verhoeff("12345678901")
        check_digit = result[-1]
        assert check_digit.isdigit()
        assert 0 <= int(check_digit) <= 9


class TestVerhoeffValidation:
    """Tests for validate_verhoeff: confirming valid and invalid numbers."""

    def test_known_valid_aadhaar(self):
        # 987654321012 is generated from base 98765432101
        aadhaar = generate_verhoeff("98765432101")
        assert validate_verhoeff(aadhaar) is True

    def test_single_digit_corruption_detected(self):
        aadhaar = generate_verhoeff("98765432101")
        for pos in range(len(aadhaar)):
            original_digit = int(aadhaar[pos])
            corrupt_digit = (original_digit + 1) % 10
            corrupted = aadhaar[:pos] + str(corrupt_digit) + aadhaar[pos + 1 :]
            assert validate_verhoeff(corrupted) is False, f"Corruption at position {pos} was not detected: {corrupted}"

    def test_adjacent_transposition_detected(self):
        aadhaar = generate_verhoeff("98765432101")
        for pos in range(len(aadhaar) - 1):
            if aadhaar[pos] == aadhaar[pos + 1]:
                continue  # swapping identical digits is a no-op
            swapped = aadhaar[:pos] + aadhaar[pos + 1] + aadhaar[pos] + aadhaar[pos + 2 :]
            assert validate_verhoeff(swapped) is False, (
                f"Transposition at positions {pos},{pos + 1} not detected: {swapped}"
            )

    def test_empty_string_returns_false(self):
        assert validate_verhoeff("") is False

    def test_non_numeric_string_returns_false(self):
        assert validate_verhoeff("abcdefghijkl") is False

    def test_short_numeric_returns_false_or_true_by_algorithm(self):
        # "0" has Verhoeff check = 0, so it validates.
        assert validate_verhoeff("0") is True
        # "1" does not.
        assert validate_verhoeff("1") is False

    def test_handles_whitespace_and_separators(self):
        aadhaar = generate_verhoeff("98765432101")
        spaced = f"{aadhaar[:4]} {aadhaar[4:8]} {aadhaar[8:]}"
        assert validate_verhoeff(spaced) is True


class TestVerhoeffEdgeCases:
    """Edge cases and boundary conditions."""

    def test_all_zeros_base(self):
        result = generate_verhoeff("00000000000")
        assert validate_verhoeff(result) is True

    def test_very_long_number(self):
        result = generate_verhoeff("1234567890123456789")
        assert validate_verhoeff(result) is True

    def test_single_digit_base(self):
        result = generate_verhoeff("5")
        assert validate_verhoeff(result) is True
        assert len(result) == 2
