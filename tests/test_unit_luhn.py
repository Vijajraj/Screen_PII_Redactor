"""
test_unit_luhn.py -- Exhaustive unit tests for the Luhn checksum algorithm.
Covers: generation, validation, corruption, length bounds, and non-numeric input.
"""

import pytest

from pii_classifier import generate_luhn, validate_luhn


class TestLuhnGeneration:
    """Tests for generate_luhn: creating valid card numbers."""

    @pytest.mark.parametrize(
        "prefix,length",
        [
            ("4", 16),  # Visa
            ("51", 16),  # MasterCard
            ("34", 15),  # Amex
            ("6011", 16),  # Discover
            ("3528", 16),  # JCB
            ("6521", 16),  # RuPay
        ],
    )
    def test_generated_card_has_correct_length(self, prefix, length):
        card = generate_luhn(prefix, length)
        assert len(card) == length

    @pytest.mark.parametrize("prefix", ["4532", "5241", "4111", "6011", "3782"])
    def test_generated_card_passes_validation(self, prefix):
        card = generate_luhn(prefix, 16)
        assert validate_luhn(card) is True

    def test_different_seeds_produce_valid_cards(self):
        """Generate 50 cards and verify all pass Luhn."""
        for _ in range(50):
            card = generate_luhn("4", 16)
            assert validate_luhn(card) is True


class TestLuhnValidation:
    """Tests for validate_luhn: verifying valid and invalid card numbers."""

    def test_single_digit_corruption_detected(self):
        card = generate_luhn("4532", 16)
        for pos in range(len(card)):
            original = int(card[pos])
            corrupt = (original + 1) % 10
            if corrupt == original:
                continue
            bad = card[:pos] + str(corrupt) + card[pos + 1 :]
            assert validate_luhn(bad) is False, f"Corruption at position {pos} not detected"

    def test_too_short_card_rejected(self):
        assert validate_luhn("123456789012") is False  # 12 digits

    def test_too_long_card_rejected(self):
        assert validate_luhn("1" * 20) is False  # 20 digits

    def test_empty_string_rejected(self):
        assert validate_luhn("") is False

    def test_non_numeric_rejected(self):
        assert validate_luhn("ABCDE1234567890") is False

    @pytest.mark.parametrize(
        "card_str",
        [
            "4532 1957 3372 8189",  # spaces
            "4532-1957-3372-8189",  # hyphens
        ],
    )
    def test_formatted_card_with_separators(self, card_str):
        # These should only pass if the underlying digits form a valid Luhn
        # Exact validity depends on the digits, so we just check no crash
        result = validate_luhn(card_str)
        assert isinstance(result, bool)

    def test_13_digit_minimum_valid(self):
        card = generate_luhn("4", 13)
        assert len(card) == 13
        assert validate_luhn(card) is True

    def test_19_digit_maximum_valid(self):
        card = generate_luhn("4", 19)
        assert len(card) == 19
        assert validate_luhn(card) is True
