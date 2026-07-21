"""Tests for sampling parameter validation."""

import pytest

from benchmarker.param_validation import validate_sampling_params


def test_validate_sampling_params_top_k_positive() -> None:
    with pytest.raises(ValueError):
        validate_sampling_params({"top_k": 0})


def test_validate_sampling_params_top_p_range() -> None:
    with pytest.raises(ValueError):
        validate_sampling_params({"top_p": 1.5})


def test_validate_sampling_params_repeat_penalty() -> None:
    with pytest.raises(ValueError):
        validate_sampling_params({"repeat_penalty": 0.5})
