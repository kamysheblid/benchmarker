"""Tests for sampling parameter validation."""

import pytest

from benchmarker.param_validation import validate_sampling_params


# --- top_k [0, INT32_MAX] ---


def test_validate_sampling_params_top_k_negative() -> None:
    with pytest.raises(ValueError):
        validate_sampling_params({"top_k": -1})


def test_validate_sampling_params_top_k_over_max() -> None:
    with pytest.raises(ValueError):
        validate_sampling_params({"top_k": 2_147_483_648})


def test_validate_sampling_params_top_k_zero_ok() -> None:
    validate_sampling_params({"top_k": 0})  # should not raise


def test_validate_sampling_params_top_k_int_type() -> None:
    with pytest.raises(ValueError):
        validate_sampling_params({"top_k": 1.5})


# --- top_p [0.0, 1.0] ---


def test_validate_sampling_params_top_p_range() -> None:
    with pytest.raises(ValueError):
        validate_sampling_params({"top_p": 1.5})


def test_validate_sampling_params_top_p_negative() -> None:
    with pytest.raises(ValueError):
        validate_sampling_params({"top_p": -0.1})


def test_validate_sampling_params_top_p_int_ok() -> None:
    validate_sampling_params({"top_p": 1})  # int 1 is valid


# --- min_p [0.0, 1.0] ---


def test_validate_sampling_params_min_p_range() -> None:
    with pytest.raises(ValueError):
        validate_sampling_params({"min_p": 1.5})


def test_validate_sampling_params_min_p_negative() -> None:
    with pytest.raises(ValueError):
        validate_sampling_params({"min_p": -0.1})


# --- xtc_probability [0.0, 1.0] ---


def test_validate_sampling_params_xtc_probability_range() -> None:
    with pytest.raises(ValueError):
        validate_sampling_params({"xtc_probability": 1.1})


def test_validate_sampling_params_xtc_probability_negative() -> None:
    with pytest.raises(ValueError):
        validate_sampling_params({"xtc_probability": -0.1})


# --- xtc_threshold [0.0, 1.0] ---


def test_validate_sampling_params_xtc_threshold_range() -> None:
    with pytest.raises(ValueError):
        validate_sampling_params({"xtc_threshold": 1.1})


def test_validate_sampling_params_xtc_threshold_negative() -> None:
    with pytest.raises(ValueError):
        validate_sampling_params({"xtc_threshold": -0.1})


# --- temperature >= 0.0 ---


def test_validate_sampling_params_temperature_negative() -> None:
    with pytest.raises(ValueError):
        validate_sampling_params({"temperature": -0.1})


def test_validate_sampling_params_temperature_zero_ok() -> None:
    validate_sampling_params({"temperature": 0.0})  # should not raise


# --- repeat_penalty >= 1 ---


def test_validate_sampling_params_repeat_penalty() -> None:
    with pytest.raises(ValueError):
        validate_sampling_params({"repeat_penalty": 0.5})


# --- frequency_penalty >= 0 ---


def test_validate_sampling_params_frequency_penalty() -> None:
    with pytest.raises(ValueError):
        validate_sampling_params({"frequency_penalty": -0.1})


# --- presence_penalty >= 0 ---


def test_validate_sampling_params_presence_penalty() -> None:
    with pytest.raises(ValueError):
        validate_sampling_params({"presence_penalty": -0.1})


# --- repeat_last_n [-1, INT32_MAX] ---


def test_validate_sampling_params_repeat_last_n_negative() -> None:
    with pytest.raises(ValueError):
        validate_sampling_params({"repeat_last_n": -2})


def test_validate_sampling_params_repeat_last_n_over_max() -> None:
    with pytest.raises(ValueError):
        validate_sampling_params({"repeat_last_n": 2_147_483_648})


def test_validate_sampling_params_repeat_last_n_minus_one_ok() -> None:
    validate_sampling_params({"repeat_last_n": -1})  # should not raise


# --- dry_allowed_length [-1, INT32_MAX] ---


def test_validate_sampling_params_dry_allowed_length_negative() -> None:
    with pytest.raises(ValueError):
        validate_sampling_params({"dry_allowed_length": -2})


def test_validate_sampling_params_dry_allowed_length_minus_one_ok() -> None:
    validate_sampling_params({"dry_allowed_length": -1})  # should not raise


# --- dry_penalty_last_n [-1, INT32_MAX] ---


def test_validate_sampling_params_dry_penalty_last_n_negative() -> None:
    with pytest.raises(ValueError):
        validate_sampling_params({"dry_penalty_last_n": -2})


def test_validate_sampling_params_dry_penalty_last_n_minus_one_ok() -> None:
    validate_sampling_params({"dry_penalty_last_n": -1})  # should not raise


# --- mirostat [0, 2] ---


def test_validate_sampling_params_mirostat_range() -> None:
    with pytest.raises(ValueError):
        validate_sampling_params({"mirostat": 3})


def test_validate_sampling_params_mirostat_negative() -> None:
    with pytest.raises(ValueError):
        validate_sampling_params({"mirostat": -1})


def test_validate_sampling_params_mirostat_ok() -> None:
    validate_sampling_params({"mirostat": 2})  # should not raise
