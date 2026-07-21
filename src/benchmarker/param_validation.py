"""Runtime validation for sampling parameters."""

from typing import Any


def validate_sampling_params(params: dict[str, Any]) -> None:
    """Validate sampling parameters and raise ``ValueError`` on bad values.

    Args:
        params: Merged sampling parameters to validate.

    Raises:
        ValueError: If any parameter is present but has an invalid type or value.
    """
    if "top_k" in params:
        top_k = params["top_k"]
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}")
    if "top_p" in params:
        top_p = params["top_p"]
        if not isinstance(top_p, float) or top_p < 0 or top_p > 1:
            raise ValueError(f"top_p must be a float in [0, 1], got {top_p!r}")
    if "repeat_penalty" in params:
        repeat_penalty = params["repeat_penalty"]
        if not isinstance(repeat_penalty, float) or repeat_penalty < 1:
            raise ValueError(
                f"repeat_penalty must be a float >= 1, got {repeat_penalty!r}"
            )
    if "frequency_penalty" in params:
        frequency_penalty = params["frequency_penalty"]
        if not isinstance(frequency_penalty, float) or frequency_penalty < 0:
            raise ValueError(
                f"frequency_penalty must be a float >= 0, got {frequency_penalty!r}"
            )
    if "presence_penalty" in params:
        presence_penalty = params["presence_penalty"]
        if not isinstance(presence_penalty, float) or presence_penalty < 0:
            raise ValueError(
                f"presence_penalty must be a float >= 0, got {presence_penalty!r}"
            )
