"""Runtime validation for sampling parameters."""

from typing import Any

_INT32_MAX = 2_147_483_647


def validate_sampling_params(params: dict[str, Any]) -> None:
    """Validate sampling parameters and raise ``ValueError`` on bad values.

    Limits are taken from llama.cpp ``server-schema.cpp``.

    Args:
        params: Merged sampling parameters to validate.

    Raises:
        ValueError: If any parameter is present but has an invalid type or value.
    """
    if "top_k" in params:
        top_k = params["top_k"]
        if not isinstance(top_k, int) or top_k < 0 or top_k > _INT32_MAX:
            raise ValueError(
                f"top_k must be an integer in [0, {_INT32_MAX}], got {top_k!r}"
            )
    if "top_p" in params:
        top_p = params["top_p"]
        if not isinstance(top_p, (int, float)) or top_p < 0.0 or top_p > 1.0:
            raise ValueError(f"top_p must be a float in [0.0, 1.0], got {top_p!r}")
    if "min_p" in params:
        min_p = params["min_p"]
        if not isinstance(min_p, (int, float)) or min_p < 0.0 or min_p > 1.0:
            raise ValueError(f"min_p must be a float in [0.0, 1.0], got {min_p!r}")
    if "xtc_probability" in params:
        xtc_prob = params["xtc_probability"]
        if not isinstance(xtc_prob, (int, float)) or xtc_prob < 0.0 or xtc_prob > 1.0:
            raise ValueError(
                f"xtc_probability must be a float in [0.0, 1.0], got {xtc_prob!r}"
            )
    if "xtc_threshold" in params:
        xtc_thresh = params["xtc_threshold"]
        if not isinstance(xtc_thresh, (int, float)) or xtc_thresh < 0.0 or xtc_thresh > 1.0:
            raise ValueError(
                f"xtc_threshold must be a float in [0.0, 1.0], got {xtc_thresh!r}"
            )
    if "temperature" in params:
        temp = params["temperature"]
        if not isinstance(temp, (int, float)) or temp < 0.0:
            raise ValueError(
                f"temperature must be a float >= 0.0, got {temp!r}"
            )
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
    if "repeat_last_n" in params:
        rln = params["repeat_last_n"]
        if not isinstance(rln, int) or rln < -1 or rln > _INT32_MAX:
            raise ValueError(
                f"repeat_last_n must be an integer in [-1, {_INT32_MAX}], got {rln!r}"
            )
    if "dry_allowed_length" in params:
        dal = params["dry_allowed_length"]
        if not isinstance(dal, int) or dal < -1 or dal > _INT32_MAX:
            raise ValueError(
                f"dry_allowed_length must be an integer in [-1, {_INT32_MAX}], got {dal!r}"
            )
    if "dry_penalty_last_n" in params:
        dpln = params["dry_penalty_last_n"]
        if not isinstance(dpln, int) or dpln < -1 or dpln > _INT32_MAX:
            raise ValueError(
                f"dry_penalty_last_n must be an integer in [-1, {_INT32_MAX}], got {dpln!r}"
            )
    if "mirostat" in params:
        mirostat = params["mirostat"]
        if not isinstance(mirostat, int) or mirostat < 0 or mirostat > 2:
            raise ValueError(
                f"mirostat must be an integer in [0, 2], got {mirostat!r}"
            )
