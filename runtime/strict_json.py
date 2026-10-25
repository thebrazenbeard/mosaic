"""Strict JSON decoding for untrusted model and local HTTP responses.

The default Python decoder silently keeps the last duplicate key and accepts
NaN/Infinity, neither of which is unambiguous standards-compliant JSON.
"""
from __future__ import annotations

import json
import math


def loads(raw: str) -> object:
    """Return JSON data or raise ValueError on ambiguous/nonfinite objects."""
    if not isinstance(raw, str):
        raise TypeError("JSON source must be text")

    def unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        out: dict[str, object] = {}
        for name, value in pairs:
            if name in out:
                raise ValueError("duplicate JSON object key")
            out[name] = value
        return out

    def reject_constant(value: str) -> object:
        raise ValueError("nonstandard JSON numeric constant")

    def finite_float(value: str) -> float:
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError("nonfinite JSON number")
        return converted

    try:
        return json.loads(raw, object_pairs_hook=unique_pairs,
                          parse_constant=reject_constant, parse_float=finite_float)
    except RecursionError as exc:
        # An untrusted response can be below the byte limit but nested deeply
        # enough to exceed Python's decoder recursion bound. Normalize this to
        # the decoder's declared invalid-input domain; its callers convert
        # ValueError to their explicit fail-closed boundary exceptions.
        raise ValueError("JSON nesting exceeds decoder depth") from exc
