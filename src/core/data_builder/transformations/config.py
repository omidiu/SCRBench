"""
Transformation configuration that must not import transformer modules.

This module intentionally contains only static configuration + helpers to avoid
circular imports between `transformations/__init__.py` and individual
transformation implementations.
"""

from __future__ import annotations

TRANSFORMER_IDS: dict[str, int] = {
    # Semantic Altering Transformers (IDs 1-4)
    "AOM": 1,  # Arithmetic Operators Misuse
    "BLM": 2,  # Boolean Literals Misuse
    "LOM": 3,  # Logical Operators Misuse
    "COM": 4,  # Comparison Operators Misuse
    # Semantic Preserving Transformers (IDs 5-9)
    "RV": 5,  # Rename Variables
    "DCI_V1": 6,  # Dead Code Insertion V1
    "DCI_V2": 7,  # Dead Code Insertion V2
    "OS": 8,  # Operand Swap
    "LT": 9,  # Loop Transformation
}

ID_TO_TRANSFORMER: dict[int, str] = {v: k for k, v in TRANSFORMER_IDS.items()}


def get_transformer_id(name: str) -> int:
    if name not in TRANSFORMER_IDS:
        raise KeyError(f"Unknown transformer name: {name}")
    return TRANSFORMER_IDS[name]


__all__ = ["TRANSFORMER_IDS", "ID_TO_TRANSFORMER", "get_transformer_id"]