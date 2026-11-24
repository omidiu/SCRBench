"""Utility functions for transformations."""
from .base import TransformationType
from .semantic_altering import ALTERING


def is_altering_transformation(transformation_id: int) -> bool:
    """
    Check if a transformation ID corresponds to an altering transformation.

    Args:
        transformation_id: The ID of the transformation to check

    Returns:
        True if the transformation is altering, False otherwise
    """
    # Check if the ID exists in the ALTERING transformations list
    return any(t.id == transformation_id for t in ALTERING)

