"""Base classes for code transformations."""
from dataclasses import dataclass
from typing import Callable, List
from enum import Enum


class TransformationType(Enum):
    """Type of transformation: semantic altering or preserving."""
    ALTERING = 'altering'
    PRESERVING = 'preserving'


@dataclass(frozen=True)
class Transformer:
    """
    A code transformer specification.
    
    Attributes:
        id: Unique identifier (starts from 1)
        name: Short identifier for the transformer (e.g., "AOM", "RV")
        short_description: Brief one-line description of the transformation
        long_description: Detailed description of what the transformation does
        operational_spec: Original concise specification of how the transformation operates.
        operational_spec_v2: Expanded specification (scope, procedure, and example).
        type: Type of transformation (altering or preserving)
        apply: Function that applies the transformation to code
    """
    id: int
    name: str
    short_description: str
    long_description: str
    operational_spec: str
    operational_spec_v2: str
    type: TransformationType
    apply: Callable[[str], str]
    
    def __call__(self, code: str) -> str:
        """Allow transformer to be called directly."""
        return self.apply(code)


def get_transformations_by_type(transformation_type: TransformationType) -> List[Transformer]:
    """
    Get list of transformations by type.
    
    Args:
        transformation_type: The type of transformation to retrieve
        
    Returns:
        List of Transformer instances of the specified type
        
    Raises:
        ValueError: If transformation_type is unknown
    """
    # Import here to avoid circular imports
    from .semantic_altering import ALTERING
    from .semantic_preserving import PRESERVING
    
    if transformation_type == TransformationType.PRESERVING:
        return PRESERVING
    elif transformation_type == TransformationType.ALTERING:
        return ALTERING
    else:
        raise ValueError(f"Unknown transformation type: {transformation_type}")

