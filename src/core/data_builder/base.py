import pandas as pd

from typing import Any, Dict

from pydantic import BaseModel, Field, field_validator




class ProgramPairsByLength(BaseModel):
    """
    Pydantic model for a dictionary mapping k dataset keys (k1, k2, k3, k4, k5)
    to their corresponding pandas DataFrames.
    """
    k1: Any = Field(..., description="DataFrame for k=1")
    k2: Any = Field(..., description="DataFrame for k=2")
    k3: Any = Field(..., description="DataFrame for k=3")
    k4: Any = Field(..., description="DataFrame for k=4")
    k5: Any = Field(..., description="DataFrame for k=5")

    @field_validator('k1', 'k2', 'k3', 'k4', 'k5')
    @classmethod
    def validate_dataframe(cls, v):
        """Validate that the value is a pandas DataFrame."""
        if not isinstance(v, pd.DataFrame):
            raise ValueError(f"Expected pandas DataFrame, got {type(v)}")
        return v

    @classmethod
    def from_dict(cls, data: Dict[str, pd.DataFrame]) -> "ProgramPairsByLength":
        """Create ProgramPairsByLength from a dictionary."""
        return cls(**data)

    def to_dict(self) -> Dict[str, pd.DataFrame]:
        """Convert ProgramPairsByLength to a dictionary."""
        return {
            "k1": self.k1,
            "k2": self.k2,
            "k3": self.k3,
            "k4": self.k4,
            "k5": self.k5,
        }