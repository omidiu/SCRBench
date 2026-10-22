from typing import List

from .rename_variables import RV
from .operand_swap import OS
from .loop_transformations import LT
from .dead_code_insertion import DCI_V1, DCI_V2

from ..base import Transformer

PRESERVING: List[Transformer] = [RV, DCI_V1, DCI_V2, OS, LT]


__all__ = [
    # Transformer instances
    "RV",
    "OS",
    "LT",
    "DCI_V1",
    "DCI_V2",
]