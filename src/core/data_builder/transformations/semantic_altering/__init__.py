from typing import List

from .Arithmetic_Operators_Misuse import AOM
from .Boolean_Literals_Misuse import BLM
from .Logical_Operators_Misuse import LOM
from .Comparison_Operators_Misuse import COM
from ..base import Transformer

ALTERING: List[Transformer] = [AOM, BLM, LOM, COM]

__all__ = [
    "AOM",
    "BLM",
    "LOM",
    "COM",
]