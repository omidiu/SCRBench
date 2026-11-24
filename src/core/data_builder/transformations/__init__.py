from typing import List
from .base import Transformer, TransformationType
 
from .semantic_altering import AOM
from .semantic_altering import BLM
from .semantic_altering import LOM
from .semantic_altering import COM
from .semantic_preserving.rename_variables import RV
from .semantic_preserving.operand_swap import OS
from .semantic_preserving.loop_transformations import LT
from .semantic_preserving.dead_code_insertion import DCI_V1, DCI_V2


from .base import Transformer

from .semantic_altering import ALTERING
from .semantic_preserving import PRESERVING

ALL_TRANSFORMATIONS: List[Transformer] = ALTERING + PRESERVING

# Registry (name -> transformer) for convenient lookups and for tests.
REG = {t.name: t for t in ALL_TRANSFORMATIONS}


TRANSFORMER_ID_TO_APPLY_FN = {
    transformer.id: transformer.apply for transformer in ALL_TRANSFORMATIONS
}

from .config import ID_TO_TRANSFORMER, TRANSFORMER_IDS, get_transformer_id

PRESERVING_TRANSFORMATION_IDS = set(t.id for t in PRESERVING)
ALTERING_TRANSFORMATION_IDS = set(t.id for t in ALTERING)





__all__ = [
    # Types
    "Transformer",
    "TransformationType",
    # Individual transformers
    "AOM",
    "BLM",
    "LOM",
    "COM",
    "RV",
    "OS",
    "LT",
    "DCI_V1",
    "DCI_V2",
    # Groups / registry
    "ALTERING",
    "PRESERVING",
    "ALL_TRANSFORMATIONS",
    "REG",
    "TRANSFORMER_ID_TO_APPLY_FN",
    # ID config helpers
    "TRANSFORMER_IDS",
    "ID_TO_TRANSFORMER",
    "get_transformer_id",
]