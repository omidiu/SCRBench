from ..base import Transformer, TransformationType
from core.data_builder.transformations.config import get_transformer_id

def apply_blm(code):
    # Replace first occurrence of True/False
    if "True" in code:
        return code.replace("True", "False", 1)
    elif "False" in code:
        return code.replace("False", "True", 1)
    return code

# Transformer instance
BLM = Transformer(
    id=get_transformer_id("BLM"),
    name="BLM",
    short_description="Replaces True with False or vice versa",
    long_description=(
        "Boolean Literals Misuse (BLM) transformation replaces the first occurrence of "
        "boolean literals in code. If 'True' is found, it is replaced with 'False', and "
        "if 'False' is found, it is replaced with 'True'. This can significantly alter "
        "the semantic behavior of conditional statements and boolean expressions."
    ),
    operational_spec=(
        "Scan source code as string. If 'True' occurs, replace its first occurrence with 'False'; "
        "else if 'False' occurs, replace its first occurrence with 'True'; else return unchanged."
    ),
    operational_spec_v2=(
        "Scope: boolean literals `True` and `False` in source text only; at most one replacement per "
        "application (first match in file order). Does not change `and`/`or`, comparisons, or "
        "non-literal booleans. "
        "Procedure: scan source as text; if `True` appears, replace the first `True` with `False`; "
        "else if `False` appears, replace the first `False` with `True`; otherwise leave code unchanged. "
        "Example: `if True: return 1` → `if False: return 1`."
    ),
    type=TransformationType.ALTERING,
    apply=apply_blm
)
