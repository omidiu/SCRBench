import ast
import astor
from ..base import Transformer, TransformationType
from core.data_builder.transformations.config import get_transformer_id

class LOMTransformer(ast.NodeTransformer):
    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.And):
            node.op = ast.Or()
        elif isinstance(node.op, ast.Or):
            node.op = ast.And()
        return node

def apply_lom(code):
    tree = ast.parse(code)
    transformed_tree = LOMTransformer().visit(tree)
    return astor.to_source(transformed_tree)

# Transformer instance
LOM = Transformer(
    id=get_transformer_id("LOM"),
    name="LOM",
    short_description="Swaps logical AND and OR operators",
    long_description=(
        "Logical Operators Misuse (LOM) transformation replaces logical operators with "
        "their opposites: 'and' becomes 'or' and 'or' becomes 'and'. This transformation "
        "fundamentally changes the logic of boolean expressions and can cause significant "
        "semantic alterations in conditional statements and boolean evaluations."
    ),
    operational_spec=(
        "parse code to AST. Visit all logical expressions (and/or); replace 'and' with 'or' and "
        "'or' with 'and'. Convert modified AST back to source code."
    ),
    operational_spec_v2=(
        "Scope: logical boolean operators in BoolOp nodes (`and`, `or`) only; does not change "
        "arithmetic, comparisons, or `True`/`False` literals. "
        "Procedure: parse to AST; visit each BoolOp; swap `and`↔`or` on that node; unparses AST "
        "back to source. "
        "Example: `if x > 0 and x < 10:` → `if x > 0 or x < 10:`."
    ),
    type=TransformationType.ALTERING,
    apply=apply_lom
)
