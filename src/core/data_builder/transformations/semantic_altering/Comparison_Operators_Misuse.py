import ast
import astor
from ..base import Transformer, TransformationType
from core.data_builder.transformations.config import get_transformer_id

class COMTransformer(ast.NodeTransformer):
    def visit_Compare(self, node):
        self.generic_visit(node)
        comp_map = {
            ast.Gt: ast.Lt, ast.Lt: ast.Gt,
            ast.GtE: ast.LtE, ast.LtE: ast.GtE,
            ast.Eq: ast.NotEq, ast.NotEq: ast.Eq
        }
        if type(node.ops[0]) in comp_map:
            node.ops[0] = comp_map[type(node.ops[0])]()
        return node

def apply_com(code):
    tree = ast.parse(code)
    transformed_tree = COMTransformer().visit(tree)
    return astor.to_source(transformed_tree)

# Transformer instance
COM = Transformer(
    id=get_transformer_id("COM"),
    name="COM",
    short_description="Replaces comparison operators with their opposites",
    long_description=(
        "Comparison Operators Misuse (COM) transformation replaces comparison operators "
        "with their opposites: greater than (>) becomes less than (<), less than (<) becomes "
        "greater than (>), greater than or equal (>=) becomes less than or equal (<=), "
        "less than or equal (<=) becomes greater than or equal (>=), equal (==) becomes "
        "not equal (!=), and not equal (!=) becomes equal (==). This alters conditional logic."
    ),
    operational_spec=(
        "parse code to AST. Visit all comparison expressions; replace each operator with its "
        "opposite: >↔<, >=↔<=, ==↔!=. Convert modified AST back to source code."
    ),
    operational_spec_v2=(
        "Scope: comparison operators in Compare nodes: >, <, >=, <=, ==, != (first operator in a "
        "compare chain only); does not swap operands or change `in`/`is` tests. "
        "Procedure: parse to AST; visit Compare nodes; map each operator to its opposite "
        "(>↔<, >=↔<=, ==↔!=); unparses AST back to source. "
        "Example: `if a == b:` → `if a != b:`; `while n > 0:` → `while n < 0:`."
    ),
    type=TransformationType.ALTERING,
    apply=apply_com
)
