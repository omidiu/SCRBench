import ast
import astor
from ..base import Transformer, TransformationType
from core.data_builder.transformations.config import get_transformer_id

class OSTransformer(ast.NodeTransformer):
    def visit_Compare(self, node):
        self.generic_visit(node)
        op_map = {
            ast.Gt: ast.Lt,
            ast.Lt: ast.Gt,
            ast.GtE: ast.LtE,
            ast.LtE: ast.GtE,
        }
        if isinstance(node.left, ast.Name) and isinstance(node.comparators[0], ast.Name):
            op = type(node.ops[0])
            if op in op_map:
                node.left, node.comparators[0] = node.comparators[0], node.left
                node.ops[0] = op_map[op]()
        return node

def apply_os(code):
    tree = ast.parse(code)
    transformed = OSTransformer().visit(tree)
    return astor.to_source(transformed)

# Transformer instance
OS = Transformer(
    id=get_transformer_id("OS"),
    name="OS",
    short_description="Swaps operands in comparison expressions and adjusts operators",
    long_description=(
        "Operand Swap (OS) transformation swaps the left and right operands in comparison "
        "expressions and adjusts the comparison operator accordingly to preserve semantics. "
        "For example, 'a > b' becomes 'b < a', 'a >= b' becomes 'b <= a'. This only applies "
        "to comparisons where both operands are variable names (ast.Name nodes). The "
        "transformation maintains logical equivalence while changing code structure."
    ),
    operational_spec=(
        "parse code to AST. Visit comparison expressions where both sides are variable names; "
        "swap left and right operands and flip the operator to preserve meaning (e.g. a>b becomes "
        "b<a). Applies to <, >, <=, >=; skips == and !=. Convert modified AST back to source code."
    ),
    operational_spec_v2=(
        "Scope: Compare nodes where both operands are simple names (ast.Name); operators >, <, "
        ">=, <= only (equality == and != are skipped). "
        "Procedure: parse to AST; for each qualifying comparison, swap left and right operands and "
        "replace the operator with the equivalent flipped form (>↔<, >=↔<=); unparses AST back to "
        "source. "
        "Example: `if a > b:` → `if b < a:`; `while low <= high:` → `while high >= low:`."
    ),
    type=TransformationType.PRESERVING,
    apply=apply_os
)
