import ast
import astor
from ..base import Transformer, TransformationType
from core.data_builder.transformations.config import get_transformer_id

"""
Arithmetic Operators Misuse (AOM) transformation module.

This module provides functionality to transform arithmetic operations in Python code
by replacing operators with their opposites:
- Addition (+) becomes Subtraction (-)
- Subtraction (-) becomes Addition (+)
- Multiplication (*) becomes Division (/)
- Division (/) becomes Multiplication (*)

This transformation applies to both binary operations and augmented assignments.

Example:
    >>> code = '''
    ... def function():
    ...     x = a + b
    ...     y = c * d
    ...     z += 10
    ...     w /= 2
    ...     return x + y
    ... '''
    >>> transformed = apply_aom(code)
    >>> print(transformed)
    def function():
        x = a - b
        y = c / d
        z -= 10
        w *= 2
        return x - y
"""

class AOMTransformer(ast.NodeTransformer):
    def visit_BinOp(self, node):
        self.generic_visit(node)
        op_map = {ast.Add: ast.Sub, ast.Sub: ast.Add,
                  ast.Mult: ast.Div, ast.Div: ast.Mult}
        op_type = type(node.op)
        if op_type in op_map:
            node.op = op_map[op_type]()
        return node

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        aug_op_map = {ast.Add: ast.Sub, ast.Sub: ast.Add,
                      ast.Mult: ast.Div, ast.Div: ast.Mult}
        op_type = type(node.op)
        if op_type in aug_op_map:
            node.op = aug_op_map[op_type]()
        return node

def apply_aom(code):
    tree = ast.parse(code)
    transformed_tree = AOMTransformer().visit(tree)
    return astor.to_source(transformed_tree)

# Transformer instance
AOM = Transformer(
    id=get_transformer_id("AOM"),
    name="AOM",
    short_description="Replaces arithmetic operators with their opposites",
    long_description=(
        "Arithmetic Operators Misuse (AOM) transformation replaces arithmetic operators "
        "with their opposites: addition (+) becomes subtraction (-), subtraction (-) becomes "
        "addition (+), multiplication (*) becomes division (/), and division (/) becomes "
        "multiplication (*). This applies to both binary operations and augmented assignments."
    ),
    operational_spec=(
        "parse code to AST. Visit all binary arithmetic operations (e.g. a+b, x*y) and augmented "
        "assignments (e.g. +=, /=); replace each operator with its opposite: +↔-, *↔/. "
        "Convert modified AST back to source code."
    ),
    operational_spec_v2=(
        "Scope: binary arithmetic operators (+, -, *, /) and augmented assignments (+=, -=, *=, /=) only; "
        "does not modify comparisons, logical operators, boolean literals, or variable names. "
        "Procedure: parse to AST; visit every BinOp and AugAssign node; replace each operator with its "
        "paired opposite (+↔-, -↔+, *↔/, /↔*); unparses AST back to source. "
        "Example: `x = a + b` → `x = a - b`; `z += 10` → `z -= 10`."
    ),
    type=TransformationType.ALTERING,
    apply=apply_aom
)
