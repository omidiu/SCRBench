import ast
import astor


from ..base import Transformer, TransformationType
from core.data_builder.transformations.config import get_transformer_id

class LTTransformer(ast.NodeTransformer):
    def visit_For(self, node):
        if not isinstance(node.iter, ast.Call) or not isinstance(node.iter.func, ast.Name) or node.iter.func.id != 'range':
            return node

        range_args = node.iter.args
        if len(range_args) == 1:
            start = ast.Constant(value=0)
            end = range_args[0]
            step = ast.Constant(value=1)
        elif len(range_args) == 2:
            start, end = range_args
            step = ast.Constant(value=1)
        elif len(range_args) == 3:
            start, end, step = range_args
        else:
            return node

        counter = node.target
        init = ast.Assign(targets=[counter], value=start)

        # Set compare_op and update_op based on static step
        if isinstance(step, ast.Constant) and isinstance(step.value, int):
            if step.value > 0:
                compare_op = ast.Lt()
                update_op = ast.Add()
            else:
                compare_op = ast.Gt()
                update_op = ast.Sub()
            test = ast.Compare(left=counter, ops=[compare_op], comparators=[end])
        else:
            # Dynamic step (we assume it's int-valued)
            test = ast.IfExp(
                test=ast.Compare(left=step, ops=[ast.Gt()], comparators=[ast.Constant(value=0)]),
                body=ast.Compare(left=counter, ops=[ast.Lt()], comparators=[end]),
                orelse=ast.Compare(left=counter, ops=[ast.Gt()], comparators=[end])
            )
            update_op = None  # Needs runtime decision

        # Increment
        increment = ast.AugAssign(
            target=counter,
            op=update_op if update_op else ast.Add(),  # default to Add if dynamic
            value=step
        )

        while_body = node.body + [increment]
        new_while = ast.While(test=test, body=while_body, orelse=node.orelse)
        return [init, new_while]


def apply_lt(code):
    tree = ast.parse(code)
    tree = LTTransformer().visit(tree)
    tree = ast.fix_missing_locations(tree)
    return astor.to_source(tree)

# Transformer instance
LT = Transformer(
    id=get_transformer_id("LT"),
    name="LT",
    short_description="Converts for-loops with range() to equivalent while-loops",
    long_description=(
        "Loop Transformation (LT) converts for-loops that use range() into semantically "
        "equivalent while-loops. The transformation extracts the start, end, and step values "
        "from range(), creates an initialization statement, builds an appropriate loop "
        "condition, and adds an increment/decrement statement. It handles both positive and "
        "negative steps, maintaining the exact iteration behavior while changing loop structure."
    ),
    operational_spec=(
        "parse code to AST. Find each 'for i in range(...)' loop. Convert it to an equivalent "
        "while loop: (1) before the loop, add an assignment initializing the loop variable to the "
        "range start; (2) replace the for with a while whose condition checks the variable against "
        "the range end (< if step positive, > if negative); (3) the while body is the original "
        "loop body plus an increment/decrement of the variable at the end. Supports range(n), "
        "range(start,end), and range(start,end,step). Convert modified AST back to source code."
    ),
    operational_spec_v2=(
        "Scope: `for` loops whose iterator is a direct `range(...)` call (one, two, or three "
        "arguments); other loop forms are left unchanged. "
        "Procedure: parse to AST; for each matching For node, emit (1) an assignment initializing "
        "the loop variable to the range start, (2) a While with end test (< for positive step, > "
        "for negative step), (3) a body equal to the original loop body plus an increment/decrement "
        "by the step; unparses AST back to source. "
        "Example: `for i in range(3): body` → `i = 0` then `while i < 3: body; i += 1`."
    ),
    type=TransformationType.PRESERVING,
    apply=apply_lt
)
