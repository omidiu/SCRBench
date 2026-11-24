import ast
import astor

class IOMTransformer(ast.NodeTransformer):
    def visit_Compare(self, node):
        self.generic_visit(node)
        if isinstance(node.ops[0], ast.Is):
            node.ops[0] = ast.IsNot()
        elif isinstance(node.ops[0], ast.IsNot):
            node.ops[0] = ast.Is()
        return node

def apply_iom(code):
    tree = ast.parse(code)
    transformed_tree = IOMTransformer().visit(tree)
    return astor.to_source(transformed_tree)
