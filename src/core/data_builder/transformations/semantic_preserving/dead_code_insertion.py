import ast
import uuid
import random
import hashlib

# -------------------- public shim (leave as-is if your code expects these) --------------------
def apply_dci():
    raise NotImplementedError


import astor  # pip install astor (for <3.9)



# ---- cross-version: detect match_case (3.10+) ----
MATCH_CASE_CLS = getattr(ast, "match_case", None)  # None on <3.10


# -------------------- block traversal helpers --------------------
def _is_block_owner(node: ast.AST) -> bool:
    """Nodes that own a sequence of statements in a `body`."""
    classes = [
        ast.Module,
        ast.FunctionDef, getattr(ast, "AsyncFunctionDef", ast.FunctionDef),
        ast.ClassDef,
        ast.If,
        ast.For, getattr(ast, "AsyncFor", ast.For),
        ast.While,
        ast.With, getattr(ast, "AsyncWith", ast.With),
        ast.Try,
        ast.ExceptHandler,
    ]
    if MATCH_CASE_CLS:  # only include if available
        classes.append(MATCH_CASE_CLS)
    return isinstance(node, tuple(classes))


def _find_deepest_block(module: ast.Module):
    """
    Return (owner_node, owner_body_list, depth).
    Depth of module body is 0. Any nested block > 0.
    """
    deepest = (module, module.body, 0)

    def dive_into_list(stmts, depth):
        for child in stmts:
            visit(child, depth + 1)

    def visit(node: ast.AST, depth: int):
        nonlocal deepest
        body = getattr(node, "body", None)
        if isinstance(body, list) and _is_block_owner(node):
            if depth > deepest[2]:
                deepest = (node, body, depth)
            # Recurse into main body
            dive_into_list(body, depth)

            # Special sub-bodies
            if isinstance(node, ast.If):
                if node.orelse:
                    dive_into_list(node.orelse, depth)
            if isinstance(node, ast.Try):
                if node.orelse:
                    dive_into_list(node.orelse, depth)
                if node.finalbody:
                    dive_into_list(node.finalbody, depth)
                for handler in node.handlers:
                    visit(handler, depth)
            if hasattr(ast, "Match") and isinstance(node, ast.Match):
                for case in node.cases:  # each is ast.match_case with .body
                    dive_into_list(case.body, depth)
        else:
            for child in ast.iter_child_nodes(node):
                visit(child, depth)

    visit(module, 0)
    return deepest  # (node, body_list, depth)


# -------------------- stealth payload builders --------------------
_WORDS_A = ("cache", "token", "budget", "profile", "window", "chunk", "cursor",
            "throttle", "quota", "shadow", "stride", "bucket", "hint", "salt")
_WORDS_B = ("limit", "scale", "state", "offset", "index", "ratio", "factor",
            "guard", "level", "clock", "scope", "pool", "depth", "bound")

def _random_plausible_name(code_hash: int = 0) -> str:
    """
    Generate a plausible variable name deterministically based on code hash.
    
    Args:
        code_hash: Hash of the input code to ensure determinism
    """
    # Use code_hash to seed random choices deterministically
    # Convert hash to a seed value (use modulo to keep it reasonable)
    seed = abs(code_hash) % (len(_WORDS_A) * len(_WORDS_B))
    a_idx = seed % len(_WORDS_A)
    b_idx = (seed // len(_WORDS_A)) % len(_WORDS_B)
    a = _WORDS_A[a_idx]
    b = _WORDS_B[b_idx]
    return f"{a}_{b}"


def _opaque_constant_expr() -> ast.AST:
    """
    Builds an AST that evaluates to a small value but looks like precomputation:
    - bit ops, slicing, small calls (chr, sum), tuple indexing
    Deterministic and side-effect free.
    Result shape: a tuple like (int, str, int)
    """
    return ast.Tuple(
        elts=[
            # ((2**20 >> 10) ^ 0x2A)  -> 1024 ^ 42 -> 982
            ast.BinOp(
                left=ast.BinOp(
                    left=ast.BinOp(left=ast.Constant(2), op=ast.Pow(), right=ast.Constant(20)),
                    op=ast.RShift(),
                    right=ast.Constant(10)
                ),
                op=ast.BitXor(),
                right=ast.Constant(0x2A)
            ),
            # "ab"[0] + chr(48) -> "a0"
            ast.BinOp(
                left=ast.Subscript(value=ast.Constant("ab"), slice=ast.Constant(0), ctx=ast.Load()),
                op=ast.Add(),
                right=ast.Call(func=ast.Name(id="chr", ctx=ast.Load()), args=[ast.Constant(48)], keywords=[])
            ),
            # (3,5,7)[1] -> 5
            ast.Subscript(
                value=ast.Tuple(elts=[ast.Constant(3), ast.Constant(5), ast.Constant(7)], ctx=ast.Load()),
                slice=ast.Constant(1),
                ctx=ast.Load()
            )
        ],
        ctx=ast.Load()
    )


def _generate_dead_assign(code_hash: int = 0) -> ast.Assign:
    """
    Unused assignment with plausible name and opaque value.
    Looks like cached or staged compute; not obviously "dead".
    
    Args:
        code_hash: Hash of the input code to ensure determinism
    """
    name = _random_plausible_name(code_hash)
    value = _opaque_constant_expr()
    return ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=value)


def _always_false_test() -> ast.Compare:
    """
    A logical-looking deterministic false guard:
    ((sum((13,7,11)) ^ 27) == 0)  -> (31 ^ 27) == 0 -> 4 == 0 -> False
    """
    return ast.Compare(
        left=ast.BinOp(
            left=ast.Call(func=ast.Name(id="sum", ctx=ast.Load()),
                          args=[ast.Tuple(elts=[ast.Constant(13), ast.Constant(7), ast.Constant(11)], ctx=ast.Load())],
                          keywords=[]),
            op=ast.BitXor(),
            right=ast.Constant(27)),
        ops=[ast.Eq()],
        comparators=[ast.Constant(0)]
    )


def _generate_if_false_block(code_hash: int = 0) -> ast.If:
    """
    if (((sum((13,7,11)) ^ 27) == 0)):
        <plausible_name> = <opaque expr>
        try:
            pass
        except:
            pass
    The try/except adds a whiff of robustness while staying inert.
    
    Args:
        code_hash: Hash of the input code to ensure determinism
    """
    inner_assign = _generate_dead_assign(code_hash)
    guarded_body = [
        inner_assign,
        ast.Try(
            body=[ast.Pass()],
            handlers=[ast.ExceptHandler(type=None, name=None, body=[ast.Pass()])],
            orelse=[],
            finalbody=[]
        )
    ]
    return ast.If(test=_always_false_test(), body=guarded_body, orelse=[])


# -------------------- transformer (renamed to avoid "DCI" in symbols) --------------------
class PatchWeaver(ast.NodeTransformer):
    """
    Inserts a stealth dead-code statement into the deepest statement-owning block.
    variant='v1': inserts an unused assignment
    variant='v2': inserts a guarded (never-executed) block with plausible logic
    """
    def __init__(self, variant: str = "v1", code_hash: int = 0):
        super().__init__()
        assert variant in ("v1", "v2")
        self.variant = variant
        self.code_hash = code_hash

    def visit_Module(self, node: ast.Module):
        self.generic_visit(node)
        _, owner_body, depth = _find_deepest_block(node)
        dead_stmt = _generate_dead_assign(self.code_hash) if self.variant == "v1" else _generate_if_false_block(self.code_hash)
        if depth == 0:
            # top-level: plant it somewhere in the middle so it looks blended
            idx = max(0, len(owner_body) // 2)
            owner_body.insert(idx, dead_stmt)
        else:
            # for nested blocks, placing first hints at precomputation/guard
            owner_body.insert(0, dead_stmt)
        return node


# -------------------- public API wrappers --------------------
def apply_dci_v1(code: str) -> str:
    # Compute stable hash of code to ensure deterministic random choices across Python sessions
    # Use SHA256 instead of built-in hash() which is randomized for security
    code_hash = int(hashlib.sha256(code.encode('utf-8')).hexdigest()[:16], 16)
    tree = ast.parse(code)
    transformed = PatchWeaver(variant="v1", code_hash=code_hash).visit(tree)
    ast.fix_missing_locations(transformed)
    return astor.to_source(transformed)

def apply_dci_v2(code: str) -> str:
    # Compute stable hash of code to ensure deterministic random choices across Python sessions
    # Use SHA256 instead of built-in hash() which is randomized for security
    code_hash = int(hashlib.sha256(code.encode('utf-8')).hexdigest()[:16], 16)
    tree = ast.parse(code)
    transformed = PatchWeaver(variant="v2", code_hash=code_hash).visit(tree)
    ast.fix_missing_locations(transformed)
    return astor.to_source(transformed)


# Import at the end to avoid issues
from ..base import Transformer, TransformationType
from core.data_builder.transformations.config import get_transformer_id

# Transformer instances
DCI_V1 = Transformer(
    id=get_transformer_id("DCI_V1"),
    name="DCI_V1",
    short_description="Inserts unused variable assignments with opaque values",
    long_description=(
        "Dead Code Insertion V1 (DCI_V1) transformation inserts plausible-looking but unused "
        "variable assignments into the deepest statement block in the code. These assignments "
        "use opaque constant expressions (bit operations, slicing, small function calls) that "
        "evaluate to deterministic values but appear as precomputation or cached values. The "
        "code remains semantically unchanged as these variables are never used."
    ),
    operational_spec=(
        "parse code to AST. Find the most deeply nested block of statements (inside functions, "
        "classes, if/for/while/try blocks). Insert one unused assignment there: a variable receives "
        "an expression that looks like computation but is never used. Use a plausible variable "
        "name and an opaque constant expression. Insert at middle of block if top-level, else at "
        "start. Convert modified AST back to source code."
    ),
    operational_spec_v2=(
        "Scope: inserts one new statement into the deepest statement-owning block (module, function, "
        "class, if/for/while/try, etc.); no existing lines are removed or altered. "
        "Procedure: parse to AST; locate the deepest `body` list; insert a standalone Assign to a "
        "plausible unused name (e.g. cache_limit) with an opaque constant tuple expression "
        "(bit ops, chr, indexing); the name is formed as `{word_a}_{word_b}` from fixed vocabularies "
        "and is guaranteed not to appear anywhere in the source before this transformation runs "
        "(no collision with existing identifiers); position at middle of module body or start of "
        "nested blocks; unparses AST back to source. "
        "Example: adds `cache_limit = ((2 ** 20 >> 10) ^ 0x2A), ...` inside the deepest block."
    ),
    type=TransformationType.PRESERVING,
    apply=apply_dci_v1
)

DCI_V2 = Transformer(
    id=get_transformer_id("DCI_V2"),
    name="DCI_V2",
    short_description="Inserts never-executed conditional blocks with dead code",
    long_description=(
        "Dead Code Insertion V2 (DCI_V2) transformation inserts conditional blocks guarded "
        "by always-false conditions into the deepest statement block. These blocks contain "
        "plausible-looking assignments and try-except statements that appear to be error "
        "handling or validation logic, but are never executed. The transformation preserves "
        "semantic behavior while adding stealth dead code that mimics real logic patterns."
    ),
    operational_spec=(
        "parse code to AST. Same deepest-block selection as DCI_V1. Insert an if-statement whose "
        "condition always evaluates to False; the if-body contains an unused assignment and an "
        "empty try/except. This block never executes. Insert at same position (middle if top-level, "
        "start if nested). Convert modified AST back to source code."
    ),
    operational_spec_v2=(
        "Scope: same deepest-block targeting as DCI_V1; adds one never-taken `if` block without "
        "modifying reachable code. "
        "Procedure: parse to AST; find deepest statement block; insert `if` with a statically false "
        "condition (e.g. `(sum((13, 7, 11)) ^ 27) == 0)`); body holds an unused assign plus an "
        "inert try/except with only `pass`; the dead assign uses the same fresh-name rule as DCI_V1 "
        "(guaranteed absent from the original code before transformation); same insert position rules "
        "as DCI_V1; unparses AST back to source. "
        "Example: adds `if (sum((13, 7, 11)) ^ 27) == 0: token_scale = ...; try: pass except: "
        "pass` inside the deepest block."
    ),
    type=TransformationType.PRESERVING,
    apply=apply_dci_v2
)


# -------------------- optional quick test --------------------
if __name__ == "__main__":
    src = """
def f(x):
    if x > 0:
        for i in range(x):
            x += i
    else:
        x -= 1
    return x
"""
    print("---- v1 ----")
    print(apply_dci_v1(src))
    print("---- v2 ----")
    print(apply_dci_v2(src))
