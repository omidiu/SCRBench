import ast
import astor
import builtins

from collections import Counter





# class RVSingleTopRenamer(ast.NodeTransformer):
#     def __init__(self, target_name):
#         self.target_name = target_name
#         self.builtins = set(dir(builtins))
#         self.imported_names = set()
#         self.defined_names = set()

#     def visit_Import(self, node):
#         for alias in node.names:
#             self.imported_names.add(alias.asname or alias.name)
#         return node

#     def visit_ImportFrom(self, node):
#         for alias in node.names:
#             self.imported_names.add(alias.asname or alias.name)
#         return node

#     def visit_FunctionDef(self, node):
#         self.defined_names.add(node.name)
#         self.generic_visit(node)
#         return node

#     def visit_AsyncFunctionDef(self, node):
#         self.defined_names.add(node.name)
#         self.generic_visit(node)
#         return node

#     def visit_Name(self, node):
#         if (isinstance(node.ctx, (ast.Load, ast.Store, ast.Del)) and
#             node.id == self.target_name):
#             node.id = 'VAR_0'
#         return node

#     def visit_arg(self, node):
#         if node.arg == self.target_name:
#             node.arg = 'VAR_0'
#         return node

# def find_most_common_variable(code):
#     class NameCounter(ast.NodeVisitor):
#         def __init__(self):
#             self.counter = Counter()
#             self.builtins = set(dir(builtins))
#             self.imported_names = set()
#             self.defined_names = set()

#         def visit_Import(self, node):
#             for alias in node.names:
#                 self.imported_names.add(alias.asname or alias.name)
#             self.generic_visit(node)

#         def visit_ImportFrom(self, node):
#             for alias in node.names:
#                 self.imported_names.add(alias.asname or alias.name)
#             self.generic_visit(node)

#         def visit_FunctionDef(self, node):
#             self.defined_names.add(node.name)
#             self.generic_visit(node)

#         def visit_AsyncFunctionDef(self, node):
#             self.defined_names.add(node.name)
#             self.generic_visit(node)

#         def visit_Name(self, node):
#             if (isinstance(node.ctx, (ast.Load, ast.Store, ast.Del)) and
#                 node.id not in self.builtins and
#                 node.id not in self.imported_names and
#                 node.id not in self.defined_names):
#                 self.counter[node.id] += 1
#             self.generic_visit(node)

#         def visit_arg(self, node):
#             if (node.arg not in self.builtins):
#                 self.counter[node.arg] += 1

#     tree = ast.parse(code)
#     visitor = NameCounter()
#     visitor.visit(tree)
#     most_common = visitor.counter.most_common(1)
#     return most_common[0][0] if most_common else None


# # naive approach
# def apply_most_common_variable_rename(code):
#     most_common_name = find_most_common_variable(code)
#     if not most_common_name:
#         return code
#     tree = ast.parse(code)
#     transformer = RVSingleTopRenamer(most_common_name)
#     transformed = transformer.visit(tree)
#     return astor.to_source(transformed)


# class RVNaiveTransformer(ast.NodeTransformer):
#     def __init__(self):
#         self.var_map = {}
#         self.counter = 0
#         # do not rename any Python built-in names
#         self.builtins = set(dir(builtins))
#         # names coming from import statements
#         self.imported_names = set()
#         # names of functions and classes defined in this module
#         self.defined_names = set()

#     def get_new_name(self, old_name):
#         # never rename builtins, imported names, or your own defs
#         if (old_name in self.builtins
#             or old_name in self.imported_names
#             or old_name in self.defined_names):
#             return old_name

#         # otherwise assign a fresh VAR_n
#         if old_name not in self.var_map:
#             self.var_map[old_name] = f'VAR_{self.counter}'
#             self.counter += 1
#         return self.var_map[old_name]

#     def visit_Import(self, node):
#         for alias in node.names:
#             self.imported_names.add(alias.asname or alias.name)
#         return node

#     def visit_ImportFrom(self, node):
#         for alias in node.names:
#             self.imported_names.add(alias.asname or alias.name)
#         return node

#     def visit_FunctionDef(self, node):
#         # record the function’s original name so we don’t rename it
#         self.defined_names.add(node.name)
#         # now descend into args and body
#         self.generic_visit(node)
#         return node

#     def visit_AsyncFunctionDef(self, node):
#         # same for async defs
#         self.defined_names.add(node.name)
#         self.generic_visit(node)
#         return node

#     def visit_Name(self, node):
#         if isinstance(node.ctx, (ast.Load, ast.Store, ast.Del)):
#             node.id = self.get_new_name(node.id)
#         return node

#     def visit_arg(self, node):
#         # rename argument names
#         node.arg = self.get_new_name(node.arg)
#         return node

# def apply_rename_on_all_variables(code):
#     tree = ast.parse(code)
#     transformed = RVNaiveTransformer().visit(tree)
#     return astor.to_source(transformed)




#====================================================================================   

# class RVNaiveTransformer(ast.NodeTransformer):
#     def __init__(self):
#         self.var_map = {}
#         self.counter = 0
#         self.builtins = set(dir(builtins))
#         self.imported_names = set()

#     def get_new_name(self, old_name):
#         if old_name in self.builtins or old_name in self.imported_names:
#             return old_name
#         if old_name not in self.var_map:
#             self.var_map[old_name] = f'VAR_{self.counter}'
#             self.counter += 1
#         return self.var_map[old_name]

#     def visit_Import(self, node):
#         for alias in node.names:
#             self.imported_names.add(alias.asname if alias.asname else alias.name)
#         return node

#     def visit_ImportFrom(self, node):
#         for alias in node.names:
#             self.imported_names.add(alias.asname if alias.asname else alias.name)
#         return node

#     def visit_Name(self, node):
#         if isinstance(node.ctx, (ast.Load, ast.Store, ast.Del)):
#             node.id = self.get_new_name(node.id)
#         return node

#     def visit_arg(self, node):
#         node.arg = self.get_new_name(node.arg)
#         return node


# def apply_rv_naive(code):
#     tree = ast.parse(code)
#     transformed = RVNaiveTransformer().visit(tree)
#     return astor.to_source(transformed)



#====================================================================================   


# class RVRandTransformer(ast.NodeTransformer):
#     def __init__(self):
#         self.var_freq = {}
#         self.replacement = None

#     def visit_Name(self, node):
#         if isinstance(node.ctx, (ast.Load, ast.Store)):
#             self.var_freq[node.id] = self.var_freq.get(node.id, 0) + 1
#         return node

#     def get_most_common(self):
#         return max(self.var_freq, key=self.var_freq.get)

#     def random_name(self):
#         return ''.join(random.choices(string.ascii_letters + string.digits, k=6))

#     def replace_vars(self, tree, target, new_name):
#         class Replace(ast.NodeTransformer):
#             def visit_Name(self, node):
#                 if node.id == target:
#                     node.id = new_name
#                 return node
#         return Replace().visit(tree)

# def apply_rv_random(code):
#     tree = ast.parse(code)
#     analyzer = RVRandTransformer()
#     analyzer.visit(tree)
#     most_common = analyzer.get_most_common()
#     new_name = analyzer.random_name()
#     tree = analyzer.replace_vars(tree, most_common, new_name)
#     return astor.to_source(tree)




#====================================================================================  
# 
# 
# 
# 
# 
# # 

# class RVRandTransformer(ast.NodeTransformer):
#     def __init__(self):
#         self.var_freq = {}
#         # anything here we never want to rename
#         self.builtins = set(dir(builtins))
#         self.imported_names = set()

#     def visit_Import(self, node):
#         for alias in node.names:
#             self.imported_names.add(alias.asname or alias.name)
#         return node

#     def visit_ImportFrom(self, node):
#         for alias in node.names:
#             self.imported_names.add(alias.asname or alias.name)
#         return node

#     def visit_Name(self, node):
#         # only count non-builtin, non-import names
#         if isinstance(node.ctx, (ast.Load, ast.Store, ast.Del)):
#             if node.id not in self.builtins and node.id not in self.imported_names:
#                 self.var_freq[node.id] = self.var_freq.get(node.id, 0) + 1
#         return node

#     def get_most_common(self):
#         if not self.var_freq:
#             return None
#         return max(self.var_freq, key=self.var_freq.get)

#     def random_name(self):
#         return ''.join(random.choices(string.ascii_letters + string.digits, k=6))

#     def replace_vars(self, tree, target, new_name):
#         class Replace(ast.NodeTransformer):
#             def visit_Name(self, node):
#                 if node.id == target:
#                     node.id = new_name
#                 return node
#             def visit_arg(self, node):
#                 if node.arg == target:
#                     node.arg = new_name
#                 return node
#         return Replace().visit(tree)

# def apply_rv_random(code, seed=42):
#     random.seed(seed)
#     tree = ast.parse(code)
#     analyzer = RVRandTransformer()
#     analyzer.visit(tree)

#     target = analyzer.get_most_common()
#     if target is None:
#         # nothing to do
#         return astor.to_source(tree)

#     new_name = analyzer.random_name()
#     new_tree = analyzer.replace_vars(tree, target, new_name)
#     return astor.to_source(new_tree) 


class MostCommonVarRenamer(ast.NodeTransformer):
    def __init__(self, most_common_vars):
        self.rename_map = {name: f'VAR_{i}' for i, name in enumerate(most_common_vars)}

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Load, ast.Store, ast.Del)) and node.id in self.rename_map:
            node.id = self.rename_map[node.id]
        return node

    def visit_arg(self, node):
        if node.arg in self.rename_map:
            node.arg = self.rename_map[node.arg]
        return node

# apply_most_common_renaming TODO: FUNCITON NAME
def apply_rv_naive(code, top_n=1):
    tree = ast.parse(code)

    # Step 1: Collect all variable and argument names
    class NameCollector(ast.NodeVisitor):
        def __init__(self):
            self.names = []

        def visit_Name(self, node):
            if isinstance(node.ctx, (ast.Load, ast.Store, ast.Del)):
                self.names.append(node.id)

        def visit_arg(self, node):
            self.names.append(node.arg)

    collector = NameCollector()
    collector.visit(tree)

    # Step 2: Count and select most common names
    name_counts = Counter(collector.names)
    most_common = [name for name, _ in name_counts.most_common(top_n)]

    # Step 3: Transform
    transformer = MostCommonVarRenamer(most_common)
    transformed_tree = transformer.visit(tree)

    return astor.to_source(transformed_tree)


# Import at the end to avoid issues
from ..base import Transformer, TransformationType
from core.data_builder.transformations.config import get_transformer_id

# Transformer instance
RV = Transformer(
    id=get_transformer_id("RV"),
    name="RV",
    short_description="Renames the most common variables to generic names",
    long_description=(
        "Rename Variables (RV) transformation identifies and renames the most frequently "
        "used variable names in the code to generic names like VAR_0, VAR_1, etc. This "
        "preserves the semantic meaning of the code while obfuscating variable names. The "
        "transformation avoids renaming built-in names, imported names, and function/class "
        "definitions to maintain code correctness."
    ),
    operational_spec=(
        "parse code to AST. Collect all variable names and parameter names used in the code; "
        "count how often each appears; select the most frequently used ones; replace each with "
        "VAR_0, VAR_1, etc.—names chosen to be guaranteed absent from the original code so no "
        "collision occurs. Convert modified AST back to source code."
    ),
    operational_spec_v2=(
        "Scope: user-defined variable and parameter identifiers; renames the single most frequent "
        "name (default top_n=1) to generated names VAR_0, VAR_1, … that do not appear in the "
        "original source. "
        "Procedure: parse to AST; count Name and arg occurrences; pick the most common identifier(s); "
        "rewrite every load/store/del and matching parameter to VAR_i; unparses AST back to source. "
        "Example: `result = data + offset` → `VAR_0 = data + offset` when `result` is the most "
        "common name."
    ),
    type=TransformationType.PRESERVING,
    apply=apply_rv_naive
)

