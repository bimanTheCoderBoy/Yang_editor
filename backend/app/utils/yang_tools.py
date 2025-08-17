import os
from pathlib import Path
from pyang import repository, context
def print_ast_dfs(node, indent=0):
            # print(' ' * indent + f"{node.keyword}: {node.arg}")
            for child in getattr(node, 'i_children', []):
                print_ast_dfs(child, indent + 2)

def ast_to_summary(ast_node, depth=0):
    """
    Recursively walk YANG AST and return an indented structure summary.
    """
    lines = []
    indent = "  " * depth

    node_type = ast_node.keyword  # e.g. 'container', 'leaf', 'list'
    node_name = ast_node.arg      # e.g. 'deviceId', 'interface'

    if node_type and node_name:
        lines.append(f"{indent}{node_type} {node_name}")
    elif node_type:  # for top-level like 'module'
        lines.append(f"{indent}{node_type}")

    for child in getattr(ast_node, 'i_children', []):
        lines.extend(ast_to_summary(child, depth + 1))

    return lines



def generate_ast_summary(ast_root):
    return "\n".join(ast_to_summary(ast_root))


TEMP_DIR = Path("tmp/yang/get-module")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = TEMP_DIR / "temp.yang"

def get_yang_ast(yang_text: str) -> bool: 
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(yang_text)

    try:
        repo = repository.FileRepository(str(TEMP_DIR))
        ctx = context.Context(repo)

        # Parse and validate
        module=ctx.add_module(OUTPUT_FILE.name, yang_text)
        ctx.validate()
        errors = [str(err) for err in ctx.errors] if ctx.errors else []
        if len(errors) == 0:
            return module
        else:
            return None
    except Exception as e:
        return None
    finally:
        # print("jj")
        # Cleanup temp file
        if OUTPUT_FILE.exists():
            os.remove(OUTPUT_FILE)
    
            
            
def get_metadata_recursive(stmt):
    """Recursively collect metadata statements for a YANG Statement."""
    meta = []
    schema_children_set = set(getattr(stmt, "i_children", []))

    for sub in stmt.substmts:
        if sub in schema_children_set:
            continue  
        if sub.substmts:
            meta.append({
                "keyword":sub.keyword,
                "arg": sub.arg,
                # "pos":getattr(sub, "pos", None),
                "substmts": get_metadata_recursive(sub)
            })
        else:
           meta.append({
                "keyword":sub.keyword,
                "arg": sub.arg,
                # "pos":getattr(sub, "pos", None),
            })
    return meta
      

def get_ast_node_by_path(ast_root, path: str):
    """
    Get only the AST node's own details (no children) for a given path.
    Path format: /module/container/leaf
    """
    path_parts = [p for p in path.strip("/").split("/") if p]

    def find_node(node, parts):
        if not parts:
            return node
        for child in getattr(node, "substmts", []):
            if getattr(child, "arg", None) == parts[0]:
                return find_node(child, parts[1:])
        return None

    node = find_node(ast_root, path_parts[1:])
    if not node:
        return None
    
    return {
        "keyword":getattr(node, "keyword", None),
        "arg":getattr(node, "arg", None),
        # "pos":getattr(node, "pos", None),
        "metadata":get_metadata_recursive(node)
    }


def find_paths_by_identifiers(ast_root, identifier):
    """
    Search for nodes in the YANG AST whose name matches any in identifiers.
    Returns a list of full paths.
    """
    results = []

    def walk(node, current_path):
        # Current node name can be in .arg (identifier) or .keyword (type like 'leaf', 'container')
        name = getattr(node, 'arg', None) or node.keyword
        if name == identifier:
            results.append("/" + "/".join(current_path + [name]))

        # Recurse into children
        if hasattr(node, 'substmts'):
            for child in node.substmts:
                walk(child, current_path + [name])

    walk(ast_root, [])
    return results



from pyang import statements,error

def update_ast_with_llm_metadata(ast_root, path, llm_node_json):
    """
    Update a YANG AST node's metadata from LLM output in custom format.

    Args:
        ast_root: The root Statement node from pyang parsing.
        path (str): Path like "/module/container/leaf".
        llm_node_json (dict): LLM output JSON in the agreed format.
    """
    
    # Step 1: Find target node by path
    def find_node(node, parts):
        if not parts:
            return node
        for child in getattr(node, "substmts", []):
            if getattr(child, "arg", None) == parts[0]:
                return find_node(child, parts[1:])
        return None


    path_parts = [p for p in path.strip("/").split("/") if p]
    target_node = find_node(ast_root, path_parts[1:])
   
    if target_node is None:
        raise ValueError(f"Node not found for path: {path}")
    # print(f"got the node {target_node}")
    # print(f"got the path {path}")
    # Step 2: Remove existing metadata substatements
    target_node.substmts = [
        s for s in target_node.substmts
        if s.keyword in getattr(target_node, "i_children", set())
    ]

    # Step 3: Recursive adder for metadata
    def add_metadata_recursive(parent_stmt, metadata_list):
        """
        Recursively add metadata to a pyang AST node (parent_stmt)
        based on our custom LLM output format.
        """
        for meta in metadata_list:
            # Inherit pos from parent, increment line number slightly
            # if parent_stmt.pos:
            #     new_pos = error.Position(
            #         parent_stmt.pos.filename,
            #         parent_stmt.pos.lineno + 0.1, 
            #         getattr(parent_stmt.pos, 'column', None)
            #     )
            # else:
            new_pos = error.Position("<generated>")

            new_stmt = statements.Statement(
                top=parent_stmt.top,
                parent=parent_stmt,
                pos=new_pos,
                keyword=meta["keyword"],
                arg=meta.get("arg", "")
            )
            parent_stmt.substmts.append(new_stmt)

            # Recurse for substmts
            substmts = meta.get("substmts", [])
            if substmts:
                add_metadata_recursive(new_stmt, substmts)

    # Step 4: Add new metadata from LLM JSON
    add_metadata_recursive(target_node, llm_node_json.get("metadata", []))

    return ast_root

def get_yang_text(ast):
    repo = repository.FileRepository(str(TEMP_DIR))
    ctx = context.Context(repo)
    ctx.add_parsed_module(ast)
    from pyang.translators import yang
    from types import SimpleNamespace
    import io
    ctx.opts = SimpleNamespace(
    yang_canonical=False,
    yang_remove_unused_imports=False,
    yang_remove_comments=False,
    yang_line_length=None
   )

    def ast_to_yang(stmt):
        buf = io.StringIO()
        yang.emit_yang(ctx,stmt, buf)  
        return buf.getvalue()
    yang_str = ast_to_yang(ast)
    #push to file 
    # print(yang_str)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(yang_str)
    
    return yang_str

        
        

            
def validate_yang_text(text)->bool:
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    
    try:
        repo = repository.FileRepository(str(TEMP_DIR))
        ctx = context.Context(repo)

        # Parse and validate
        ctx.add_module(OUTPUT_FILE.name, text)
        ctx.validate()

        errors = [str(err) for err in ctx.errors] if ctx.errors else []
        return len(errors) == 0, errors
    finally:
        if OUTPUT_FILE.exists():
            os.remove(OUTPUT_FILE)

    
def verify_ast(ast) -> bool:
    return validate_yang_text(get_yang_text(ast))