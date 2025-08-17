"""
This is a store for storing AST Objects.
The problem behind on the reson behind it is :
    in langgraph state we cannot manage custom object that are not JSON serilizable like AST object fro pyang
    so we need to find out a option that how an we manage AST objects.

SOLUTION: ->
    a centralized store for AST objects kind of a repository from where other nodes can put and retrieve AST objects.
    -it is Thread safe.
"""

import uuid
import threading
import time
from typing import Optional, Any, Dict, Tuple

# Thread-safe in-process AST store.
# Stores entries as ast_id -> (ast_object, expiry_timestamp_or_None)
_AST_STORE: Dict[str, Tuple[Any, Optional[float]]] = {}
_AST_STORE_LOCK = threading.RLock()

def _current_ts() -> float:
    return time.time()

def save_ast(ast_obj: Any, ttl_seconds: Optional[int] = None) -> str:
    """
    Save `ast_obj` in the in-process store and return an ast_id (string).
    Optionally specify ttl_seconds to auto-drop the AST after TTL.
    """
    aid = str(uuid.uuid4())
    expiry = _current_ts() + ttl_seconds if ttl_seconds is not None else None
    with _AST_STORE_LOCK:
        _AST_STORE[aid] = (ast_obj, expiry)
    if ttl_seconds is not None:
        # schedule a background drop
        def _drop_later(aid_local):
            with _AST_STORE_LOCK:
                entry = _AST_STORE.get(aid_local)
                if entry is None:
                    return
                _, exp = entry
                if exp is None:
                    return
                if _current_ts() >= exp:
                    _AST_STORE.pop(aid_local, None)
        timer = threading.Timer(ttl_seconds + 1, _drop_later, args=(aid,))
        timer.daemon = True
        timer.start()
    return aid

def update_or_save_ast(ast_id, new_ast):
    """
    If ast_id exists, update its value.
    If not, save new_ast and return the new id.
    """
    with _AST_STORE_LOCK:
        if ast_id in _AST_STORE:
            _AST_STORE[ast_id] = new_ast
            return ast_id
        else:
            return save_ast(new_ast)

def get_ast(ast_id: str) -> Optional[Any]:
    """Retrieve AST object by id, or None if not present/expired."""
    with _AST_STORE_LOCK:
        entry = _AST_STORE.get(ast_id)
        if not entry:
            return None
        ast_obj, expiry = entry
        if expiry is not None and _current_ts() >= expiry:
            # expired
            _AST_STORE.pop(ast_id, None)
            return None
        return ast_obj

def drop_ast(ast_id: str) -> None:
    """Remove AST from store (idempotent)."""
    with _AST_STORE_LOCK:
        _AST_STORE.pop(ast_id, None)