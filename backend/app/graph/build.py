from langgraph.graph import StateGraph,add_messages, END
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import  InMemorySaver
from typing import List
from app.graph.state import AgentState
from app.graph.pipeline import (
    validate_initial_yang,
    action_executer,
   actionlist_builder,
   convert_to_ast
)
    
builder=StateGraph(AgentState)

builder.add_node("VALIDATOR",validate_initial_yang)
builder.add_node("ACTION_EXECUTER",action_executer)
builder.add_node("ACTIONLIST_BUILDER",actionlist_builder)
builder.add_node("AST_CONVERTER",convert_to_ast)
# builder.add_node("INITIAL_LANGUAGE_DETECT",initial_language_detect)
# builder.add_node("VERIFY_LANGUAGE_DETECT",verify_language_detect)
# builder.add_node("SYNTAX_FIXER",syntax_fixer)

# builder.set_entry_point("INITIAL_LANGUAGE_DETECT")
builder.set_entry_point("VALIDATOR")

graph =builder.compile(checkpointer=InMemorySaver())

