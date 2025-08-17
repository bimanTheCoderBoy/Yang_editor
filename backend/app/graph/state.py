from pydantic import BaseModel, Field
from typing import List
# from pyang.statements import Statement
    

class Action(BaseModel):
    action:str=Field(description="the single action need to be perform on the yang model")
    targets:List[str]=Field(description="List of yang identifier like container, leaf, etc names involved in the action")
    details:str=Field(description="The detailed info about the action. kind of a context for the action")

class ActionList(BaseModel):
    action_list:List[Action]=Field(description="List of unit actions")
    
    
    
class ASTNode(BaseModel):
    keyword: str
    arg: str
    substmts: List["ASTNode"] = Field(default_factory=list)

class ASTRoot(BaseModel):
    keyword: str
    arg: str
    metadata: List[ASTNode] = Field(default_factory=list)


# for grouping purposes
class TAction(BaseModel):
    action: str = Field(description="The action to be performed")
    details: str = Field(description="Details about the action")

class Target(BaseModel):
    target_value: str = Field(description="The value of the target identifier")
    action_list: List[TAction] = Field(description="List of actions to be performed on this target")


class AgentState(BaseModel):
    messages:List=[]
    retry_count:int =0
    input_text:str=None
    user_query:str=""
    current_yang_model:str=""
    isError:bool=False
    updated_yang_model:str=""
    errors:List=[]
    event_message:str=""
    ast_summary:str=None
    ast_id:str=None
    action_list:List[Action]=None