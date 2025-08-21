from langgraph.types import Command
from app.graph.state import AgentState
from langchain_core.messages import AIMessage, HumanMessage,SystemMessage
from app.utils.yang_tools import ( get_yang_ast,
                                  generate_ast_summary,
                                  get_ast_node_by_path,
                                  find_paths_by_identifiers,
                                  update_ast_with_llm_metadata,
                                  verify_ast,get_yang_text,
                                  validate_yang_text, )
from pydantic import BaseModel, Field
from typing import List
from app.graph.llm import llm_actioner_splitter, llm_action_executer
from langchain.output_parsers import PydanticOutputParser
from app.graph.state import ActionList,Action,ASTNode, ASTRoot,Target, TAction
from app.graph.llm import llm_action_executer, llm_actioner_splitter
from app.store.AST_store import (save_ast, get_ast)
from app.rag.repo import get_optimal_docs
import json



#INITIAL YANG VALIDATOR
def validate_initial_yang(state:AgentState):
    print("Validating the initial YANG model text...")
    text=state.input_text
    check, errors=validate_yang_text(text)
    if check :
        print("YANG VALIDATION SUCCESSFULL") 
        return Command(
            goto="AST_CONVERTER",
            update={
                "event_message":"YANG VALIDATION SUCCESSFULL",
                "retry_count":0,
                "errors": []
            }
        )
           
    else :
        print("YANG VALIDATION FAILED")
        return {
            "event_message": "YANG VALIDATION FAILED --- (Upload a Correct Yang Data)",
            "errors":errors
        }
        
    


#CONVERT INTO AST
def convert_to_ast(state:AgentState):
    print("Converting YANG to AST...")
    text=state.input_text
    ast=get_yang_ast(text)
    if not ast :
        return {
            "event_message": "YANG TO AST PARSING FAILED --- (Upload a Correct Yang Data)",
        }
    
    
    ast_id=save_ast(ast, ttl_seconds=3600)  
    summary=generate_ast_summary(ast)
    print(summary)
    print("YANG TO AST PARSING SUCCESSFULL")
    return Command(
        goto="ACTIONLIST_BUILDER",
        update={
            "event_message": "YANG TO AST PARSING SUCCESSFULL",
            "ast_summary":summary,
            "ast_id":ast_id
        }
    )
    
    
#ACTIONLIST BUILDER
def actionlist_builder(state:AgentState):
    print("Building action list from AST...")
    parser = PydanticOutputParser(pydantic_object=ActionList)
    prompt=f"""
    You are a Yang Model Task devider. You have access to user_query and the Yang model's tree summary. 
    Your task is to first reason with the user_query and Yang model summary and then divide
    a the whole task in to smaller single unit actions and find out it's paths the action applies to, from the ast tree you are provided to you bellow.
    USER QUERY:
    {state.user_query}
    SUMMARY of YANG MODEL AST(abstruct syntax tree):
    {state.ast_summary}
    give output in bellow format.
    f"{parser.get_format_instructions()}"
    """
    structured_llm=llm_actioner_splitter.with_structured_output(ActionList)
    print("Calling the LLM for getting the Structured Action List ...")
    try:
        action_list:ActionList=structured_llm.invoke(prompt)
        action_list=action_list.action_list
        print ("GOT the Action list..")
        print(str(action_list))
        print("-------------------------------------------")
        
        
        return Command(
            goto="ACTION_EXECUTER",
            update={
                 "event_message": "SUCCESSFULY GOT THE ACTION LIST",
                 "action_list":action_list
            }
        )
    except Exception as e:
        print(e)
        print("Error while getting the action list")
        return{
                "event_message": "ACTIONLIST GENERATION FAILED",
            }
    
    
    

#ACTION EXECUTER
def action_executer(state:AgentState):
    print("Executing actions...")
    action_list=state.action_list
    ast_id=state.ast_id
    ast=get_ast(ast_id)
    #preparing the action list 
    target_list:List[Target]=[]
    target_set=set()
    for act in action_list:
        for target in act.paths:
            target_set.add(target)
            

    for target in target_set:
       
        actions=[action for action in action_list if  target in action.paths]
        for act in actions:
            target_list.append(
                Target(
                    target_value=target,
                    action_list=[
                        TAction(
                            action=act.action,
                            details=act.details
                        ) for act in actions
                    ]
                )
                
            )
        
        
    print(f"Grouped target List : \n{target_list}")
    parser= PydanticOutputParser(pydantic_object=ASTRoot)
    
    print(f"LENGTH of the Action List : {len(target_list)}")
    print("Processing the action one by one...")
    for target in target_list:
        actions=target.action_list
        
        target_value=target.target_value

        data=get_ast_node_by_path(ast,target_value)


           
        print("GOT the AST Node for this path and action")
        print(f"Going to execute for :-> \n    Actions: {json.dumps([a.model_dump() for a in actions], indent=2)}\n ")
        print(f"Path: {target_value}")
        print(f"Node: {data}")
           

        #---------------------------------Dynamic Data Retrieval Hybrid Search with reranking--------------------------------------
        # preparing data for JSON RULE Book retrieval
       

        word=data["keyword"]
        word=word.lower()

        action_str=""
        for at in actions:
            action_str+=f"{at.action} {at.details}"
                
            # final data
        retrieval_data=f"{word} {action_str}"
        retrieval_results=get_optimal_docs(retrieval_data)

        
                
                
        system_prompt = """
              You are a YANG AST node editor (authoritative, deterministic). 

            You will receive:
            1) One AST node in our internal JSON format (not textual YANG). The node will follow the ASTNode/ASTRoot schema described below.
            2) An Action List: an array of objects (TAction) where each object has:
                - "action": a short, imperative instruction (what to change)
                - "details": any context or values required to perform the action
                Example of action-list JSON: [{"action":"Set allowed values","details":"ACTIVE, INACTIVE, PENDING"}]
            3) The target identifier is provided externally (your code finds node by path). Your job is to apply **all** TAction items **in order** to the given node.

            Goal:
            Apply each TAction to the provided node and return the **entire modified node** as valid JSON (matching the ASTRoot model). Do not include any extra text, comments, or explanation. If the output is not valid JSON, the caller will retry.

            Strict Output Requirements:
            - Output **ONLY** the JSON object representing the full modified node (start with `{`, end with `}`), nothing else.
            - The JSON must match the ASTRoot model:
                {
                    "keyword": "<node_keyword>",
                    "arg": "<node_name>",
                    "metadata": [
                    { "keyword": "<metadata_keyword>", "arg": "<arg_or_empty_string>", "substmts": [ ... ] },
                    ...
                    ]
                }
            - Each `substmts` element must itself be an object with `keyword`, `arg`, and `substmts` (lists may be empty).
            - `metadata` must be a list (empty list allowed).
            - `arg` fields must be strings (use `""` if not applicable).
            - Return the full node (do not return partial fragments).
            - If you cannot comply with the action exactly, make the **most reasonable, standards-compliant choice** and still return a single JSON node.

            Behavior Rules & Conventions (apply these when interpreting actions):
            1. **Do not remove existing metadata** unless the action explicitly requires removal. If the action requires a conflicting change (e.g., change `type` to enumeration), replace or modify the existing conflicting metadata as needed.
            2. **Enumerations**: when action says allowed values are X,Y,Z — set `type` to `"enumeration"` and add `enum` substmts for each value.
            3. **Length constraints**:
                - "does not exceed N characters" → add/replace inner `length` with `"arg": "1..N"`.
                - "exactly N characters" → set `length` substmt with `"arg": "N"`.
                - "minimum N characters" → set `length` substmt with `"arg": "N.."`.
            4. **Patterns**: if asked to add a regex constraint, add a `pattern` substatement under the `type` node with the regex as `arg`.
            5. **Type changes**: convert or replace the `type` metadata where required to satisfy the action.
            6. **If action mentions an allowed set of values** but the node is not a leaf, apply the change only if that is standards-meaningful; otherwise decide the most reasonable place to put the constraint (but prefer to change `type` for leaves).
            7. Always ensure JSON is syntactically valid (double quotes, correct arrays / objects).


            Temperature and style:
            Use deterministic, precise phrasing in your transformations (programmatic style). Avoid commentary, natural language explanation, or extra fields.

            ---
            Examples :
             
                Example 1 — length + pattern (multiple TAction items, order matters)
                Input Node:
                    {
                    "keyword": "leaf",
                    "arg": "neid",
                    "metadata": [
                        { "keyword": "type", "arg": "string" }
                    ]
                    }
                
                Action List:
                [
                { "action": "Limit maximum length", "details": "50" },
                { "action": "Add pattern", "details": "^[A-Z0-9]{1,50}$ (uppercase alphanumeric)" }
                ]
                
                Valid Output:
                    {
                        "keyword": "leaf",
                        "arg": "neid",
                        "metadata": [
                            {
                            "keyword": "type",
                            "arg": "string",
                            "substmts": [
                                { "keyword": "length", "arg": "1..50", "substmts": [] },
                                { "keyword": "pattern", "arg": "^[A-Z0-9]{1,50}$", "substmts": [] }
                            ]
                            }
                        ]
                        }





                """

        system_prompt+=f"\n At runtime append the exact Pydantic format instructions:{parser.get_format_instructions()}"


            
        isError=False
            #get relement RAG rule
            
            
            # [json.dumps(json.loads(r),indent=2) for r in json_retrieval_results]
            
        for i in range(0,2):
            human_prompt=f"""
                Below is the node and action list you must process, plus relevant YANG rule data retrieved from the official rule book.
                
                === BEGIN RELEVANT RULES===
                {retrieval_results}
                === END RELEVANT RULES ===
               
                Input Node:
                {json.dumps(data, indent=2)}

                Action List (ordered array of TAction objects):
                {json.dumps([a.model_dump() for a in actions], indent=2)}


                Apply each action in order, Obey all the above rules use these rules to answer. After applying all actions, return the entire modified node as JSON following the ASTRoot schema.
                    """
            if state.errors:
                human_prompt+=f"\n You did following error in your previous responss.\n {state.errors}\n Ensure there is no such error in your output"


                # print(f"DEBUG  --- \n {human_prompt} \n\n {system_prompt}\n --- DEBUG")
                # print("DEBUG---------------")
                # print(human_prompt)
            print("Calling the LLM for getting the AST Node with applied actions ...")
            result:ASTRoot=llm_action_executer.with_structured_output(ASTRoot).invoke([SystemMessage(system_prompt),HumanMessage(human_prompt)])
                
                
                
            print("GOT LLM RESULT... FOR this PATH and ACTION")
            print(str(result.dict()))
            print("Validating the RESULT...")
            u_ast=update_ast_with_llm_metadata(ast,target_value,result.dict())
            ast=u_ast
            error_check , errors=verify_ast(u_ast)
            if(error_check):
                isError=False
                print("Validation done Going to execute next ACTION ")
                break
            else:
                isError=True
                print(f"GOT error RETRYING.... {i+1}")
                state.errors=errors
        if isError:
            print("RESTARTING The full EXECUTION....")
            if state.retry_count < 2:
                state.retry_count+=1
                return Command(
                        goto="AST_CONVERTER",
                        update={
                            "isError": True,
                            "retry_count": state.retry_count
                        }
                    )
            else:
                print("Error happening MULTIPLE times.. RETRY by  ADJUSTING your prompt")
                return {
                        "event_message": "Action execution failed after multiple retries. Please adjust your prompt.",
                        
                    } 
             
    error_check , errors=verify_ast(ast)
    if error_check:
        print("GOT the FINAL result storing the data into file....")
        get_yang_text(ast)
        print("DATA stored successfully..")
        return {
            "event_message": "Action execution completed successfully.",
            "updated_yang_model": get_yang_text(ast),
           
        }
    else: 
        print("Getting error try Again adjusting your prompt ")
        return {
            "event_message": "Action execution failed.",
            "errors": errors,
            "ast_id": None
        }