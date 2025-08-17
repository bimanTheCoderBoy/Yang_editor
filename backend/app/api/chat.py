from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import StreamingResponse
import json
from app.graph.build import graph

router = APIRouter()


@router.post("/chat")
async def chat(thread_id: str = Form(...), user_query: str = Form(...), text:str = Form(...)):

    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"user_query": user_query,"input_text":text}
    result=await graph.ainvoke(inputs, config=config)
    return {"data":result}
    # async def event_stream():
    #     async for event in graph.astream(inputs, config=config):
    #         # event is like {"GUARDRAIL": {...}} or {"VALIDATOR": {...}}
    #         for node_name, node_state in event.items():
    #             # ====== Handle GUARDRAIL ======
    #             if node_name == "GUARDRAIL":
    #                 yield f"data: {json.dumps({'type': 'status', 'message': 'GUARDRAIL Running'})}\n\n"

    #             # ====== Handle RULE_ENGINE ======
    #             elif node_name == "RULE_ENGINE":
    #                 yield f"data: {json.dumps({'type': 'status', 'message': 'RULE Engine Running'})}\n\n"
    #                 # # If RULE_ENGINE updates `updated_yang_model`
    #                 # if 'updated_yang_model' in node_state and node_state['updated_yang_model']:
    #                 #     # you can also stream partial content here if needed
    #                 #     pass

    #             # ====== Handle VALIDATOR ======
    #             elif node_name == "VALIDATOR":
    #                 is_error = node_state.get("isError", False)
    #                 retry_count = node_state.get("retry_count", 0)

    #                 yield f"data: {json.dumps({'type': 'status', 'message': 'Validator Running'})}\n\n"

    #                 if is_error:
    #                     if retry_count > 3:
    #                         yield f"data: {json.dumps({'type': 'error', 'message': 'Retried 3+ times still failed, try again'})}\n\n"
    #                     else:
    #                         yield f"data: {json.dumps({'type': 'error', 'message': 'Validation failed. Retrying...'})}\n\n"
    #                 else:
    #                     # Success: send the updated yang model
    #                     updated_model = node_state.get("updated_yang_model", "")
    #                     yield f"data: {json.dumps({'type': 'success', 'updated_yang_model': updated_model})}\n\n"

    #     # Final done event
    #     yield f"data: {json.dumps({'type': 'end'})}\n\n"

    # return StreamingResponse(event_stream(), media_type="text/event-stream")
