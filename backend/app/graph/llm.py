from langchain_openai import ChatOpenAI

llm_actioner_splitter =ChatOpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1",
    model_name="openai/gpt-oss-20b:free",
    temperature=0.1,
    # model_name="mistralai/devstral-small-2505:free"
   
)

# llm_basic = ChatOpenAI(
#     api_key="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",
#     base_url="https://openrouter.ai/api/v1",
#     model_name="mistralai/devstral-small-2505:free"
# )
#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
#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
from langchain_ollama.chat_models import ChatOllama

# llm_basic = ChatOllama(
#     model="qwen3:0.6b",
#     server_url="http://localhost:11434", 
#     reasoning=False,
#     temperature=0
# )
#"qwen3:4b"
#granite-code:3b
# llm_action_executer = ChatOllama(
#     model="granite-code:3b",
#     server_url="http://localhost:11434",
#     # reasoning=True,
#     # temperature=.2
# )

llm_action_executer =ChatOpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1",
    model_name="openai/gpt-oss-20b:free",
    # model_name="mistralai/devstral-small-2505:free"
)
# llm_actioner_splitter = ChatOllama(
#     model="phi3:mini",
#     server_url="http://localhost:11434",
#     # reasoning=True,
#     temperature=.1
# )