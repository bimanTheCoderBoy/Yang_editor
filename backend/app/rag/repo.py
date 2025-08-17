import json
import os
from dotenv import load_dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


OLLAMA_EMBEDING_MODEL_NAME=os.getenv("OLLAMA_EMBEDING_MODEL_NAME","nomic-embed-text:latest")
embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDING_MODEL_NAME)


#------------------------------------------CHROMA DB REPO-------------------------------------------------------------------
INPUT_JSON_PATH = os.getenv("INPUT_JSON_PATH", "rule-book.json")
PERSIST_JSON_DIR = os.getenv("JSON_DB_DIR", "app/rag/chroma_db/json/")
PERSIST_TXT_DIR = os.getenv("TXT_DB_DIR", "app/rag/chroma_db/txt/")




def store_json_data():
    docs = []
    with open(INPUT_JSON_PATH, "r") as f:
        rule_list = json.loads(f.read())
        for rule in rule_list:
            # Remove "substatements" from searchable text
            searchable_rule = {k: v for k, v in rule.items() if k != "substatements"}

            # Give high priority to "name" field by repeating it
            boosted_name = (rule["name"] + " ") * 3
            page_content = boosted_name + json.dumps(searchable_rule)

            docs.append(
                Document(
                    page_content=page_content,
                    metadata={"record": json.dumps(rule, ensure_ascii=False)}
                )
            )
    return docs

json_rulebook_retriver=None
txt_rulebook_retriver=None

def get_txt_rulebook_retriever():
    return txt_rulebook_retriver
def get_json_rulebook_retriever():
    return json_rulebook_retriver

def load_database():
    
    #------------------load json data------------------------------------
    json_vector_store = Chroma(persist_directory=PERSIST_JSON_DIR, embedding_function=embeddings)
    json_vector_retriever = json_vector_store.as_retriever(search_kwargs={"k": 7})
    
    all_docs = store_json_data() 
    json_bm25_retriever = BM25Retriever.from_documents(all_docs)
    json_bm25_retriever.k = 5

    # Combine retrievers
    json_ensemble_retriever = EnsembleRetriever(
        retrievers=[json_bm25_retriever, json_vector_retriever],
        weights=[0.4, 0.6]
    )

    # Reranker
    model = HuggingFaceCrossEncoder(model_name='mixedbread-ai/mxbai-rerank-xsmall-v1')
    reranker = CrossEncoderReranker(model=model, top_n=5)

    # Combine retrievers and reranker
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=json_ensemble_retriever
    )
    
    #set the retriever
    global json_rulebook_retriver
    json_rulebook_retriver=compression_retriever

#------------------TXT data------------------------------------
    txt_vector_store = Chroma(persist_directory=PERSIST_TXT_DIR, embedding_function=embeddings)
    txt_vector_retriever = txt_vector_store.as_retriever(search_kwargs={"k": 7})

    all_docs = store_json_data()
    txt_bm25_retriever = BM25Retriever.from_documents(all_docs)
    txt_bm25_retriever.k = 5

    # Combine retrievers
    txt_ensemble_retriever = EnsembleRetriever(
        retrievers=[txt_bm25_retriever, txt_vector_retriever],
        weights=[0.4, 0.6]
    )

    # Reranker
    model = HuggingFaceCrossEncoder(model_name='mixedbread-ai/mxbai-rerank-xsmall-v1')
    reranker = CrossEncoderReranker(model=model, top_n=5)

    # Combine retrievers and reranker
    txt_compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=txt_ensemble_retriever
    )


    global txt_rulebook_retriver
    txt_rulebook_retriver=txt_compression_retriever



