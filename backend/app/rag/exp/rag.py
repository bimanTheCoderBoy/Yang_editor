import json
import os
import re
from typing import List

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Choose embeddings backend:
# - SentenceTransformerEmbeddings is portable and works locally (default)
# - If you have Ollama and want to use it, you can switch below.
from langchain_community.embeddings import SentenceTransformerEmbeddings
# from langchain_ollama import OllamaEmbeddings    # optional, if available

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever


# -------------------- CONFIG --------------------
INPUT_JSON_PATH = "rule-book.json"   
INPUT_TXT_PATH = "rule-book.txt"  
PERSIST_JSON_DIR = "./chroma_db/json/"
PERSIST_TXT_DIR = "./chroma_db/txt/"


# Choose embedding model (default local)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# OLLAMA_EMBEDING_MODEL_NAME = "nomic-embed-text:latest"
# embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDING_MODEL_NAME)

# Cross-encoder model for reranking
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  


JSON_BM25_K = 5
JSON_VEC_K = 5
TXT_BM25_K = 4
TXT_VEC_K = 5
MERGE_TOP_K = 12  
FINAL_TOP_K = 6   

os.makedirs(PERSIST_JSON_DIR, exist_ok=True)
os.makedirs(PERSIST_TXT_DIR, exist_ok=True)


def get_json_docs():
    #loading the json data
    with open(INPUT_JSON_PATH, "r") as f:
        json_data = json.load(f)

    # making the documents
    docs = []
    
    # documenting in a way so get parent data or more over context
    for i, rule in enumerate(json_data):
        name = rule.get("name", "")
        intent = rule.get("intent", "")
        syntax = rule.get("syntax", "")
        examples = rule.get("examples", [])
        applies_to = " ".join(rule.get("applies_to", []))
        node_text = f"{name} {intent} {syntax} {applies_to} {' '.join(examples)}"
        node_meta = {
            "doc_type": "node",
            "name": name,
            "idx": i,
            "raw": json.dumps(rule, ensure_ascii=False)
        }
        docs.append(Document(page_content=node_text, metadata=node_meta))
    # Substatement docs / getting exact data
        for s_idx, sub in enumerate(rule.get("substatements", []) or []):
            sub_name = sub.get("name", "")
            cardinality = sub.get("cardinality", "")
            sub_text = f"{sub_name} cardinality:{cardinality} parent:{name}"
            sub_meta = {
                "doc_type": "substatement",
                "name": name,
                "sub_name": sub_name,
                "parent_idx": i,
                "sub_idx": s_idx,
                "raw": json.dumps(sub, ensure_ascii=False)
            }
            docs.append(Document(page_content=sub_text, metadata=sub_meta))

    return docs



def get_txt_docs():
    # loading the txt data
    with open(INPUT_TXT_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # splitting the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(raw_text)
    docs = []
    # making documents from chunks
    for i, c in enumerate(chunks):
        docs.append(Document(page_content=c, metadata={"doc_type": "txt_chunk", "chunk_id": i}))
        
    return docs


# inded all data
def index_all_data():
    json_docs = get_json_docs()
    txt_docs = get_txt_docs()
    Chroma.from_documents(json_docs, embeddings, persist_directory=PERSIST_JSON_DIR)
    print(f"Indexed {len(json_docs)} JSON documents into Chroma.")
    Chroma.from_documents(txt_docs, embeddings, persist_directory=PERSIST_TXT_DIR)
    print(f"Indexed {len(txt_docs)} TXT documents into Chroma.")




def build_retrievers(json_docs: List[Document], txt_docs: List[Document]):
    # Vector retrievers (Chroma)
    json_vector_store = Chroma(persist_directory=PERSIST_JSON_DIR, embedding_function=embeddings)
    json_vector_retriever = json_vector_store.as_retriever(search_kwargs={"k": JSON_VEC_K})

    txt_vector_store = Chroma(persist_directory=PERSIST_TXT_DIR, embedding_function=embeddings)
    txt_vector_retriever = txt_vector_store.as_retriever(search_kwargs={"k": TXT_VEC_K})

    # BM25 retrievers (from the lists)
    json_bm25 = BM25Retriever.from_documents(json_docs)
    json_bm25.k = JSON_BM25_K

    txt_bm25 = BM25Retriever.from_documents(txt_docs)
    txt_bm25.k = TXT_BM25_K

    # Ensemble retrievers
    json_ensemble = EnsembleRetriever(retrievers=[json_bm25, json_vector_retriever], weights=[0.35, 0.65])
    txt_ensemble = EnsembleRetriever(retrievers=[txt_bm25, txt_vector_retriever], weights=[0.4, 0.6])

    # Cross-encoder reranker wrapper
    cross_enc = HuggingFaceCrossEncoder(model_name=CROSS_ENCODER_MODEL)
    cross_reranker = CrossEncoderReranker(model=cross_enc, top_n=MERGE_TOP_K)

    json_compressed = ContextualCompressionRetriever(base_compressor=cross_reranker, base_retriever=json_ensemble)
    txt_compressed = ContextualCompressionRetriever(base_compressor=cross_reranker, base_retriever=txt_ensemble)

    return {
        "json_vector_retriever": json_vector_retriever,
        "json_bm25": json_bm25,
        "json_ensemble": json_ensemble,
        "json_compressed": json_compressed,
        "txt_vector_retriever": txt_vector_retriever,
        "txt_bm25": txt_bm25,
        "txt_ensemble": txt_ensemble,
        "txt_compressed": txt_compressed,
        "cross_reranker": cross_reranker
    }










def extract_arg_from_query(q: str) -> str:
    """
    Heuristic to find identifier like 'GLDSSSW' in the user query.
    1) try 'leaf NAME' or 'node NAME'
    2) fallback: find all-uppercase tokens or underscores that look like identifiers
    """
    m = re.search(r"\bleaf\s+([A-Za-z0-9_\-]+)\b", q, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"\bnode\s+([A-Za-z0-9_\-]+)\b", q, flags=re.IGNORECASE)
    if m:
        return m.group(1)

    # fallback: uppercase-like token of length >=3 (common in YANG names you showed)
    tokens = re.findall(r"\b[A-Z0-9_]{3,}\b", q)
    if tokens:
        return tokens[0]
    return None


if __name__ == "__main__":
    # index_all_data()
    json_docs = get_json_docs()
    txt_docs = get_txt_docs()
    retrievers = build_retrievers(json_docs, txt_docs)
    query = """
    Validate that leaf GLDSSSW length does not exceed 50 characters
    Check the string length of the GLDSSSW leaf in each moi entry to ensure it is 50 characters or fewer
    """
    arg = extract_arg_from_query(query)
    print(f"Extracted argument from query: {arg}")