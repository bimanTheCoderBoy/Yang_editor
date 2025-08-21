import json
import os
import re
from typing import List

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Choose embeddings backend:
# - SentenceTransformerEmbeddings is portable and works locally (default)
# - If you have Ollama and want to use it, you can switch below.
from langchain_huggingface import HuggingFaceEmbeddings
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
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# OLLAMA_EMBEDING_MODEL_NAME = "nomic-embed-text:latest"
# embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDING_MODEL_NAME)

# Cross-encoder model for reranking
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  


JSON_BM25_K = 5
JSON_VEC_K = 5
TXT_BM25_K = 4
TXT_VEC_K = 5
MERGE_TOP_K = 15
FINAL_TOP_K = 10

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




def build_retrievers():
    json_docs = get_json_docs()
    txt_docs = get_txt_docs()
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
        "cross_reranker": cross_reranker,
        "json_docs": json_docs,
        "txt_docs": txt_docs
    }










import re

YANG_KEYWORDS = [
    "action",
    "anydata",
    "anyxml",
    "argument",
    "augment",
    "base",
    "belongs-to",
    "bit",
    "case",
    "choice",
    "config",
    "contact",
    "container",
    "decimal64",
    "default",
    "description",
    "deviation",
    "deviate",
    "deviate-add",
    "deviate-delete",
    "deviate-replace",
    "deviate-not-supported",
    "enum",
    "error-app-tag",
    "error-message",
    "extension",
    "feature",
    "fraction-digits",
    "grouping",
    "identity",
    "if-feature",
    "import",
    "include",
    "input",
    "key",
    "leaf",
    "leaf-list",
    "length",
    "list",
    "mandatory",
    "max-elements",
    "min-elements",
    "module",
    "must",
    "namespace",
    "notification",
    "ordered-by",
    "organization",
    "output",
    "path",
    "pattern",
    "position",
    "prefix",
    "presence",
    "range",
    "reference",
    "refine",
    "require-instance",
    "revision",
    "revision-date",
    "rpc",
    "status",
    "submodule",
    "type",
    "typedef",
    "unique",
    "units",
    "uses",
    "value",
    "when",
    "yang-version",
    "yin-element"
]


def extract_keyword_from_query(q: str):
    """
    Detect YANG keywords (like 'leaf', 'length', etc.) in the query.
    Returns the first keyword found, or None if none found.
    """
    q_lower = q.lower()
    keys=set()
    for kw in YANG_KEYWORDS:
        if re.search(rf"\b{kw}\b", q_lower):
            keys.add(kw)
    return keys

def merge_preserve_provenance(json_docs, txt_docs, max_total=MERGE_TOP_K):
    seen = set()
    merged = []

    def add(d: Document, source: str):
        key = d.page_content.strip()[:300]
        if key in seen:
            return
        seen.add(key)
        meta = dict(d.metadata) if d.metadata else {}
        meta["provenance"] = source
        merged.append(Document(page_content=d.page_content, metadata=meta))

    for d in json_docs:
        add(d, "json")
    for d in txt_docs:
        add(d, "txt")

    return merged[:max_total]
def rerank_with_cross_encoder(query: str, candidate_docs: List[Document], reranker: CrossEncoderReranker, top_k=FINAL_TOP_K):
    """
    Use cross-encoder wrapper to rerank candidate docs. The CrossEncoderReranker wrapper will compress docs;
    fallback to simple ranking if an error occurs.
    """
    try:
        # CrossEncoderReranker has a method to compress/rerank given a list of docs and a query
        compressed = reranker.compress_documents(candidate_docs, query)
        # compress_documents often returns documents sorted by score
        return compressed[:top_k]
    except Exception as e:
        print("[WARN] cross-encoder rerank failed:", str(e))
        # fallback: return first top_k candidates
        return candidate_docs[:top_k]


retrievers=build_retrievers()

def get_optimal_docs():
    json_docs = retrievers["json_docs"]
    query = """
    Validate that leaf GLDSSSW length does not exceed 50 characters
    Check the string length of the GLDSSSW leaf in each moi entry to ensure it is 50 characters or fewer
    """
    
    keys = extract_keyword_from_query(query)
    exact_candidates = []
    if keys:
        # search json_docs for node with name == arg OR substatement parent == arg
        for d in json_docs:
            meta = d.metadata or {}
            name = meta.get("name", "")
            sub_name = meta.get("sub_name", "")
            if name == keys or sub_name == keys:
                # attach provenance
                meta2 = dict(meta)
                meta2["provenance"] = "json_exact"
                exact_candidates.append(Document(page_content=d.page_content, metadata=meta2))
    if exact_candidates:
        print(f"[INFO] Exact identifier '{keys}' found in JSON docs; including as high-priority candidates.")
    json_ensemble_docs = retrievers["json_ensemble"].invoke(query)
    txt_ensemble_docs = retrievers["txt_ensemble"].invoke(query)
    
    merged_pre = []
    for d in exact_candidates:
        merged_pre.append(d)
    merged_pre += merge_preserve_provenance(json_ensemble_docs, txt_ensemble_docs, max_total=MERGE_TOP_K - len(merged_pre))
    
    reranker_wrapper = retrievers["json_compressed"].base_compressor  
    final_top = rerank_with_cross_encoder(query, merged_pre, reranker_wrapper, top_k=FINAL_TOP_K)
    
    docs=[]
    for doc in final_top:
        meta = doc.metadata or {}
        if meta.get("raw"):
            docs.append(meta.get("raw"))
        else:
            docs.append(doc.page_content)
    return docs


if __name__ == "__main__":
   print(get_optimal_docs())