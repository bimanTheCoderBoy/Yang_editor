# rag_yang_pipeline.py
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
INPUT_JSON_PATH = "rule-book.json"   # must be a JSON array of objects
INPUT_TXT_PATH = "rule-book.txt"     # RFC / normative text
PERSIST_JSON_DIR = "./chroma_db/json/"
PERSIST_TXT_DIR = "./chroma_db/txt/"

# Choose embedding model (default local)
#model = SentenceTransformer("ml-enthusiast13/telecom_bge_embedding_model")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# If using Ollama instead (optional):
# OLLAMA_EMBEDING_MODEL_NAME = "nomic-embed-text:latest"
# embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDING_MODEL_NAME)

# Cross-encoder model for reranking

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # or your chosen HF reranker

# Retrieval hyperparams (tuneable)
JSON_BM25_K = 5
JSON_VEC_K = 5
TXT_BM25_K = 4
TXT_VEC_K = 5
MERGE_TOP_K = 12     # how many merged candidates to pass to cross-encoder
FINAL_TOP_K = 6      # final number to return to LLM/user

os.makedirs(PERSIST_JSON_DIR, exist_ok=True)
os.makedirs(PERSIST_TXT_DIR, exist_ok=True)


# -------------------- Helpers: parse & create documents --------------------
def load_json_rules(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array of rule objects.")
    return data


def json_to_documents(rules: List[dict]) -> List[Document]:
    """
    Create:
      - node documents (one per top-level rule)
      - substatement documents (one per substatement)
    All documents are short canonical text intended for BM25 + embedding.
    """
    docs = []
    for i, rule in enumerate(rules):
        name = rule.get("name", "")
        intent = rule.get("intent", "")
        syntax = rule.get("syntax", "")
        examples = rule.get("examples", [])
        applies_to = " ".join(rule.get("applies_to", []))

        # Node doc (represents the whole rule)
        node_text = f"{name} {intent} {syntax} {applies_to} {' '.join(examples)}"
        node_meta = {
            "doc_type": "node",
            "name": name,
            "idx": i,
            "raw": json.dumps(rule, ensure_ascii=False)
        }
        docs.append(Document(page_content=node_text, metadata=node_meta))

        # Substatement docs
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


def txt_to_documents(txt_path: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    with open(txt_path, "r", encoding="utf-8") as f:
        raw = f.read()
    chunks = splitter.split_text(raw)
    docs = []
    for i, c in enumerate(chunks):
        docs.append(Document(page_content=c, metadata={"doc_type": "txt_chunk", "chunk_id": i}))
    return docs


# -------------------- Indexing (Chroma) --------------------
def index_json(docs: List[Document]):
    # create or load chroma collection for JSON docs
    col = Chroma.from_documents(docs, embeddings, persist_directory=PERSIST_JSON_DIR)
    print(f"[INDEX] Stored {len(docs)} JSON-derived documents into Chroma at {PERSIST_JSON_DIR}")
    return col


def index_txt(docs: List[Document]):
    col = Chroma.from_documents(docs, embeddings, persist_directory=PERSIST_TXT_DIR)
    print(f"[INDEX] Stored {len(docs)} TXT chunks into Chroma at {PERSIST_TXT_DIR}")
    return col


# -------------------- Build retrievers --------------------
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


# -------------------- Utility: extract probable identifier (arg) from query --------------------
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


# -------------------- Merge + rerank flow --------------------
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


# -------------------- Main retrieval demo --------------------
def run_demo():
    # Load source data
    rules = load_json_rules(INPUT_JSON_PATH)
    json_docs = json_to_documents(rules)
    txt_docs = txt_to_documents(INPUT_TXT_PATH)

    # Index (create or re-create)
    index_json(json_docs)
    index_txt(txt_docs)

    # Build retrievers
    retrievers = build_retrievers(json_docs=json_docs, txt_docs=txt_docs)

    # Example query (yours)
    query = """
    Validate that leaf GLDSSSW length does not exceed 50 characters
    Check the string length of the GLDSSSW leaf in each moi entry to ensure it is 50 characters or fewer
    """

    # 1) Exact-arg prefilter
    arg = extract_arg_from_query(query)
    exact_candidates = []
    if arg:
        # search json_docs for node with name == arg OR substatement parent == arg
        for d in json_docs:
            meta = d.metadata or {}
            name = meta.get("name", "")
            sub_name = meta.get("sub_name", "")
            if name == arg or sub_name == arg:
                # attach provenance
                meta2 = dict(meta)
                meta2["provenance"] = "json_exact"
                exact_candidates.append(Document(page_content=d.page_content, metadata=meta2))
    if exact_candidates:
        print(f"[INFO] Exact identifier '{arg}' found in JSON docs; including as high-priority candidates.")

    # 2) Get ensemble candidates from JSON and TXT (pre-rerank)
    json_ensemble_docs = retrievers["json_ensemble"].get_relevant_documents(query)
    txt_ensemble_docs = retrievers["txt_ensemble"].get_relevant_documents(query)

    # 3) Merge: prefer exact candidates first, then ensemble results
    merged_pre = []
    for d in exact_candidates:
        merged_pre.append(d)
    merged_pre += merge_preserve_provenance(json_ensemble_docs, txt_ensemble_docs, max_total=MERGE_TOP_K - len(merged_pre))

    print("\n--- Pre-Rerank Candidates (merged) ---")
    for i, d in enumerate(merged_pre, 1):
        prov = d.metadata.get("provenance", "unknown")
        print(f"{i}. provenance={prov} | text={d.page_content[:200]} | meta_keys={list(d.metadata.keys())}")

    # 4) Rerank merged candidates using cross-encoder wrapper
    # Build a CrossEncoderReranker wrapper instance (we have it inside retrievers['json_compressed'] but we can reuse its underlying model)
    reranker_wrapper = retrievers["json_compressed"].base_compressor  # CrossEncoderReranker instance
    final_top = rerank_with_cross_encoder(query, merged_pre, reranker_wrapper, top_k=FINAL_TOP_K)

    print("\n=== FINAL TOP after cross-encoder rerank ===")
    for i, d in enumerate(final_top, 1):
        prov = d.metadata.get("provenance", "unknown")
        print(f"{i}. provenance={prov} | text={d.page_content[:350]} | meta_keys={list(d.metadata.keys())}")

    # Example: present final candidates to LLM (you would pass 'final_top' + the actual AST node for the LLM)
    # We print them so you can see what would be passed forward.
    return final_top


if __name__ == "__main__":
    final = run_demo()
