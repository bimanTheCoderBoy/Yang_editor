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
from sentence_transformers.cross_encoder import CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv(override=True)

# -------------------- Config --------------------
INPUT_JSON_PATH =  "./rule-book.json"
INPUT_TXT_PATH = "rule-book.txt"
PERSIST_DIR = "./chroma_db/json"
# -------------------- Embeddings --------------------
# embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

# -------------------- Store JSON Data --------------------
# Here understanding the Data We Have to do a twik
    #give more priorify to the "name" field
    #remove statement from seaching but remain in the results
def store_json_data():
    docs = []
    with open("./rule-book.json", "r") as f:
        rule_list = json.loads(f.read())
        for rule in rule_list:
            # Remove "substatements" from searchable text
            # searchable_rule = {k: v for k, v in rule.items() if k != "substatements"}

            # Give high priority to "name" field by repeating it
            boosted_name = (rule["name"] + " ") * 7
            page_content = boosted_name + json.dumps(rule)

            docs.append(
                Document(
                    page_content=page_content,
                    metadata={"record": json.dumps(rule, ensure_ascii=False)}
                )
            )
    return docs



def store_txt_data():
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    with open(INPUT_TXT_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()
    chunks = text_splitter.split_text(raw_text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    return docs


def load_json_data():
    docs = store_json_data()
    vector_store = Chroma.from_documents(docs, embeddings, persist_directory=PERSIST_DIR)
    # vector_store.persist()
    print(f"Loaded {len(docs)} documents into Chroma.")

def load_txt_data():
    docs = store_txt_data()
    vector_store = Chroma.from_documents(docs, embeddings, persist_directory=PERSIST_DIR)
    # vector_store.persist()
    print(f"Loaded {len(docs)} documents into Chroma.") 

# -------------------- Retrieve Data --------------------
def retrieve_data():
    # Load Chroma DB
    vector_store = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    # Vector retriever
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})


    all_docs = store_json_data() 
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = 5

    # Combine retrievers
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.6, 0.4]
    )

    # Reranker
    model = CrossEncoder('mixedbread-ai/mxbai-rerank-xsmall-v1',device='cpu')
    reranker = CrossEncoderReranker(model=model, top_n=4
                                    )

    # Combine retrievers and reranker
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble_retriever
    )

    # Query
    query = """
           
    """
    docs = compression_retriever.invoke(query)
    docs=[doc.metadata["record"] for doc in docs ]
    docs=set(docs)

    for i, doc in enumerate(docs, start=1):
        print(f"Result {i}:\n{doc}\n")



# -------------------- Main --------------------
if __name__ == "__main__":
    # Run once to load data
    load_json_data()
    # load_txt_data()

    # Retrieve
    # retrieve_data()