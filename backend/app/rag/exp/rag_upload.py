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
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv(override=True)

# -------------------- Config --------------------
OLLAMA_EMBEDING_MODEL_NAME="nomic-embed-text:latest"
embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDING_MODEL_NAME)


#------------------------------------------CHROMA DB REPO-------------------------------------------------------------------
INPUT_JSON_PATH = "rule-book.json"
INPUT_TXT_PATH ="rule-book.txt"
PERSIST_JSON_DIR =  "./chroma_db/json/"
PERSIST_TXT_DIR = "./chroma_db/txt/"
# -------------------- Embeddings --------------------
# embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

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
    vector_store = Chroma.from_documents(docs, embeddings, persist_directory=PERSIST_JSON_DIR)
    # vector_store.persist()
    print(f"Loaded {len(docs)} documents into Chroma.")

def load_txt_data():
    docs = store_txt_data()
    vector_store = Chroma.from_documents(docs, embeddings, persist_directory=PERSIST_JSON_DIR)
    # vector_store.persist()
    print(f"Loaded {len(docs)} documents into Chroma.") 

# -------------------- Retrieve Data --------------------



    
    #------------------load json data------------------------------------
json_vector_store = Chroma(persist_directory=PERSIST_JSON_DIR, embedding_function=embeddings)
json_vector_retriever = json_vector_store.as_retriever(search_kwargs={"k": 5})
    
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

json_rulebook_retriver=compression_retriever

#------------------TXT data------------------------------------
txt_vector_store = Chroma(persist_directory=PERSIST_TXT_DIR, embedding_function=embeddings)
txt_vector_retriever = txt_vector_store.as_retriever(search_kwargs={"k": 5})

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



txt_rulebook_retriver=txt_compression_retriever




# -------------------- Main --------------------
if __name__ == "__main__":
    # Run once to load data
    # load_json_data()
    # load_txt_data()

    # Retrieve
    
  q2=  """
    Validate that leaf GLDSSSW length does not exceed 50 characters
    Check the string length of the GLDSSSW leaf in each moi entry to ensure it is 50 characters or fewer
    """
  q1=  """
    Validate that leaf GLDSSSW length does not exceed 50 characters
    Check the string length of the GLDSSSW leaf in each moi entry to ensure it is 50 characters or fewer
    leaf type description
    """
  r1=json_rulebook_retriver.invoke(q1)
  r1=[doc.metadata["record"] for doc in r1]
#   r2=txt_rulebook_retriver.invoke(q2)
  print(f"JSON DATA: \n {r1}")
#   print(f"TXT DATA: \n {r2}")
  print("---------------------------------------------------------------------------")
  

  r1=json_ensemble_retriever.invoke(q1)
  r1=[doc.metadata["record"] for doc in r1]
#   r2=txt_ensemble_retriever.invoke(q2)
  print(f"JSON DATA: \n {r1}")
#   print(f"TXT DATA: \n {r2}")
  print("---------------------------------------------------------------------------")

  r1=json_vector_retriever.invoke(q1)
  r1=[doc.metadata["record"] for doc in r1]
#   r2=txt_vector_retriever.invoke(q2)
  print(f"JSON DATA: \n {r1}")
#   print(f"TXT DATA: \n {r2}")

  print("---------------------------------------------------------------------------")
  r1=json_bm25_retriever.invoke(q1)
  r1=[doc.metadata["record"] for doc in r1]
#   r2=txt_bm25_retriever.invoke(q2)
  print(f"JSON DATA: \n {r1}")
#   print(f"TXT DATA: \n {r2}")
  