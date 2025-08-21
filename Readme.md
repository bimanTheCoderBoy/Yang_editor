## Core Components

The system is composed of three main components:

1.  **AST Store (`app/store/AST_store.py`):** An in-memory, thread-safe repository for storing and managing Abstract Syntax Tree (AST) objects. This is crucial because the `langgraph` library, used for orchestrating the pipeline, cannot handle non-JSON serializable objects like the `pyang` ASTs.

2.  **RAG Repository (`app/rag/repo.py`):** A Retrieval-Augmented Generation (RAG) component responsible for providing relevant context and rules to the language models. It uses a hybrid retrieval approach (keyword-based and vector-based) with reranking to fetch information from both JSON and text-based rulebooks.

3.  **Graph Pipeline (`app/graph/pipeline.py`):** The core of the system, built using `langgraph`. It defines a stateful graph where each node represents a step in the process of updating the YANG model. This pipeline orchestrates the entire workflow, from initial validation to the final generation of the updated YANG model.

## Detailed Data Flow

The data flow is orchestrated by the `langgraph` pipeline defined in `app/graph/pipeline.py`. Here is a step-by-step breakdown:

**Step 1: Initial YANG Validation (`validate_initial_yang` node)**

*   **Input:** The process begins with the user providing a YANG model as a string of text.
*   **Process:** The `validate_initial_yang` function takes this text and uses the `validate_yang_text` utility to check for syntax errors.
*   **Output:**
    *   If the validation is successful, the pipeline transitions to the `AST_CONVERTER` state.
    *   If validation fails, the process terminates and returns an error message to the user.

**Step 2: AST Conversion (`convert_to_ast` node)**

*   **Input:** The validated YANG model text.
*   **Process:**
    1.  The `get_yang_ast` function parses the YANG text and converts it into an AST object.
    2.  This AST object is then saved to the `AST_store`, which returns a unique `ast_id` for future retrieval.
    3.  A summary of the AST is generated using `generate_ast_summary`.
*   **Output:** The pipeline transitions to the `ACTIONLIST_BUILDER` state, with the `ast_id` and `ast_summary` added to the agent's state.

**Step 3: Action List Generation (`actionlist_builder` node)**

*   **Input:** The user's natural language query and the summary of the AST.
*   **Process:**
    1.  A large language model (`llm_actioner_splitter`) is invoked with a prompt containing the user's query and the AST summary.
    2.  The LLM's task is to understand the user's intent and break it down into a structured list of single-unit actions (`ActionList`). Each action in the list specifies the task and the target YANG identifiers involved.
*   **Output:** The pipeline transitions to the `ACTION_EXECUTER` state, with the generated `action_list` added to the agent's state.

**Step 4: Action Execution (`action_executer` node)**

This is the most complex and critical part of the pipeline. It iterates through the generated actions and applies them to the AST.

*   **Input:** The `action_list` and the `ast_id`.
*   **Process:**
    1.  The AST is retrieved from the `AST_store` using the `ast_id`.
    2.  The actions are grouped by their target YANG identifiers.
    3.  For each target identifier, the corresponding node in the AST is located.
    4.  **RAG-based Context Retrieval:**
        *   A query is constructed based on the AST node's content and the actions to be performed.
        *   This query is used to invoke the RAG system, which retrieves relevant rules and examples from both the JSON and text-based rulebooks.
    5.  **LLM-based AST Modification:**
        *   A second, more specialized LLM (`llm_action_executer`) is invoked.
        *   The prompt for this LLM includes the specific AST node to be modified, the list of actions, and the contextual information retrieved by the RAG system.
        *   The LLM modifies the AST node according to the instructions and returns the updated node in a structured JSON format.
    6.  **Verification and Retry Mechanism:**
        *   The main AST is updated with the modified node.
        *   The entire AST is then verified for correctness using `verify_ast`.
        *   If the verification fails, the system retries the modification process up to two times. If it continues to fail, the entire execution is restarted from the `AST_CONVERTER` step (2 times). If the issue persists, the process terminates with an error.

**Step 5: Final Output**

*   **Input:** The fully updated and verified AST.
*   **Process:** The `get_yang_text` function converts the final AST back into a human-readable YANG model text.
*   **Output:** The updated YANG model text is returned to the user.

## Architecture Diagram

```
+-----------------+      +----------------------+      +------------------------+
|  User Input     |----->| validate_initial_yang|----->|     convert_to_ast     |<-------Restart-------------------<-----
| (YANG text,     |      +----------------------+      +------------------------+                                       |
|  user_query)    |                                               |                                                     |
+-----------------+                                               |                                                     |
      ^                                                           |                                                     ^
      |                                                           v                                                     |
      |                                               +------------------------+                                        |
      |                                               |    actionlist_builder  |                                        |
      |                                               | (LLM for planning)     |                                        |
      |                                               +------------------------+                                        |
      |                                                           |                                                     |
      |                                                           |                                                     |
      |                                                           v                                                     |
+----------------------+                               +------------------------+                                       |
|   Final YANG Model   |<------------------------------|     action_executer    |<----------------------------------------
+----------------------+                               +-----------+------------+                                        |
                                                                    |                                                    |
                                                                    |                                                    |
                                          +-------------------------+-------------------------+                          |
                                          |                                                   |                          |
                                          v                                                   v                          |
                               +----------------------+                            +------------------------+            |
                               | RAG System           |                            |   Action              |             |
                               | (JSON & TXT Rulebooks)|                           | Action Detailed       |             |
                               +----------------------+                            | Path and Node Data    |             |
                                                                                    +------------------------+           |
                                          |                                                   |                          ^
                                          |                                                   |                          |
                                          v                                                   v                          |
                                          +--------------------------|-------------------------+                         |
                                                                     |                                                   |
                                                        +------------------------+                                       |
                                                        | LLM for Code Generation|                                       |
                                                        | (AST Node Editor)      |                                       |
                                                        +------------------------+                                       |
                                                                    |                                                    |
                                                                    |                                                    |
                                                        +------------------------+                                       |
                                                        | Update that Node into  |                                       |
                                                        | Previous AST           |                                       |
                                                        +------------------------+                                       |
                                                                    |                                                    |
                                                        +------------------------+                                       |
                                                        | AST VALIDATION         |--------------Retry / Next Action------        
                                                        +------------------------+



```


## THERE WILL SOME IMPROVEMENT POINTS


1) VVI (RAG)

    ----------------------------------RAG-------------------------------------------------
    Here We are using hybrid search (BM25 + vector (chroma)) + reranking
            JSON Structured RAG 
            TXT Unstrutured RAG
    Here a Lot of improvement point that we need to do to understand and get the right doc for right user Action


    -Adjust Embedding model
    -Continous JSON RULE improvement
    -Right Retrieval Stretagy
    -Adhust Reranking Model

2)  Adjust LLM


-------------------NOT NEEDED THAT MUCH FOR NOW---------------------------------------------------------------------
3)  Positional discrepancy
4)  POST proccessing RULE based function(need data and thousands of question)