import sys
from ingestion import load_or_build_indexes
from retriever import vector_retrieve
from llm import generate_answer
from utils import CONTEXT_K

"""
    Initializes the retrieval system by loading the existing vector index or building it if missing.
    Calls load_or_build_indexes() which:
        - Loads Chroma (vector DB)
        - Builds it if not found
        - test 2345
"""
# Load or build the vector 
text_db, image_db = load_or_build_indexes()

def ask(query):
    """
        Main pipeline function that processes a user query and returns an answer.
        Combines semantic retrieval + LLM generation to produce accurate responses.
        
        Step 1: Retrieve documents using semantic search
            - Vector search (semantic similarity)
            - Optional term-overlap re-ranking

        Step 2: Select top-k context
            - Limits context size using CONTEXT_K
            - Prevents LLM overload and improves relevance

        Step 3: Generate answer
            - Sends query + context to LLM
            - Produces final response
    """
    vector_docs = vector_retrieve(query)

    # Use only top CONTEXT_K for LLM
    context_docs = vector_docs[:CONTEXT_K]

    answer = generate_answer(query, context_docs)

    return answer

if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0].lower() == "ask":
        args = args[1:]
    query = " ".join(args)
    print(ask(query))