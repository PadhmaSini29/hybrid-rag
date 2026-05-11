from langchain_chroma import Chroma
from utils import *
from ingestion import get_embeddings
from embedders.clip_embedder import LangChainCLIPEmbeddings

def load_vectorstores():
    """
    Loads both text and image vector stores.
    """
    embeddings = get_embeddings()
    text_db = Chroma(
        persist_directory=CHROMA_TEXT_DIR,
        embedding_function=embeddings
    )
    
    image_db = None
    if ENABLE_IMAGE_EMBEDDINGS:
        clip_embeddings = LangChainCLIPEmbeddings(model_name=CLIP_MODEL_NAME)
        image_db = Chroma(
            persist_directory=CHROMA_IMAGE_DIR,
            embedding_function=clip_embeddings
        )
    
    return text_db, image_db

def hybrid_retrieve(query):
    """
    Performs multimodal retrieval:
    1. Text search against paragraphs/OCR.
    2. CLIP text-to-image search against images/charts.
    3. Fuses results using RRF (Reciprocal Rank Fusion).
    """
    text_db, image_db = load_vectorstores()
    
    # 1. Text retrieval
    text_results = text_db.similarity_search(query, k=TEXT_TOP_K)
    
    # 2. Image retrieval
    image_results = []
    if image_db:
        image_results = image_db.similarity_search(query, k=IMAGE_TOP_K)
    
    # 3. Reciprocal Rank Fusion (RRF)
    all_docs = {}
    
    # Helper to calculate RRF score
    def add_to_rrf(results, weight=1.0):
        for rank, doc in enumerate(results):
            # Using content as key for deduplication (or image path if image)
            doc_id = doc.metadata.get("image_path") or doc.page_content[:100]
            if doc_id not in all_docs:
                all_docs[doc_id] = {"doc": doc, "score": 0.0}
            
            # RRF formula: 1 / (rank + k)
            all_docs[doc_id]["score"] += weight * (1.0 / (rank + 60))

    add_to_rrf(text_results, weight=1.0)
    add_to_rrf(image_results, weight=1.0) # Weighted equally for now

    # Sort by score
    sorted_results = sorted(all_docs.values(), key=lambda x: x["score"], reverse=True)
    
    # Return top FUSION_K documents
    return [item["doc"] for item in sorted_results[:FUSION_TOP_K]]

def vector_retrieve(query):
    """Fallback or legacy wrapper."""
    return hybrid_retrieve(query)