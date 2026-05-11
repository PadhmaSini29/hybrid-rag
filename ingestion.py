import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_aws import BedrockEmbeddings
from langchain_core.documents import Document
import boto3
from utils import *
from extractors.images import extract_images_from_pdf
from embedders.clip_embedder import LangChainCLIPEmbeddings
import uuid

def load_pdfs(data_path="data"):
    """
        Loads all PDF files from a given directory and converts them into LangChain Document objects.
        Used during the ingestion phase before building vector indexes.
        - Iterates through all files in the given directory.
        - Filters only `.pdf` files.
        - Uses PyPDFLoader to read each PDF.
    """
    docs = []
    for file in os.listdir(data_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_path, file))
            docs.extend(loader.load())
    return docs

import shutil
import re

def clean_text(text):
    """
        Cleans raw text extracted from PDFs by removing unwanted noise.
        PDF extraction often includes page numbers, footers, and artifacts that reduce
        retrieval quality and embedding accuracy.
        Used before splitting documents into chunks.
        - Removes patterns like "Page X of Y".
        - Removes standalone numeric lines (page numbers).
        - Uses regex for pattern matching and replacement.
    """
    # Remove page numbers and common footers
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
    return text

def split_docs(docs, embeddings):
    """
        Splits documents into semantically meaningful chunks using embedding-based similarity.
        - Uses SemanticChunker to find breakpoints where the topic changes.
        - The breakpoint threshold is controlled by SEMANTIC_BREAKPOINT_PERCENTILE.
        - Combines all document text and cleans it before splitting.
        - Filters out:
            * Very small chunks (<100 chars)
            * Table of contents-like noise (many dots)
    """
    # Combine and clean
    print(f"Combining and cleaning {len(docs)} documents...")
    full_text = ""
    for doc in docs:
        full_text += clean_text(doc.page_content) + "\n"

    print(f"Total characters to process: {len(full_text)}")
    print(f"Initializing SemanticChunker with percentile={SEMANTIC_BREAKPOINT_PERCENTILE}...")

    # Initialize SemanticChunker
    # percentile: distance > Pth percentile of all distances between adjacent sentences are split.
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=SEMANTIC_BREAKPOINT_PERCENTILE
    )

    # Split the text
    print("Performing semantic splitting (this may take a while depending on text length)...")
    semantic_docs = splitter.create_documents([full_text])
    print(f"Generated {len(semantic_docs)} initial semantic segments.")
    
    # Second-stage splitting for oversized semantic chunks
    print(f"Applying second-stage splitting (MAX_SIZE={MAX_CHUNK_SIZE}) and filtering...")
    
    second_stage_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    final_chunks = []
    for doc in semantic_docs:
        text = doc.page_content.strip()
        
        # Skip if too small
        if len(text) < MIN_CHUNK_SIZE:
            continue
            
        # TOC filter
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if any(l.count('.') > 10 for l in lines):
            continue
            
        # If the semantic chunk is too large, split it further
        if len(text) > MAX_CHUNK_SIZE:
            sub_chunks = second_stage_splitter.create_documents([text])
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(doc)
            
    return final_chunks

def get_embeddings():
    """
        Initializes the embedding model using AWS Bedrock.
    """
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    return BedrockEmbeddings(
        client=session.client("bedrock-runtime"),
        model_id=BEDROCK_EMBEDDING_MODEL_ID
    )


def has_chroma_data():
    """
        Checks if Chroma vector database already exists.
    """
    return os.path.isdir(CHROMA_DIR) and any(os.scandir(CHROMA_DIR))

def build_indexes():
    """
        Builds dual modality vector indexes (Text: Bedrock, Image: CLIP).
    """
    # Clear existing data
    try:
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
        if os.path.exists(ARTIFACTS_DIR):
            shutil.rmtree(ARTIFACTS_DIR)
    except PermissionError:
        print(f"Warning: Could not clear directories. Proceeding...")

    data_path = "data"
    all_pdf_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".pdf")]
    
    text_artifacts = []
    image_artifacts = []
    raw_text_docs = []

    for pdf_path in all_pdf_files:
        print(f"Processing {pdf_path}...")
        
        # 1. Load raw text
        loader = PyPDFLoader(pdf_path)
        raw_text_docs.extend(loader.load())

        # 2. Extract Images
        if ENABLE_IMAGE_EXTRACTION:
            print(f"  Extracting images...")
            images = extract_images_from_pdf(pdf_path)
            image_artifacts.extend(images)

    # Convert to Documents
    text_art_docs = [Document(page_content=art.content, metadata=art.metadata) for art in text_artifacts]
    # For CLIP, page_content should be the image path so it can be embedded as an image
    image_art_docs = [Document(page_content=art.metadata.get("image_path", art.content), metadata=art.metadata) for art in image_artifacts]

    bedrock_embeddings = get_embeddings()
    
    # Split text docs
    print("Splitting text documents into semantic chunks...")
    text_chunks = split_docs(raw_text_docs, bedrock_embeddings)
    
    # Hybrid Linking (Optional enrichment)
    if ENABLE_HYBRID_LINKING:
        print("Applying hybrid linking...")
        # We link text_chunks to image_art_docs and text_art_docs
        all_combined = apply_hybrid_linking(text_chunks, text_art_docs + image_art_docs)
        # Extract them back out
        text_chunks = [d for d in all_combined if d in text_chunks]
        text_art_docs = [d for d in all_combined if d in text_art_docs]
        image_art_docs = [d for d in all_combined if d in image_art_docs]

    # 1. Text Index (paragraphs + OCR + table summaries)
    print(f"Building Text Chroma index at {CHROMA_TEXT_DIR}...")
    final_text_docs = text_chunks + text_art_docs
    text_db = Chroma.from_documents(
        documents=final_text_docs,
        embedding=bedrock_embeddings,
        persist_directory=CHROMA_TEXT_DIR
    )

    # 2. Image Index (CLIP)
    if ENABLE_IMAGE_EMBEDDINGS:
        print(f"Building Image Chroma index at {CHROMA_IMAGE_DIR}...")
        clip_embeddings = LangChainCLIPEmbeddings(model_name=CLIP_MODEL_NAME)
        
        image_db = Chroma.from_documents(
            documents=image_art_docs,
            embedding=clip_embeddings,
            persist_directory=CHROMA_IMAGE_DIR
        )
        return text_db, image_db

    return text_db, None

def apply_hybrid_linking(text_chunks, artifact_docs):
    """
    Links text chunks to artifact chunks (images, tables, charts) 
    if they share the same page and PDF source.
    """
    import uuid
    
    # Create a mapping of (source, page) -> list of artifact IDs
    artifact_map = {}
    for i, art in enumerate(artifact_docs):
        source = art.metadata.get("source_pdf")
        page = art.metadata.get("page")
        if not source or page is None:
            continue
        
        key = (source, page)
        if key not in artifact_map:
            artifact_map[key] = []
        
        # Assign a unique link_id if not present
        if "link_id" not in art.metadata:
            art.metadata["link_id"] = str(uuid.uuid4())
        
        artifact_map[key].append(art.metadata["link_id"])

    # Link text chunks to artifacts on the same page
    for chunk in text_chunks:
        source = chunk.metadata.get("source") # PyPDFLoader uses 'source'
        page = chunk.metadata.get("page")
        if not source or page is None:
            continue
        
        # Normalize page (PyPDFLoader is 0-indexed, my extractors are 1-indexed)
        # Actually let's check what PyPDFLoader uses. It usually starts from 0.
        # My extractors used 1-indexed (page_index + 1).
        key = (source, page + 1)
        
        if key in artifact_map:
            chunk.metadata["related_artifact_ids"] = artifact_map[key]
            chunk.metadata["has_related_artifacts"] = True

    return text_chunks + artifact_docs

def load_or_build_indexes():
    """
        Loads existing vector indexes or builds new ones if missing.
        Returns (text_db, image_db).
    """
    if not has_chroma_data():
        return build_indexes()

    from retriever import load_vectorstores
    return load_vectorstores()

if __name__ == "__main__":
    build_indexes()
    print(" Ingestion complete")