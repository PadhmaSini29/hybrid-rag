import os
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
AWS_PROFILE = os.getenv("AWS_PROFILE")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID")
BEDROCK_EMBEDDING_MODEL_ID = os.getenv("BEDROCK_EMBEDDING_MODEL_ID")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

#How many documents to fetch initially from each retriever
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 20))

# Extraction Flags
ENABLE_IMAGE_EXTRACTION = os.getenv("ENABLE_IMAGE_EXTRACTION", "True").lower() == "true"
ENABLE_TABLE_EXTRACTION = False
ENABLE_CHART_OCR = False

# Advanced Chunking & OCR Flags
ENABLE_IMAGE_CHUNKING = os.getenv("ENABLE_IMAGE_CHUNKING", "True").lower() == "true"
ENABLE_TABLE_CHUNKING = False
ENABLE_OCR = False
ENABLE_CAPTIONING = False
ENABLE_HYBRID_LINKING = os.getenv("ENABLE_HYBRID_LINKING", "True").lower() == "true"

# CLIP / Multimodal Configuration
ENABLE_IMAGE_EMBEDDINGS = os.getenv("ENABLE_IMAGE_EMBEDDINGS", "True").lower() == "true"
ENABLE_MULTIMODAL_FUSION = os.getenv("ENABLE_MULTIMODAL_FUSION", "True").lower() == "true"
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")

# Retrieval Fusion Settings
TEXT_TOP_K = int(os.getenv("TEXT_TOP_K", 10))
IMAGE_TOP_K = int(os.getenv("IMAGE_TOP_K", 5))
FUSION_TOP_K = int(os.getenv("FUSION_TOP_K", 8))

# Output Formats & Paths
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "./artifacts")
IMAGES_DIR = os.path.join(ARTIFACTS_DIR, "images")

# Vector DB Paths
CHROMA_TEXT_DIR = os.path.join(CHROMA_DIR, "text")
CHROMA_IMAGE_DIR = os.path.join(CHROMA_DIR, "image")

# How many documents go into the LLM prompt
CONTEXT_K = int(os.getenv("CONTEXT_K", 8))

# Semantic Chunking Parameters
SEMANTIC_BREAKPOINT_PERCENTILE = float(os.getenv("SEMANTIC_BREAKPOINT_PERCENTILE", 95))
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", 100))
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
