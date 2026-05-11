# Multimodal RAG with Semantic Chunking

This project implements a sophisticated RAG (Retrieval-Augmented Generation) pipeline that handles text, tables, images, and charts using a dual-modality embedding approach (Bedrock + CLIP).

## Prerequisites

1. **Python 3.8+**
2. **Tesseract OCR**:
   - Install from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).
   - Default path is expected at `C:\Program Files\Tesseract-OCR\tesseract.exe` (update in `utils.py` if different).
3. **Java Runtime (JRE)**: Required for `tabula-py` (table extraction).
4. **Ghostscript**: Required for `camelot-py` (table extraction).
5. **AWS Account**: Access to Amazon Bedrock models (Claude and Titan Embeddings).

## Installation

1. **Clone the repository** (if applicable) and navigate to the directory.
2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Update the `.env` file in the root directory:

```env
AWS_REGION=your-region
AWS_PROFILE=your-profile
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v1
CHROMA_DIR=./chroma_db
```

Additional configuration for extraction and fusion can be found in `utils.py`.

## Usage

### 1. Prepare Data
Place your PDF files in the `data/` directory.

### 2. Run Ingestion
This will process the PDFs, extract artifacts (tables, images, charts), and build the dual vector indexes (Text & Image).
```bash
python ingestion.py
```
*Note: Extracted files will be saved in the `artifacts/` folder.*

### 3. Query the System
Run the query interface:
```bash
python app.py "What is the growth trend shown in the charts on page 5?"
```

## Features

- **Semantic Chunking**: Breaks text at topic changes rather than fixed sizes.
- **Multimodal Retrieval**: Searches text using Bedrock and images/charts using CLIP.
- **Hybrid Linking**: Automatically connects text paragraphs to tables and charts on the same page.
- **Table Handling**: Automatically splits large tables into manageable row-wise chunks with summaries.
- **Chart Enrichment**: Performs OCR on graphs to extract trend summaries and axis labels for better retrieval.
