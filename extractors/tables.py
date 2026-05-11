import os
import camelot
import tabula
import pandas as pd
from typing import List
from extractors.schema import ExtractedArtifact
from utils import TABLES_DIR, TABLE_OUTPUT_FORMAT

def extract_tables_from_pdf(pdf_path: str) -> List[ExtractedArtifact]:
    """
    Extracts tables using Camelot (primary) and Tabula (fallback).
    Saves tables as CSV/JSON.
    """
    pdf_name = os.path.basename(pdf_path).replace(".pdf", "")
    pdf_output_dir = os.path.join(TABLES_DIR, pdf_name)
    os.makedirs(pdf_output_dir, exist_ok=True)

    artifacts = []

    # 1. Try Camelot (Lattice mode first, then Stream)
    try:
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
        if len(tables) == 0:
            tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
        
        for i, table in enumerate(tables):
            df = table.df
            process_table_df(df, i, table.page, pdf_path, pdf_output_dir, artifacts)
            
    except Exception as e:
        print(f"Camelot failed for {pdf_path}: {e}. Falling back to Tabula.")
        # 2. Fallback to Tabula
        try:
            tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
            for i, df in enumerate(tables):
                # Tabula doesn't easily give page numbers per table in the same way, 
                # but we'll default to 0 or try to infer if needed.
                process_table_df(df, i, 0, pdf_path, pdf_output_dir, artifacts)
        except Exception as e2:
            print(f"Tabula also failed for {pdf_path}: {e2}")

    return artifacts

def process_table_df(df, index, page, pdf_path, output_dir, artifacts):
    if df.empty:
        return

    pdf_name = os.path.basename(pdf_path).replace(".pdf", "")
    base_filename = f"page_{page}_table_{index}"
    
    csv_path = os.path.join(output_dir, f"{base_filename}.csv")
    json_path = os.path.join(output_dir, f"{base_filename}.json")

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records")

    # 1. Check if table is large
    num_rows = len(df)
    if num_rows > TABLE_MAX_ROWS_PER_CHUNK:
        # Split into row groups
        for start_row in range(0, num_rows, TABLE_MAX_ROWS_PER_CHUNK):
            end_row = min(start_row + TABLE_MAX_ROWS_PER_CHUNK, num_rows)
            chunk_df = df.iloc[start_row:end_row]
            table_text = chunk_df.to_string(index=False)
            
            artifacts.append(ExtractedArtifact(
                type="table_part",
                source_pdf=pdf_path,
                page=page,
                content=f"Table part ({start_row}-{end_row}) from {pdf_name} page {page}:\n{table_text}",
                metadata={
                    "csv_path": csv_path,
                    "json_path": json_path,
                    "table_id": f"{pdf_name}_{page}_{index}",
                    "row_range": f"{start_row}-{end_row}",
                    "is_part": True
                }
            ))
        
        # Optionally add a summary chunk if enabled
        if TABLE_SUMMARY_IF_LARGE:
            summary_text = f"Large table found in {pdf_name} page {page} with {num_rows} rows. Columns: {', '.join(df.columns.tolist())}"
            artifacts.append(ExtractedArtifact(
                type="table_summary",
                source_pdf=pdf_path,
                page=page,
                content=summary_text,
                metadata={
                    "table_id": f"{pdf_name}_{page}_{index}",
                    "is_summary": True
                }
            ))
    else:
        # Small table - single chunk
        table_text = df.to_string(index=False)
        artifacts.append(ExtractedArtifact(
            type="table",
            source_pdf=pdf_path,
            page=page,
            content=f"Table data from {pdf_name} page {page}:\n{table_text}",
            metadata={
                "csv_path": csv_path,
                "json_path": json_path,
                "table_id": f"{pdf_name}_{page}_{index}"
            }
        ))
