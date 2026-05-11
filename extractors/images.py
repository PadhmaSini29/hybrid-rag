import os
import fitz # PyMuPDF
from typing import List
from extractors.schema import ExtractedArtifact
from utils import IMAGES_DIR

def extract_images_from_pdf(pdf_path: str) -> List[ExtractedArtifact]:
    """
    Extracts images from a PDF using PyMuPDF (fitz).
    Saves images to artifacts/images/<pdf_name>/<page>_<idx>.png
    """
    pdf_name = os.path.basename(pdf_path).replace(".pdf", "")
    pdf_output_dir = os.path.join(IMAGES_DIR, pdf_name)
    os.makedirs(pdf_output_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    artifacts = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            image_filename = f"page_{page_index+1}_img_{img_index+1}.{image_ext}"
            image_path = os.path.join(pdf_output_dir, image_filename)

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            artifacts.append(ExtractedArtifact(
                type="image",
                source_pdf=pdf_path,
                page=page_index + 1,
                content=f"Image extracted from {pdf_name} page {page_index+1}",
                metadata={
                    "image_path": image_path,
                    "format": image_ext,
                    "xref": xref
                }
            ))

    doc.close()
    return artifacts
