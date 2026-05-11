import os
import cv2
import pytesseract
from typing import List
from extractors.schema import ExtractedArtifact
from utils import IMAGES_DIR, TESSERACT_CMD

# Configure tesseract path
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def extract_charts_from_pdf(pdf_path: str, image_artifacts: List[ExtractedArtifact]) -> List[ExtractedArtifact]:
    """
    Identifies charts among extracted images and performs OCR on them.
    Heuristic: Charts often have high edge density or specific aspect ratios, 
    but for now we'll treat all images as potential charts and run OCR to find labels.
    """
    chart_artifacts = []
    pdf_name = os.path.basename(pdf_path).replace(".pdf", "")

    for img_art in image_artifacts:
        img_path = img_art.metadata.get("image_path")
        if not img_path or not os.path.exists(img_path):
            continue

        # Basic heuristic for "is it a chart?" 
        # (Could be expanded with a proper classifier)
        is_chart, chart_type, ocr_text, trend_summary = analyze_image_for_chart(img_path)

        if is_chart:
            # Create caption (mock or placeholder for now)
            caption = f"Figure from {pdf_name} page {img_art.page}"
            
            content = f"Chart/Graph found in {pdf_name} page {img_art.page}.\n"
            content += f"Caption: {caption}\n"
            content += f"Trend Summary: {trend_summary}\n"
            content += f"OCR Labels/Text: {ocr_text}"

            chart_artifacts.append(ExtractedArtifact(
                type="chart",
                source_pdf=pdf_path,
                page=img_art.page,
                content=content,
                metadata={
                    "image_path": img_path,
                    "chart_type": chart_type,
                    "ocr_text": ocr_text,
                    "caption": caption,
                    "trend_summary": trend_summary,
                    "axis_labels": extract_axis_labels(ocr_text)
                }
            ))
    
    return chart_artifacts

def extract_axis_labels(ocr_text: str):
    """Mock axis label extraction from OCR text."""
    # Simple heuristic: lines that look like numbers or months
    labels = [word for word in ocr_text.split() if len(word) > 2]
    return labels[:5] # Return first 5 for now

def analyze_image_for_chart(image_path: str):
    """
    Uses OpenCV and Tesseract to identify if an image is likely a chart.
    Returns (is_chart, chart_type, ocr_text, trend_summary)
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False, "unknown", "", "none"

        # Perform OCR
        ocr_text = pytesseract.image_to_string(img).strip()
        
        # Simple heuristic: if OCR contains common chart words
        chart_keywords = ["axis", "value", "figure", "chart", "graph", "percent", "%", "total", "revenue", "growth"]
        found_keywords = [w for w in chart_keywords if w in ocr_text.lower()]
        
        # If we find keywords or the image has a lot of lines/edges, we tag it
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        edge_density = (edges > 0).mean()

        is_chart = len(found_keywords) > 0 or edge_density > 0.05
        chart_type = "detected_chart" if is_chart else "image"

        # Trend heuristic (very basic)
        trend_summary = "Stable"
        if "increase" in ocr_text.lower() or "growth" in ocr_text.lower():
            trend_summary = "Upward Trend"
        elif "decrease" in ocr_text.lower() or "decline" in ocr_text.lower():
            trend_summary = "Downward Trend"

        return is_chart, chart_type, ocr_text, trend_summary
    except Exception as e:
        print(f"Error analyzing chart {image_path}: {e}")
        return False, "error", "", "error"
