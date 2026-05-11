import os
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import List, Union
import numpy as np

class CLIPEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        if isinstance(text, str):
            text = [text]
        inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features.cpu().numpy()

    def embed_image(self, image_paths: Union[str, List[str]]) -> np.ndarray:
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        
        images = [Image.open(path) for path in image_paths]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features.cpu().numpy()

# For LangChain compatibility
from langchain_core.embeddings import Embeddings

class LangChainCLIPEmbeddings(Embeddings):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.client = CLIPEmbedder(model_name)

    def _is_image_path(self, text: str) -> bool:
        return os.path.isfile(text) and text.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            if self._is_image_path(text):
                # Embed as image
                emb = self.client.embed_image(text)
            else:
                # Embed as text
                emb = self.client.embed_text(text)
            embeddings.append(emb[0].tolist())
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        # Queries are always text
        embedding = self.client.embed_text(text)
        return embedding[0].tolist()
