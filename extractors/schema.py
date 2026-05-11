from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

@dataclass
class ExtractedArtifact:
    type: str # "image", "table", "chart"
    source_pdf: str
    page: int
    content: str # Flattened text for retrieval
    metadata: Dict[str, Any]

    def to_dict(self):
        return asdict(self)
