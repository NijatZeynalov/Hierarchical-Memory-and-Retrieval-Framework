from typing import Dict, Optional, Tuple
import torch
from dataclasses import dataclass


@dataclass
class ProcessedQuery:
    """Processed query information."""
    embedding: torch.Tensor
    query_type: str  # 'explicit' or 'implicit'
    spatial_hint: Optional[torch.Tensor] = None
    constraints: Dict = None


class QueryProcessor:
    """Processes queries for memory retrieval."""

    def __init__(
            self,
            encoder,
            spatial_keywords: List[str] = ["near", "at", "in"]
    ):
        self.encoder = encoder
        self.spatial_keywords = spatial_keywords

    def process(
            self,
            query: str,
            context: Optional[Dict] = None
    ) -> ProcessedQuery:
        """Process and enhance query."""
        # Get query embedding
        embedding = self.encoder.encode(query)

        # Determine query type
        query_type = self._get_query_type(query)

        # Extract spatial hints
        spatial_hint = self._extract_spatial(query)

        return ProcessedQuery(
            embedding=embedding,
            query_type=query_type,
            spatial_hint=spatial_hint
        )

    def _get_query_type(self, query: str) -> str:
        """Determine if query is explicit or implicit."""
        explicit_keywords = ["find", "locate", "where", "go to"]
        return "explicit" if any(k in query.lower() for k in explicit_keywords) else "implicit"

    def _extract_spatial(self, query: str) -> Optional[torch.Tensor]:
        """Extract spatial information from query."""
        for keyword in self.spatial_keywords:
            if keyword in query.lower():
                # Get text after spatial keyword
                spatial_text = query.lower().split(keyword)[1].strip()
                return self.encoder.encode(spatial_text)
        return None