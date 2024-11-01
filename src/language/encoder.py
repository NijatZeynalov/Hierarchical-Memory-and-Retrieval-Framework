import torch
from typing import Union, List, Dict
from transformers import AutoTokenizer, AutoModel
import numpy as np


class TextEncoder:
    """Encodes text and visual inputs into embeddings for memory storage."""

    def __init__(
            self,
            model_name: str = "sentence-transformers/all-mpnet-base-v2",
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            max_length: int = 128
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device
        self.max_length = max_length

    def encode(
            self,
            text: Union[str, List[str]]
    ) -> torch.Tensor:
        """Encode text into embedding."""
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self._pool_outputs(outputs)

        return embeddings

    def encode_batch(
            self,
            texts: List[str]
    ) -> torch.Tensor:
        """Encode batch of texts."""
        return self.encode(texts)

    def _pool_outputs(
            self,
            outputs
    ) -> torch.Tensor:
        """Pool transformer outputs into single embedding."""
        return outputs.last_hidden_state.mean(dim=1)

    def compute_similarity(
            self,
            emb1: torch.Tensor,
            emb2: torch.Tensor
    ) -> torch.Tensor:
        """Compute cosine similarity between embeddings."""
        return torch.nn.functional.cosine_similarity(emb1, emb2)