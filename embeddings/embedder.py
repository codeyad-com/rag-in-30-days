from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from openai import OpenAI
from settings import RAGConfig


class Embedder:
    """
    Embedder supporting both API and local models.
    
    Usage:
        config = RAGConfig()
        embedder = Embedder(config)
        embeddings = embedder.embed_text(["hello world"])
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self.embedder_type = config.embedder_type

        if self.embedder_type == "local":
            self.model = SentenceTransformer(
                config.local_embedding_model,
                device="cpu"
            )
            print(f"[Embedder] Local model loaded: {config.local_embedding_model}")
        elif self.embedder_type == "api":
            self.client = OpenAI(
                base_url=config.embedding_base_url,
                api_key=config.embedding_api_key
            )
            self.model_name = config.embedding_model
            print(f"[Embedder] API mode using model: {self.model_name}")
        else:
            raise ValueError(f"Invalid embedder_type: {self.embedder_type}. Must be 'api' or 'local'")

    def embed_text(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of text chunks (for indexing documents).
        """
        if self.embedder_type == "local":
            prefixed = ["passage: " + t for t in texts]
            return self.model.encode(
                prefixed,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False
            )
        else:  # api
            prefixed = [f"passage: {t}" for t in texts]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=prefixed
            )
            return np.array([item.embedding for item in response.data])

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string (for searching).
        """
        if self.embedder_type == "local":
            result = self.model.encode(
                ["query: " + query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return result[0]
        else:  # api
            response = self.client.embeddings.create(
                model=self.model_name,
                input=[f"query: {query}"]
            )
            return np.array(response.data[0].embedding)