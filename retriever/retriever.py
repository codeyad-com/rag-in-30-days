import logging
from typing import List, Dict, Any, Optional

from vector_db import QdrantDB
from embeddings import Embedder
from settings import RAGConfig

logger = logging.getLogger(__name__)


class Retriever:
    """
    Finds relevant document chunks for a user query.
    
    How it works:
    1. Take user's question
    2. Convert to vector (using Embedder)
    3. Search Qdrant for similar vectors
    4. Return top matches with scores
    """
    
    def __init__(
        self,
        config: RAGConfig,
        embedder: Optional[Embedder] = None,
        collection_name: Optional[str] = None,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ):
        """
        Initialize the retriever.
        
        Args:
            config: RAGConfig instance with all settings
            embedder: Embedder instance (creates new if None)
            collection_name: Override the default collection name (optional)
            top_k: Default number of results to return (overrides config)
            min_score: Minimum similarity score (overrides config)
        """
        self.config = config
        self.top_k = top_k if top_k is not None else config.default_top_k
        self.min_score = min_score if min_score is not None else config.min_score_threshold
        
        # Initialize components
        self.vector_db = QdrantDB(
            config=config,
            collection_name=collection_name
        )
        self.embedder = embedder if embedder is not None else Embedder(config)
        
        # Check if collection exists
        if not self.vector_db.collection_exists():
            logger.warning(
                f"Collection '{self.vector_db.collection_name}' not found. "
                "Run index_builder.py first to index your documents."
            )
        
        logger.info(f"Retriever initialized with collection: {self.vector_db.collection_name}")
        logger.info(f"Top K: {self.top_k}, Min Score: {self.min_score}")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find relevant documents for a query.
        
        Args:
            query: User's question
            top_k: How many results to return (default: self.top_k)
            min_score: Minimum similarity score (default: self.min_score)
            
        Returns:
            List of results, each containing:
            - text: The document chunk text
            - score: Similarity score (0-1)
            - metadata: Document info (doc_id, page_number, source)
        """
        top_k = top_k or self.top_k
        min_score = min_score or self.min_score
        
        # Step 1: Convert query to vector
        logger.debug(f"Embedding query: {query[:50]}...")
        query_vector = self.embedder.embed_query(query)
        
        # Step 2: Search Qdrant
        logger.debug(f"Searching Qdrant for top {top_k} results...")
        results = self.vector_db.search(
            query_vector=query_vector,
            top_k=top_k,
            min_score=min_score,
        )
        
        # Step 3: Log results
        if results:
            logger.info(f"Found {len(results)} results for query: {query[:50]}...")
            for i, r in enumerate(results, 1):
                logger.debug(f"  [{i}] Score: {r['score']:.3f} | {r['text'][:50]}...")
        else:
            logger.warning(f"No results found for query: {query[:50]}...")
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return self.vector_db.get_collection_info()
    
    def health_check(self) -> bool:
        """Check if the vector database is healthy."""
        return self.vector_db.health_check()
    
    def update_config(self, top_k: Optional[int] = None, min_score: Optional[float] = None):
        """
        Update retriever parameters at runtime (for UI).
        
        Args:
            top_k: New default top_k value
            min_score: New minimum score threshold
        """
        if top_k is not None:
            self.top_k = top_k
            logger.info(f"Updated top_k to {top_k}")
        if min_score is not None:
            self.min_score = min_score
            logger.info(f"Updated min_score to {min_score}")
    
    def close(self) -> None:
        """Clean up connections."""
        pass  # QdrantDB doesn't need explicit cleanup