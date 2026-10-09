"""
RAG Agent - Orchestrates the retrieval and generation pipeline.

Flow:
1. User asks a question
2. Retriever finds relevant documents
3. Context is built with citations
4. LLM generates answer based on context
"""

import logging
from typing import List, Dict, Any, Optional

from retriever import Retriever
from agent.llm_wrapper import LLMWrapper
from settings import RAGConfig

logger = logging.getLogger(__name__)


class RAGAgent:
    """
    RAG (Retrieval-Augmented Generation) Agent.
    
    Combines document retrieval with LLM generation to provide
    accurate, sourced answers.
    """
    
    def __init__(
        self,
        config: RAGConfig,
        retriever: Optional[Retriever] = None,
        llm: Optional[LLMWrapper] = None,
        top_k: Optional[int] = None,
    ):
        """
        Initialize RAG agent.
        
        Args:
            config: RAGConfig instance with all settings
            retriever: Retriever instance (creates new if None)
            llm: LLM wrapper instance (creates new if None)
            top_k: Number of documents to retrieve (overrides config)
        """
        self.config = config
        self.top_k = top_k if top_k is not None else config.default_top_k
        
        # Initialize components (inject config)
        self.retriever = retriever or Retriever(config=config)
        self.llm = llm or LLMWrapper(config=config)
        
        logger.info(f"RAG Agent initialized with top_k={self.top_k}")
    
    def _build_context_and_sources(
        self, 
        results: List[Dict[str, Any]]
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Build context string and sources list from retrieval results.
        
        Args:
            results: List of retrieved documents from retriever
            
        Returns:
            Tuple of (context_text, sources_list)
        """
        context_blocks = []
        sources = []
        
        for i, result in enumerate(results, 1):
            # Create citation
            source_info = result["metadata"]
            citation = (
                f"[{i}] (منبع: {source_info.get('source', 'نامشخص')}, "
                f"صفحه: {source_info.get('page_number', 'نامشخص')})"
            )
            
            # Add to context
            context_blocks.append(f"{citation}\n{result['text']}")
            
            # Track source
            sources.append({
                "citation_number": i,
                "source": source_info.get("source", ""),
                "page": source_info.get("page_number", 0),
                "score": result["score"],
                "text_preview": result['text'][:200] + "..." if len(result['text']) > 200 else result['text'],
            })
        
        context_text = "\n\n".join(context_blocks)
        return context_text, sources
    
    def answer(
        self,
        query: str,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG.
        
        Args:
            query: User's question
            top_k: Number of documents to retrieve (overrides instance default)
            temperature: LLM temperature for this query only (optional)
            
        Returns:
            Dict with:
            - answer: The generated answer
            - sources: List of source documents with citations
            - query: The original query
            - num_sources: Number of sources used
        """
        top_k = top_k or self.top_k
        
        logger.info(f"Processing query: {query[:50]}...")
        
        # Step 1: Retrieve relevant documents
        results = self.retriever.retrieve(query, top_k=top_k)
        
        if not results:
            logger.warning(f"No documents found for query: {query[:50]}...")
            return {
                "answer": "متأسفم، اطلاعات مرتبطی در مستندات پیدا نشد.",
                "sources": [],
                "query": query,
                "num_sources": 0,
            }
        
        # Step 2: Build context with citations
        context_text, sources = self._build_context_and_sources(results)
        
        # Step 3: Build prompt using template from config
        prompt = self.config.rag_prompt_template.format(
            context=context_text,
            query=query
        )
        
        # Step 4: Generate answer (with optional temperature override)
        if temperature is not None:
            original_temp = self.llm.temperature
            self.llm.update_config(temperature=temperature)
            answer = self.llm.generate(prompt)
            self.llm.update_config(temperature=original_temp)
        else:
            answer = self.llm.generate(prompt)
        
        logger.info(f"Generated answer for query: {query[:50]}... (used {len(sources)} sources)")
        
        return {
            "answer": answer,
            "sources": sources,
            "query": query,
            "num_sources": len(sources),
        }
    
    def answer_stream(
        self,
        query: str,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
    ):
        """
        Answer a question with streaming output.
        
        Args:
            query: User's question
            top_k: Number of documents to retrieve
            temperature: LLM temperature for this query only
            
        Yields:
            Chunks of the answer
        """
        top_k = top_k or self.top_k
        
        # Retrieve documents
        results = self.retriever.retrieve(query, top_k=top_k)
        
        if not results:
            yield "متأسفم، اطلاعات مرتبطی در مستندات پیدا نشد."
            return
        
        # Build context and sources
        context_text, sources = self._build_context_and_sources(results)
        
        # Build prompt using template from config
        prompt = self.config.rag_prompt_template.format(
            context=context_text,
            query=query
        )
        
        # Stream response (with optional temperature override)
        if temperature is not None:
            original_temp = self.llm.temperature
            self.llm.update_config(temperature=temperature)
            for chunk in self.llm.generate_stream(prompt):
                yield chunk
            self.llm.update_config(temperature=original_temp)
        else:
            for chunk in self.llm.generate_stream(prompt):
                yield chunk
    
    def update_config(
        self,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Update agent configuration at runtime (for UI).
        
        Args:
            top_k: New number of documents to retrieve
            temperature: New LLM temperature
            system_prompt: New system prompt for LLM
        """
        if top_k is not None:
            self.top_k = top_k
            self.retriever.update_config(top_k=top_k)
            logger.info(f"Updated agent top_k to {top_k}")
        
        if temperature is not None or system_prompt is not None:
            self.llm.update_config(
                temperature=temperature,
                system_prompt=system_prompt
            )
    
    def health_check(self) -> Dict[str, bool]:
        """
        Check health of all components.
        
        Returns:
            Dict with health status of each component
        """
        return {
            "qdrant": self.retriever.health_check(),
            "llm_api": self.llm.health_check(),
            "collection_exists": self.retriever.vector_db.collection_exists(),
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics from retriever."""
        return self.retriever.get_stats()