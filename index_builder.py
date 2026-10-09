"""
Build Index Script

Indexes PDF documents into Qdrant vector database.

Usage:
    python index_builder.py                    # Index all PDFs in Data/
    python index_builder.py --file path.pdf     # Index single PDF
    python index_builder.py --rebuild           # Rebuild entire index
"""

import argparse
import logging
from typing import Dict, Any, List
import re
import pymupdf 
from hazm import sent_tokenize
import uuid
from pathlib import Path

from vector_db import QdrantDB
from embeddings import Embedder
from settings import RAGConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_index")


class PDFProcessor:
    """Process PDF files into clean, chunked text."""
    
    @staticmethod
    def extract_text(pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract text from PDF, page by page."""
        pages = []
        doc = pymupdf.open(pdf_path) 
        
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:
                pages.append({
                    "page_number": page_num,
                    "text": text,
                })
        
        doc.close()
        logger.info(f"Extracted {len(pages)} pages from {pdf_path.name}")
        return pages

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean extracted PDF text."""
        if not text:
            return ""
        
        # Remove bullets and special symbols
        text = re.sub(r"[•▪◦★✦]+", " ", text)
        
        # Remove numbering patterns (1. 1) 1:)
        text = re.sub(r"\b\d+[\.\،\:\)\-]\s*", " ", text)
        
        # Remove isolated numbers on lines
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
        
        # Merge broken lines into paragraphs
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        merged = []
        buffer = ""
        
        for line in lines:
            if buffer:
                # If buffer doesn't end with sentence-ending punctuation
                if not re.search(r"[.!؟]", buffer[-1]):
                    buffer += " " + line
                else:
                    merged.append(buffer)
                    buffer = line
            else:
                buffer = line
        
        if buffer:
            merged.append(buffer)
        
        # Normalize whitespace
        merged = [re.sub(r"\s+", " ", l) for l in merged]
        return "\n".join(merged)
    
    @staticmethod
    def chunk_text(
        pages: List[Dict[str, Any]],
        chunk_size: int,
        overlap_sentences: int,
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        """
        Split text into overlapping chunks.
        
        Args:
            pages: List of pages with text
            chunk_size: Target chunk size in characters
            overlap_sentences: Number of sentences to overlap between chunks
            
        Returns:
            chunks: List of text chunks
            metadata: List of metadata dicts
        """
        chunks = []
        metadata = []
        
        for page in pages:
            page_number = page["page_number"]
            sentences = sent_tokenize(page["text"])
            
            current_chunk = []
            
            for sent in sentences:
                current_chunk.append(sent)
                current_len = sum(len(s) for s in current_chunk) + len(current_chunk)
                
                if current_len >= chunk_size:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(chunk_text)
                    
                    metadata.append({
                        "text": chunk_text,
                        "page_number": page_number,
                        "chunk_id": str(uuid.uuid4())[:8],
                    })
                    
                    # Keep last N sentences for overlap
                    overlap = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
                    current_chunk = overlap
            
            # Final chunk for page
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                metadata.append({
                    "text": chunk_text,
                    "page_number": page_number,
                    "chunk_id": str(uuid.uuid4())[:8],
                })
        
        logger.info(f"Created {len(chunks)} chunks from {len(pages)} pages")
        return chunks, metadata











class IndexBuilder:
    """Build and manage document index in Qdrant."""
    
    def __init__(self, config: RAGConfig):
        """
        Initialize IndexBuilder with configuration.
        
        Args:
            config: RAGConfig instance with all settings
        """
        self.config = config
        self.vector_db = QdrantDB(config=config)
        self.embedder = Embedder(config=config)
        self.collection_name = config.qdrant_collection
    
    def build_collection(self, force_recreate: bool = False) -> None:
        """Create or recreate the collection."""
        self.vector_db.create_collection(force_recreate=force_recreate)
    
    def index_document(
        self,
        pdf_path: Path,
        doc_id: str,
        source_name: str,
    ) -> int:
        """
        Index a single PDF document.
        
        Returns:
            Number of chunks indexed
        """
        logger.info(f"📄 Indexing: {pdf_path.name}")
        
        # Step 1: Extract text from PDF
        pages = PDFProcessor.extract_text(pdf_path)
        
        # Step 2: Clean text
        cleaned_pages = [
            {"page_number": p["page_number"], "text": PDFProcessor.clean_text(p["text"])}
            for p in pages
        ]

    
        
        # Step 3: Chunk text (use config values)
        chunks, chunk_metadata = PDFProcessor.chunk_text(
            cleaned_pages,
            chunk_size=self.config.chunk_size,
            overlap_sentences=self.config.overlap_sentences
        )
        
        if not chunks:
            logger.warning(f"No chunks created from {pdf_path.name}")
            return 0
        
        # Step 4: Embed chunks (using embedder's unified method)
        logger.info(f"   Embedding {len(chunks)} chunks...")
        embeddings = self.embedder.embed_text(chunks)
        
        # Step 5: Prepare payloads for Qdrant
        payloads = []
        for chunk, meta in zip(chunks, chunk_metadata):
            payloads.append({
                "text": chunk,
                "doc_id": doc_id,
                "source": source_name,
                "page_number": meta["page_number"],
                "chunk_id": meta["chunk_id"],
            })
        
        # Step 6: Insert into Qdrant
        count = self.vector_db.add_vectors(
            vectors=embeddings,
            payloads=payloads,
        )
        
        logger.info(f"   ✅ Indexed {count} chunks from {pdf_path.name}")
        return count
    
    def index_directory(self, data_dir: Path, force_recreate: bool = False) -> Dict[str, int]:
        """Index all PDFs in a directory."""
        self.build_collection(force_recreate=force_recreate)
        
        results = {}
        pdf_files = list(data_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {data_dir}")
            return results
        
        logger.info(f"Found {len(pdf_files)} PDF files to index")
        
        for pdf_path in pdf_files:
            doc_id = pdf_path.stem
            try:
                count = self.index_document(
                    pdf_path=pdf_path,
                    doc_id=doc_id,
                    source_name=pdf_path.name,
                )
                results[pdf_path.name] = count
            except Exception as e:
                logger.error(f"Failed to index {pdf_path.name}: {e}")
                results[pdf_path.name] = 0
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return self.vector_db.get_collection_info()
    
    def collection_exists(self) -> bool:
        """Check if collection exists."""
        return self.vector_db.collection_exists()
    
    def delete_collection(self) -> bool:
        """Delete the entire collection."""
        return self.vector_db.delete_all()
    
    def close(self) -> None:
        """Close connections."""
        pass


def main():

    parser = argparse.ArgumentParser(description="Build document index for RAG")
    parser.add_argument("--file", type=Path, help="Index a single PDF file")
    parser.add_argument("--dir", type=Path, default=None, help="Directory with PDFs (overrides config)")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild entire index")
    parser.add_argument("--stats", action="store_true", help="Show index statistics")
    
    args = parser.parse_args()
    
    # Load configuration
    config = RAGConfig()
    
    # Override data directory if provided
    if args.dir:
        config.data_dir = args.dir
    
    builder = IndexBuilder(config=config)
    
    try:
        if args.stats:
            stats = builder.get_stats()
            print("\n📊 Index Statistics:")
            print(f"   Collection: {config.qdrant_collection}")
            print(f"   Points: {stats.get('points_count', 0)}")
            print(f"   Status: {stats.get('status', 'N/A')}")
            return
        
        if args.file:
            builder.build_collection(force_recreate=args.rebuild)
            doc_id = args.file.stem
            count = builder.index_document(
                pdf_path=args.file,
                doc_id=doc_id,
                source_name=args.file.name,
            )
            print(f"\n✅ Indexed {count} chunks from {args.file.name}")
        
        else:
            results = builder.index_directory(config.data_dir, force_recreate=args.rebuild)
            
            print("\n" + "=" * 50)
            print("✅ Indexing Complete")
            print("=" * 50)
            for filename, count in results.items():
                status = "✓" if count > 0 else "✗"
                print(f"   {status} {filename}: {count} chunks")
            
            total = sum(results.values())
            print(f"\n📊 Total: {total} chunks indexed")
            
            # Show stats
            stats = builder.get_stats()
            print(f"📊 Collection: {stats.get('points_count', 0)} points")
    
    finally:
        builder.close()


if __name__ == "__main__":
    main()