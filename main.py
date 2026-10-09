"""
Main entry point for testing RAG system in terminal with verbose output.

Usage:
    python main.py
"""

import logging
from settings import RAGConfig
from agent import RAGAgent
from index_builder import IndexBuilder

# Configure logging to show everything
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def print_header():
    """Print application header."""
    print("\n" + "=" * 70)
    print("🤖 RAG Agent - Terminal Interface")
    print("=" * 70)
    print("   Type your questions below (or 'exit' to quit)")
    print("   The system will show you each step:")
    print("     1️⃣  Query embedding")
    print("     2️⃣  Vector search in Qdrant")
    print("     3️⃣  Retrieved chunks with scores")
    print("     4️⃣  Context construction")
    print("     5️⃣  LLM generation")
    print("=" * 70)


def print_step(step_num: int, title: str, icon: str = "🔍"):
    """Print a step header."""
    print(f"\n{icon} Step {step_num}: {title}")
    print("-" * 50)


def print_embedding_info(query: str, vector_preview: list):
    """Show embedding step details."""
    print(f"   📝 Original query: {query}")
    print(f"   📊 Vector dimension: {len(vector_preview)}")
    print(f"   🔢 Vector preview: [{', '.join([f'{x:.4f}' for x in vector_preview[:5]])}...]")


def print_retrieval_results(results: list):
    """Show retrieved chunks with scores."""
    if not results:
        print("   ❌ No results found")
        return
    
    print(f"   ✅ Found {len(results)} results:")
    for i, r in enumerate(results, 1):
        print(f"\n   ┌─ Result {i} ─────────────────────")
        print(f"   │ Score: {r['score']:.4f}")
        print(f"   │ Source: {r['metadata']['source']}")
        print(f"   │ Page: {r['metadata']['page_number']}")
        preview = r['text'][:150].replace("\n", " ")
        print(f"   │ Preview: {preview}...")
        print(f"   └────────────────────────────────")


def print_context_info(context_text: str, num_sources: int):
    """Show context built for LLM."""
    print(f"   📦 Context size: {len(context_text)} characters")
    print(f"   📚 Sources included: {num_sources}")
    preview = context_text[:200].replace("\n", " ")
    print(f"   📄 Context preview: {preview}...")


def print_llm_response(answer: str):
    """Show LLM response."""
    print(f"\n💡 FINAL ANSWER:")
    print("-" * 50)
    print(answer)
    print("-" * 50)


def check_health(agent: RAGAgent) -> bool:
    """Check system health before starting."""
    print("\n🔍 Checking system health...")
    
    health = agent.health_check()
    
    if not health["qdrant"]:
        print("   ❌ Qdrant: NOT REACHABLE")
        print("      Please start Qdrant: docker run -p 6333:6333 qdrant/qdrant")
        return False
    else:
        print("   ✅ Qdrant: Connected")
    
    if not health["llm_api"]:
        print("   ⚠️ LLM API: NOT RESPONDING")
        print("      Check your API key and connection")
    else:
        print("   ✅ LLM API: Ready")
    
    if not health["collection_exists"]:
        print("   ⚠️ Collection: DOES NOT EXIST")
        print("      Run: python index_builder.py")
        return False
    else:
        print("   ✅ Collection: Exists")
    
    return True


def interactive_loop(agent: RAGAgent):
    """Main interactive loop with verbose output."""
    
    while True:
        print("\n" + "=" * 70)
        user_query = input("🔷 Your question (or 'exit'): ").strip()
        
        if not user_query:
            continue
        
        if user_query.lower() in ["exit", "quit", "bye", "خروج"]:
            print("\n👋 Goodbye!")
            break
        
        print("\n" + "🚀 " + "=" * 67)
        print(f"📖 PROCESSING QUERY: {user_query}")
        print("=" * 70)
        
        try:
            # ===== STEP 1: Embedding =====
            print_step(1, "Embedding Query", "🔢")
            
            # Get the embedder from retriever
            embedder = agent.retriever.embedder
            query_vector = embedder.embed_query(user_query)
            
            print_embedding_info(user_query, query_vector.tolist())
            
            # ===== STEP 2: Vector Search =====
            print_step(2, "Vector Search in Qdrant", "🔎")
            
            results = agent.retriever.retrieve(
                query=user_query,
                top_k=agent.top_k
            )
            
            print_retrieval_results(results)
            
            if not results:
                print("\n❌ No relevant documents found. Cannot generate answer.")
                continue
            
            # ===== STEP 3: Build Context =====
            print_step(3, "Building Context", "📦")
            
            context_text, sources = agent._build_context_and_sources(results)
            print_context_info(context_text, len(sources))
            
            # ===== STEP 4: LLM Generation =====
            print_step(4, "LLM Generation", "🧠")
            print(f"   🤖 Model: {agent.llm.model}")
            print(f"   🌡️ Temperature: {agent.llm.temperature}")
            print(f"   📝 Max tokens: {agent.llm.max_tokens}")
            print("   ⏳ Generating response...")
            
            # Build prompt
            prompt = agent.config.rag_prompt_template.format(
                context=context_text,
                query=user_query
            )
            
            print(f"   📝 Prompt size: {len(prompt)} characters")
            
            # Generate answer
            answer = agent.llm.generate(prompt)
            
            # ===== STEP 5: Final Output =====
            print_step(5, "Final Response", "💡")
            print_llm_response(answer)
            
            # Show sources summary
            if sources:
                print("\n📚 SOURCES SUMMARY:")
                for src in sources:
                    print(f"   [{src['citation_number']}] {src['source']} (page {src['page']}) - Score: {src['score']:.3f}")
            
            print("\n" + "✅ " + "=" * 68)
            print("Query completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            print(f"\n❌ ERROR: {e}")


def rebuild_if_needed(config: RAGConfig):
    """Offer to rebuild index if collection doesn't exist."""
    from vector_db import QdrantDB
    
    db = QdrantDB(config=config)
    
    if not db.collection_exists():
        print("\n⚠️ No index found in Qdrant!")
        response = input("   Do you want to rebuild the index now? (y/n): ").strip().lower()
        
        if response in ["y", "yes", "بله"]:
            print("\n🔄 Rebuilding index...")
            builder = IndexBuilder(config=config)
            try:
                results = builder.index_directory(config.data_dir, force_recreate=True)
                print("\n✅ Index rebuild complete:")
                for filename, count in results.items():
                    print(f"   • {filename}: {count} chunks")
                total = sum(results.values())
                print(f"   📊 Total: {total} chunks indexed")
            except Exception as e:
                print(f"❌ Failed to rebuild: {e}")
            finally:
                builder.close()
        else:
            print("   Please run: python index_builder.py")
            return False
    
    return True


def main():
    """Main entry point."""
    print_header()
    
    # Load configuration
    print("\n📋 Loading configuration...")
    config = RAGConfig()
    print(f"   Embedder type: {config.embedder_type}")
    print(f"   LLM model: {config.llm_model}")
    print(f"   Top K: {config.default_top_k}")
    print(f"   Temperature: {config.temperature}")
    print(f"   Data directory: {config.data_dir}")
    
    # Check if index exists, offer rebuild
    if not rebuild_if_needed(config):
        return
    
    # Initialize agent
    print("\n🚀 Initializing RAG Agent...")
    agent = RAGAgent(config=config)
    
    # Health check
    if not check_health(agent):
        return
    
    print("\n✅ System ready! Entering interactive mode...")
    
    # Start interactive loop
    interactive_loop(agent)


if __name__ == "__main__":
    main()