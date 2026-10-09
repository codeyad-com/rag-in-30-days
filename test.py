from embeddings import Embedder


from settings import RAGConfig
config = RAGConfig()
emb = Embedder(config)



result = emb.embed_query("What is the capital of France?")


print(result)