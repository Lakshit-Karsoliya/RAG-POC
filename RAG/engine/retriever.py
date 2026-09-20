from typing import List, Dict, Any, Optional
import chromadb

# Import your custom OllamaAdapter
from .adapters.ollama_adapter import OllamaAdapter

class Retriever:
    """Retriever class to query ChromaDB using vector embeddings from OllamaAdapter."""

    def __init__(
        self, 
        config_path: str = "adapters/adapter_config.json", 
        db_dir: str = "./chroma_db", 
        collection_name: str = "rag_collection"
    ):
        """Initializes the vector store client and Ollama adapter."""
        self.adapter = OllamaAdapter(config_path)
        self.db_dir = db_dir
        self.collection_name = collection_name
        
        # Connect to existing persistent ChromaDB
        self.client = chromadb.PersistentClient(path=self.db_dir)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Generates an embedding for the user query and retrieves top_k documents from ChromaDB.

        :param query: Natural language user query.
        :param top_k: Number of most relevant context chunks to return.
        :return: A list of dictionaries containing document text, metadata, and distance metrics.
        """
        if not query.strip():
            print("[WARNING] Empty query provided.")
            return []

        # Step 1: Embed the input query using OllamaAdapter
        query_embedding = self.adapter.embeddings(query)
        if not query_embedding:
            raise RuntimeError(f"Failed to generate embedding for query: '{query}'")

        # Step 2: Query ChromaDB collection using vector similarity search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # Step 3: Format query output into structured dictionaries
        formatted_results = []
        
        # ChromaDB returns nested arrays for documents, metadatas, and distances
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        for doc_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            formatted_results.append({
                "id": doc_id,
                "content": doc,
                "metadata": meta,
                "distance": dist  # Lower distance indicates higher vector similarity
            })

        return formatted_results

    def get_context_str(self, query: str, top_k: int = 3) -> str:
        """
        Helper method to retrieve and format relevant context as a single text block
        ready to feed into an LLM prompt.
        """
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return ""

        context_blocks = []
        for idx, res in enumerate(results, start=1):
            source_file = res["metadata"].get("source", "Unknown Source")
            context_blocks.append(f"--- Document Chunk {idx} (Source: {source_file}) ---\n{res['content']}")

        return "\n\n".join(context_blocks)


# Verification / Standalone Test
if __name__ == "__main__":
    retriever = Retriever()
    
    test_query = "What is Retrieval-Augmented Generation?"
    print(f"Query: '{test_query}'\n")

    try:
        # Retrieve structured items
        docs = retriever.retrieve(test_query, top_k=2)
        print(f"Retrieved {len(docs)} matching document(s):\n")
        
        for idx, doc in enumerate(docs, start=1):
            print(f"Result #{idx}:")
            print(f"  - Content: {doc['content']}")
            print(f"  - Source: {doc['metadata'].get('source')}")
            print(f"  - Distance: {doc['distance']:.4f}\n")

        print("--- Formatted Context String for LLM Prompt ---")
        context_str = retriever.get_context_str(test_query, top_k=2)
        print(context_str)

    except Exception as err:
        print("Retrieval Error:", err)