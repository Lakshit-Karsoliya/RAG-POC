import os
import hashlib
from typing import List, Dict, Any
import chromadb


from .adapters.ollama_adapter import OllamaAdapter

class TXTIngestor:
    def __init__(self, db_dir: str = "./chroma_db", collection_name: str = "rag_collection"):
        self.adapter = OllamaAdapter("engine/adapters/adapter_config.json")
        self.db_dir = db_dir
        
        # Initialize persistent Chroma client
        self.client = chromadb.PersistentClient(path=self.db_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def _chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start += chunk_size - chunk_overlap
            if chunk_overlap >= chunk_size:
                break
                
        return chunks

    def _compute_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def ingest_file(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100):
        if not file_path.endswith(".txt"):
            print(f"[SKIP] {file_path} - Only .txt files are supported.")
            return

        if not os.path.exists(file_path):
            print(f"[ERROR] File not found: {file_path}")
            return

        print(f"\nProcessing file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read()

        chunks = self._chunk_text(full_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print(f"Total Chunks Created: {len(chunks)}")

        existing_records = self.collection.get()
        existing_ids = set(existing_records["ids"]) if existing_records and "ids" in existing_records else set()

        documents_to_add = []
        embeddings_to_add = []
        metadatas_to_add = []
        ids_to_add = []
        
        skipped_count = 0

        for idx, chunk in enumerate(chunks):
            chunk_hash = self._compute_hash(chunk)
            
            if chunk_hash in existing_ids:
                skipped_count += 1
                continue
            embedding = self.adapter.embeddings(chunk)

            if not embedding:
                print(f"[WARNING] Skipping chunk {idx} - Failed to obtain vector embedding.")
                continue

            ids_to_add.append(chunk_hash)
            documents_to_add.append(chunk)
            embeddings_to_add.append(embedding)
            metadatas_to_add.append({
                "source": os.path.basename(file_path),
                "chunk_index": idx,
                "chunk_length": len(chunk)
            })

        if ids_to_add:
            self.collection.add(
                ids=ids_to_add,
                documents=documents_to_add,
                embeddings=embeddings_to_add,
                metadatas=metadatas_to_add
            )
            print(f"[SUCCESS] Ingested {len(ids_to_add)} new unique chunk(s).")
        
        if skipped_count > 0:
            print(f"[DEDUPLICATION] Skipped {skipped_count} existing chunk(s).")


if __name__ == "__main__":
    ingestor = TXTIngestor()
    
    sample_file = "sample_data.txt"
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write(
            "Retrieval-Augmented Generation (RAG) is an architectural technique that enhances the "
            "capabilities of Large Language Models (LLMs) by augmenting them with external knowledge sources. "
            "Instead of relying purely on parametric memory learned during training, RAG retrieves relevant "
            "document snippets from a vector database like ChromaDB using vector embeddings.\n\n"
            "This approach significantly reduces hallucinations and allows LLMs to cite up-to-date sources."
        )

   
    ingestor.ingest_file(sample_file, chunk_size=1000, chunk_overlap=100)
    print("\n--- Testing Deduplication (Re-running same file) ---")
    ingestor.ingest_file(sample_file, chunk_size=1000, chunk_overlap=100)