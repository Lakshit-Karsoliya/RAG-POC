from typing import Dict, Any, Optional
from .adapters.ollama_adapter import OllamaAdapter
from .retriever import Retriever

class RAGQueryPipeline:
    """Orchestrates the Retrieval-Augmented Generation (RAG) pipeline."""

    def __init__(self, config_path: str = "adapters/adapter_config.json", db_dir: str = "./chroma_db"):
        """Initializes the retriever and LLM adapter."""
        self.retriever = Retriever(config_path=config_path, db_dir=db_dir)
        self.adapter = OllamaAdapter(config_path=config_path)

    def _build_prompt(self, query: str, context: str) -> str:
        """Constructs a strict context-grounded prompt for the LLM."""
        if not context.strip():
            context = "No relevant context found in the database."

        prompt = f"""You are a helpful AI assistant. Answer the user's question using ONLY the provided context below. 
If the context does not contain enough information to answer the question, state that you do not have enough information.

--- CONTEXT START ---
{context}
--- CONTEXT END ---

User Question: {query}

Answer:"""
        return prompt

    def run(self, query: str, top_k: int = 3, stream: bool = False) -> Dict[str, Any]:
        """
        Executes the RAG pipeline end-to-end:
        1. Retrieves top_k context chunks from ChromaDB.
        2. Constructs a context-infused prompt.
        3. Calls OllamaAdapter to generate the response.

        :param query: Natural language user question.
        :param top_k: Number of relevant document chunks to retrieve.
        :param stream: Whether to stream the output or wait for full completion.
        :return: Dict containing answer, retrieved context chunks, and prompt metadata.
        """
        if not query.strip():
            return {
                "answer": "Please provide a valid non-empty query.",
                "retrieved_docs": [],
                "prompt": ""
            }

        # Step 1: Retrieve matching documents
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k)
        
        # Format documents into a unified context block
        context_str = self.retriever.get_context_str(query, top_k=top_k)

        # Step 2: Build the RAG prompt
        prompt = self._build_prompt(query=query, context=context_str)

        # Step 3: Invoke the LLM model via OllamaAdapter
        llm_response = self.adapter.generate(prompt=prompt, stream=stream)
        answer_text = llm_response.get("response", "").strip()

        return {
            "answer": answer_text,
            "retrieved_docs": retrieved_docs,
            "prompt": prompt
        }


# Standalone Verification / Execution
if __name__ == "__main__":
    rag_pipeline = RAGQueryPipeline()

    user_query = "What is Retrieval-Augmented Generation?"
    print(f"--- Running RAG Query: '{user_query}' ---")

    result = rag_pipeline.run(query=user_query, top_k=2)

    print("\n[Retrieved Context Docs]:")
    for idx, doc in enumerate(result["retrieved_docs"], start=1):
        print(f"  Doc #{idx} (Source: {doc['metadata'].get('source')}):")
        print(f"  Content: {doc['content'][:120]}...\n")

    print("[Generated LLM Answer]:")
    print(result["answer"])