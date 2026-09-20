import json
import requests
from typing import Dict, Any, Optional, Generator, List

class OllamaAdapter:

    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.base_url, self.default_model, self.default_embed_model = self._load_config()

    def _load_config(self) -> tuple[str, str, str]:
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                
            ollama_config = config.get("adapters", {}).get("ollama", {})
            url = ollama_config.get("url", "http://localhost:11434")
            model = ollama_config.get("model", "llama3")
            embed_model = ollama_config.get("embedding_model", "nomic-embed-text")
            
            # Remove trailing slashes for clean URL concatenation
            return url.rstrip("/"), model, embed_model

        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in configuration file: {self.config_path}")

    def generate(self, prompt: str, model: Optional[str] = None, stream: bool = False, **options) -> Dict[str, Any]:
        endpoint = f"{self.base_url}/api/generate"
        payload = {
            "model": model or self.default_model,
            "prompt": prompt,
            "stream": stream,
            "options": options
        }

        try:
            response = requests.post(endpoint, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to communicate with Ollama server at {endpoint}: {e}")

    def generate_stream(self, prompt: str, model: Optional[str] = None, **options) -> Generator[str, None, None]:
        """
        Streams response chunks from Ollama line-by-line.
        """
        endpoint = f"{self.base_url}/api/generate"
        payload = {
            "model": model or self.default_model,
            "prompt": prompt,
            "stream": True,
            "options": options
        }

        try:
            with requests.post(endpoint, json=payload, stream=True, timeout=60) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line.decode("utf-8"))
                        yield chunk.get("response", "")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Streaming failed with Ollama at {endpoint}: {e}")

    def embeddings(self, prompt: str, model: Optional[str] = None) -> List[float]:
        """
        Generates vector embeddings for input text via Ollama /api/embed endpoint.
        """
        endpoint = f"{self.base_url}/api/embed"
        selected_model = model or self.default_embed_model
        payload = {
            "model": selected_model,
            "input": prompt
        }

        try:
            response = requests.post(endpoint, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # /api/embed returns a list of embeddings arrays
            embeddings_list = data.get("embeddings", [])
            return embeddings_list[0] if embeddings_list else []
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to generate embeddings from Ollama at {endpoint}: {e}")



if __name__ == "__main__":
    adapter = OllamaAdapter("adapter_config.json")
    print(f"Connected to Ollama at: {adapter.base_url}")
    print(f"Generation Model: {adapter.default_model}")
    print(f"Embedding Model: {adapter.default_embed_model}")
    
    print("\n--- Testing Response ---")
    try:
        result = adapter.generate("Say hello in one word!")
        print("Response:", result.get("response", "").strip())
    except Exception as err:
        print("Error:", err)

    print("\n--- Testing Embeddings ---")
    try:
        embeds = adapter.embeddings("Retrieval Augmented Generation")
        print(f"Vector Dimensions: {len(embeds)}")
    except Exception as err:
        print("Error:", err)