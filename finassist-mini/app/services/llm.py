"""Service for communicating with Ollama's generate API."""

import requests


class LLMService:
    """Calls Ollama using a configured model and timeout."""

    def __init__(self, base_url: str, model_name: str, timeout_seconds: int = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds

    def generate_answer(self, prompt: str) -> str:
        """Generate an answer using Ollama and validate the response payload."""
        endpoint = f"{self.base_url}/api/generate"
        payload = {"model": self.model_name, "prompt": prompt, "stream": False}

        try:
            response = requests.post(endpoint, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
        except requests.Timeout as exc:
            raise RuntimeError("Ollama request timed out.") from exc
        except requests.RequestException as exc:
            raise RuntimeError("Failed to reach Ollama. Ensure Ollama is running and model is pulled.") from exc

        data = response.json()
        answer = data.get("response")
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("Ollama returned an invalid response payload.")
        return answer.strip()
