from __future__ import annotations

from dataclasses import dataclass

import requests


@dataclass
class OpenAICompatibleEmbeddingConfig:
    base_url: str
    api_key: str
    model: str
    timeout: int = 120


class OpenAICompatibleEmbeddingProvider:
    """OpenAI-compatible embeddings provider for memory retrieval."""

    def __init__(self, config: OpenAICompatibleEmbeddingConfig) -> None:
        self.config = config

    def embed_text(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Embedding input text must not be empty.")

        url = self.config.base_url.rstrip("/") + "/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload = {
            "model": self.config.model,
            "input": text,
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.config.timeout,
        )
        if not response.ok:
            snippet = response.text[:500]
            raise RuntimeError(
                f"OpenAI-compatible embedding request failed with status {response.status_code}: {snippet}"
            )

        data = response.json()
        try:
            embedding = data["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Malformed OpenAI-compatible embedding response: {str(data)[:500]}"
            ) from exc
        if not isinstance(embedding, list) or not all(isinstance(value, (int, float)) for value in embedding):
            raise RuntimeError(
                f"Malformed embedding vector in response: {str(data)[:500]}"
            )
        return [float(value) for value in embedding]
