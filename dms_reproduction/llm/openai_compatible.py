from __future__ import annotations

from typing import Any, Dict, List

import requests

from dms_reproduction.llm.base_client import OpenAICompatibleConfig


class OpenAICompatibleClient:
    """Call an OpenAI-compatible chat completion endpoint via HTTP."""

    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self.config = config

    def generate(self, messages: List[Dict[str, Any]], temperature: float = 0.0) -> str:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.config.max_tokens,
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
                f"OpenAI-compatible request failed with status {response.status_code}: {snippet}"
            )

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Malformed OpenAI-compatible response: {str(data)[:500]}"
            ) from exc
