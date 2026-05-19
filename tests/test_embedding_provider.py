from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from dms_reproduction.memory.embedding import (
    OpenAICompatibleEmbeddingConfig,
    OpenAICompatibleEmbeddingProvider,
)


class OpenAICompatibleEmbeddingProviderTest(unittest.TestCase):
    def test_embed_text_calls_openai_compatible_embeddings_endpoint(self) -> None:
        response = MagicMock()
        response.ok = True
        response.json.return_value = {"data": [{"embedding": [0.25, 0.75]}]}

        provider = OpenAICompatibleEmbeddingProvider(
            OpenAICompatibleEmbeddingConfig(
                base_url="http://127.0.0.1:8000/v1",
                api_key="secret",
                model="bge-small-en-v1.5",
                timeout=30,
            )
        )

        with patch("dms_reproduction.memory.embedding.requests.post", return_value=response) as post:
            vector = provider.embed_text("hello world")

        self.assertEqual(vector, [0.25, 0.75])
        post.assert_called_once()
        self.assertEqual(post.call_args.kwargs["json"]["model"], "bge-small-en-v1.5")
        self.assertEqual(post.call_args.kwargs["json"]["input"], "hello world")
        self.assertEqual(post.call_args.args[0], "http://127.0.0.1:8000/v1/embeddings")

    def test_empty_text_raises_value_error(self) -> None:
        provider = OpenAICompatibleEmbeddingProvider(
            OpenAICompatibleEmbeddingConfig(
                base_url="http://127.0.0.1:8000/v1",
                api_key="secret",
                model="bge-small-en-v1.5",
            )
        )

        with self.assertRaises(ValueError):
            provider.embed_text("   ")

    def test_http_error_raises_readable_runtime_error(self) -> None:
        response = MagicMock()
        response.ok = False
        response.status_code = 500
        response.text = "internal error"
        provider = OpenAICompatibleEmbeddingProvider(
            OpenAICompatibleEmbeddingConfig(
                base_url="http://127.0.0.1:8000/v1",
                api_key="secret",
                model="bge-small-en-v1.5",
            )
        )

        with patch("dms_reproduction.memory.embedding.requests.post", return_value=response):
            with self.assertRaises(RuntimeError) as ctx:
                provider.embed_text("hello")

        self.assertIn("embedding request failed", str(ctx.exception))

    def test_malformed_response_raises_readable_runtime_error(self) -> None:
        response = MagicMock()
        response.ok = True
        response.json.return_value = {"unexpected": []}
        provider = OpenAICompatibleEmbeddingProvider(
            OpenAICompatibleEmbeddingConfig(
                base_url="http://127.0.0.1:8000/v1",
                api_key="secret",
                model="bge-small-en-v1.5",
            )
        )

        with patch("dms_reproduction.memory.embedding.requests.post", return_value=response):
            with self.assertRaises(RuntimeError) as ctx:
                provider.embed_text("hello")

        self.assertIn("Malformed OpenAI-compatible embedding response", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
