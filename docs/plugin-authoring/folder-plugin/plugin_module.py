"""Example folder plugin module."""

from __future__ import annotations


class ExampleEmbedder:
    def __init__(self) -> None:
        self.dim = 3

    def embed_texts(self, texts):
        return [[0.0, 0.0, 0.0] for _ in texts]


def create_embedder(_context, **_kwargs):
    return ExampleEmbedder()
