from __future__ import annotations

import hashlib
import math
from typing import Iterable

import chromadb
from chromadb.api.models.Collection import Collection

from backend.config import settings


class DeterministicEmbedding:
    """Local deterministic embedding to avoid external dependency at runtime."""

    dim: int = 128

    def _embed_text(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = [t.lower() for t in text.split() if t.strip()]
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.sha256(token.encode('utf-8')).digest()
            for i in range(self.dim):
                value = digest[i % len(digest)] / 255.0
                vec[i] += value
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]

    def __call__(self, input: Iterable[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in input]

    def name(self) -> str:
        # Chroma validates embedding-function identity via name().
        return 'deterministic_hash_v1'


class ChromaStore:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=settings.chroma_path)
        self.embedding_fn = DeterministicEmbedding()
        self.collection: Collection = self.client.get_or_create_collection(
            name='narrative_knowledge',
            embedding_function=self.embedding_fn,
            metadata={'hnsw:space': 'cosine'},
        )

    def reset(self) -> None:
        try:
            self.client.delete_collection('narrative_knowledge')
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name='narrative_knowledge',
            embedding_function=self.embedding_fn,
            metadata={'hnsw:space': 'cosine'},
        )

    def add_documents(self, docs: list[dict]) -> None:
        if not docs:
            return
        ids = [f"{doc['type']}::{doc['name']}::{idx}" for idx, doc in enumerate(docs)]
        documents = [doc['description'] for doc in docs]
        metadatas = [doc['metadata'] | {'type': doc['type'], 'name': doc['name']} for doc in docs]
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def search(self, query: str, k: int = 5) -> list[dict]:
        result = self.collection.query(query_texts=[query], n_results=k)
        docs = []
        for doc, metadata, dist in zip(
            result.get('documents', [[]])[0],
            result.get('metadatas', [[]])[0],
            result.get('distances', [[]])[0],
            strict=False,
        ):
            docs.append({'content': doc, 'metadata': metadata, 'distance': dist})
        return docs


chroma_store = ChromaStore()
