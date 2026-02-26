from __future__ import annotations

import hashlib
import math
import os
import shutil
from typing import Iterable

import chromadb
from chromadb.config import Settings


class DeterministicEmbedding:
    dim = 128

    def _embed_text(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = [t.lower() for t in text.split() if t.strip()]
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.sha256(token.encode('utf-8')).digest()
            for i in range(self.dim):
                vec[i] += digest[i % len(digest)] / 255.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec

    def __call__(self, input: Iterable[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in input]

    def name(self) -> str:
        return 'deterministic_hash_v1'

    # Chroma v0.5+ embedding API compatibility
    def embed_query(self, input: str | list[str]) -> list[float] | list[list[float]]:
        if isinstance(input, list):
            return [self._embed_text(text) for text in input]
        return self._embed_text(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in input]


class ChromaStore:
    def __init__(self, path: str = '.chroma') -> None:
        self.embedding_fn = DeterministicEmbedding()
        self.client = self._init_client(path)
        self.collection = self.client.get_or_create_collection(
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

    def _init_client(self, path: str) -> chromadb.PersistentClient:
        try:
            return chromadb.PersistentClient(path=path)
        except BaseException:
            # If the persistent store is corrupted, recreate it.
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            try:
                return chromadb.PersistentClient(path=path)
            except BaseException:
                # Fallback to explicit Settings-based client creation.
                settings = Settings(is_persistent=True, persist_directory=path, allow_reset=True)
                client = chromadb.Client(settings=settings)
                try:
                    client.reset()
                except Exception:
                    pass
                return client

    def add_from_scenes(self, scenes: list[dict]) -> None:
        docs: list[dict] = []
        for scene in scenes:
            for plot in scene.get('plots', []):
                for npc in plot.get('npc', []):
                    docs.append({'type': 'npc', 'name': npc, 'description': f'NPC {npc} in {scene["scene_id"]}/{plot["plot_id"]}', 'metadata': {'scene_id': scene['scene_id'], 'plot_id': plot['plot_id']}})
                for location in plot.get('locations', []):
                    docs.append({'type': 'location', 'name': location, 'description': f'Location {location} in {scene["scene_id"]}/{plot["plot_id"]}', 'metadata': {'scene_id': scene['scene_id'], 'plot_id': plot['plot_id']}})
                for event in plot.get('mandatory_events', []):
                    docs.append({'type': 'event', 'name': event[:80], 'description': event, 'metadata': {'scene_id': scene['scene_id'], 'plot_id': plot['plot_id']}})
        if not docs:
            return
        ids = [f"{d['type']}::{d['name']}::{i}" for i, d in enumerate(docs)]
        self.collection.add(
            ids=ids,
            documents=[d['description'] for d in docs],
            metadatas=[d['metadata'] | {'type': d['type'], 'name': d['name']} for d in docs],
        )

    def search(self, query: str, k: int = 5) -> list[dict]:
        result = self.collection.query(query_texts=[query], n_results=k)
        out = []
        for doc, meta, dist in zip(
            result.get('documents', [[]])[0],
            result.get('metadatas', [[]])[0],
            result.get('distances', [[]])[0],
            strict=False,
        ):
            out.append({'content': doc, 'metadata': meta, 'distance': dist})
        return out
