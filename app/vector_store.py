from __future__ import annotations

import hashlib
import math
import os
import shutil
from typing import Iterable, List

import chromadb
from chromadb.config import Settings
import re

from sentence_transformers import SentenceTransformer


class ModelEmbedding:
    _model = None

    def __init__(self):
        if ModelEmbedding._model is None:
            ModelEmbedding._model = SentenceTransformer("intfloat/multilingual-e5-base")
        self.model = ModelEmbedding._model

    # ----------------------------
    # Preprocess for E5 format
    # ----------------------------
    def _prepare_text(self, text: str) -> str:
        lang = self._detect_language(text)

        # E5 family models expect special prefix
        # query: for search query
        # passage: for documents

        # Here we default to passage format
        # embed_query() will override to query
        return text.strip()

    # ----------------------------
    # Core embedding
    # ----------------------------
    def _embed(self, texts: List[str], prefix: str) -> List[List[float]]:
        processed = [f"{prefix}: {t.strip()}" for t in texts]
        embeddings = self.model.encode(
            processed,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    # ----------------------------
    # Chroma compatible API
    # ----------------------------
    def __call__(self, input: Iterable[str]) -> List[List[float]]:
        return self.embed_documents(list(input))

    def name(self) -> str:
        return "multilingual_e5_base"

    def embed_query(self, input: str | List[str]):
        if isinstance(input, list):
            return self._embed(input, prefix="query")
        return self._embed([input], prefix="query")[0]

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        return self._embed(input, prefix="passage")


class ChromaStore:
    def __init__(self, path: str = '.chroma') -> None:
        self.embedding_fn = ModelEmbedding()
        self.client = self._init_client(path)
        self.collection = self.client.get_or_create_collection(
            name='narrative_knowledge',
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
        embeddings = self.embedding_fn.embed_documents(
            [d['description'] for d in docs]
        )
        self.collection.add(
            ids=ids,
            documents=[d['description'] for d in docs],
            metadatas=[d['metadata'] | {'type': d['type'], 'name': d['name']} for d in docs],
            embeddings=embeddings
        )

    def search(self, query: str, k: int = 5) -> list[dict]:
        query_embedding = self.embedding_fn.embed_query(query)

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        out = []
        for doc, meta, dist in zip(
            result.get('documents', [[]])[0],
            result.get('metadatas', [[]])[0],
            result.get('distances', [[]])[0],
            strict=False,
        ):
            out.append({'content': doc, 'metadata': meta, 'distance': dist})
        return out
