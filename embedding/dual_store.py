import faiss
import numpy as np
import json
import os
from config import INDICES_DIR
from embedding import clip_embedder, minilm_embedder


class DualStore:
    """Manages two FAISS indices (CLIP + MiniLM) and chunk metadata."""

    def __init__(self):
        self.clip_index = None
        self.minilm_index = None
        self.chunks = []  # list of chunk dicts
        self._load_or_init()

    def _load_or_init(self):
        """Load existing indices or create empty ones."""
        clip_path = os.path.join(INDICES_DIR, "clip.index")
        minilm_path = os.path.join(INDICES_DIR, "minilm.index")
        meta_path = os.path.join(INDICES_DIR, "chunks.json")

        if all(os.path.exists(p) for p in [clip_path, minilm_path, meta_path]):
            self.clip_index = faiss.read_index(clip_path)
            self.minilm_index = faiss.read_index(minilm_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
        else:
            clip_dim = clip_embedder.get_dimension()
            minilm_dim = minilm_embedder.get_dimension()
            self.clip_index = faiss.IndexFlatIP(clip_dim)    # inner product (cosine after norm)
            self.minilm_index = faiss.IndexFlatIP(minilm_dim)
            self.chunks = []

    def add_chunks(self, chunks: list[dict]):
        """Embed and add chunks to both indices."""
        if not chunks:
            return

        texts = [c["text"] for c in chunks]

        # Embed with both models
        clip_embs = clip_embedder.embed_texts(texts)
        minilm_embs = minilm_embedder.embed_texts(texts)

        # Normalize for cosine similarity (IndexFlatIP)
        faiss.normalize_L2(clip_embs)
        faiss.normalize_L2(minilm_embs)

        # Add to indices
        self.clip_index.add(clip_embs)
        self.minilm_index.add(minilm_embs)
        self.chunks.extend(chunks)

        self._save()

    def search(self, query: str, top_k: int = 5) -> dict:
        """Search both indices and return results with scores."""
        # Embed query with both models
        clip_q = clip_embedder.embed_query(query).reshape(1, -1)
        minilm_q = minilm_embedder.embed_query(query).reshape(1, -1)

        faiss.normalize_L2(clip_q)
        faiss.normalize_L2(minilm_q)

        # Search both
        clip_scores, clip_ids = self.clip_index.search(clip_q, top_k)
        minilm_scores, minilm_ids = self.minilm_index.search(minilm_q, top_k)

        return {
            "clip": {
                "ids": clip_ids[0].tolist(),
                "scores": clip_scores[0].tolist(),
            },
            "minilm": {
                "ids": minilm_ids[0].tolist(),
                "scores": minilm_scores[0].tolist(),
            },
        }

    def get_chunk(self, idx: int) -> dict:
        """Get chunk by index."""
        if 0 <= idx < len(self.chunks):
            return self.chunks[idx]
        return {}

    def total_chunks(self) -> int:
        return len(self.chunks)

    def _save(self):
        """Persist indices and metadata to disk."""
        faiss.write_index(self.clip_index, os.path.join(INDICES_DIR, "clip.index"))
        faiss.write_index(self.minilm_index, os.path.join(INDICES_DIR, "minilm.index"))
        with open(os.path.join(INDICES_DIR, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2)

    def clear(self):
        """Reset everything."""
        clip_dim = clip_embedder.get_dimension()
        minilm_dim = minilm_embedder.get_dimension()
        self.clip_index = faiss.IndexFlatIP(clip_dim)
        self.minilm_index = faiss.IndexFlatIP(minilm_dim)
        self.chunks = []
        self._save()
