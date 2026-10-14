from sentence_transformers import SentenceTransformer
import numpy as np

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("clip-ViT-B-32")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts using CLIP."""
    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return np.array(embeddings, dtype="float32")


def embed_query(query: str) -> np.ndarray:
    """Embed a single query using CLIP."""
    return embed_texts([query])[0]


def get_dimension() -> int:
    """Return embedding dimension."""
    return _get_model().get_sentence_embedding_dimension()
