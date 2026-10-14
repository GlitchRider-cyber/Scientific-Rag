from embedding.dual_store import DualStore
from retrieval.reranker import rerank
from config import TOP_K


def retrieve(query: str, store: DualStore, top_k: int = TOP_K) -> list[dict]:
    """
    Retrieve top-k chunks:
    1. Get top-k from CLIP index
    2. Get top-k from MiniLM index
    3. Rerank all using RRF + cosine sim
    4. Return top-k reranked
    """
    search_results = store.search(query, top_k=top_k)
    reranked = rerank(search_results, store)
    return reranked[:top_k]
