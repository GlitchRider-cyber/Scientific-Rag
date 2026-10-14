from config import RRF_K, RERANK_ALPHA


def rerank(search_results: dict, store) -> list[dict]:
    """
    Rerank chunks using Reciprocal Rank Fusion + cosine similarity.

    Formula:
        RRF_score(chunk) = sum(1 / (k + rank)) across all sources
        final_score = alpha * RRF_score + (1 - alpha) * avg_cosine_sim
    """
    chunk_scores = {}  # chunk_id -> {"rrf": float, "cosine_scores": [float]}

    for source_name in ["clip", "minilm"]:
        ids = search_results[source_name]["ids"]
        scores = search_results[source_name]["scores"]

        for rank, (chunk_id, cosine_score) in enumerate(zip(ids, scores)):
            if chunk_id == -1:  # FAISS returns -1 for empty slots
                continue

            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {"rrf": 0.0, "cosine_scores": []}

            # RRF: 1 / (k + rank), rank is 0-indexed
            chunk_scores[chunk_id]["rrf"] += 1.0 / (RRF_K + rank + 1)
            chunk_scores[chunk_id]["cosine_scores"].append(cosine_score)

    # Compute final scores
    ranked = []
    for chunk_id, data in chunk_scores.items():
        avg_cosine = sum(data["cosine_scores"]) / len(data["cosine_scores"])
        final_score = RERANK_ALPHA * data["rrf"] + (1 - RERANK_ALPHA) * avg_cosine
        chunk = store.get_chunk(chunk_id)
        chunk["score"] = round(final_score, 4)
        chunk["chunk_index"] = chunk_id
        ranked.append(chunk)

    # Sort by final score descending
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked
