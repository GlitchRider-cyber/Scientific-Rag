from google import genai
from config import GEMINI_API_KEY, GEMINI_MODEL

client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """You are a scientific research assistant. Answer the user's question based ONLY on the provided context passages.

Rules:
1. Use ONLY information from the context. Do not use outside knowledge.
2. Cite every claim using [Source: filename, Chunk N] format.
3. If the context doesn't contain enough information, say so clearly.
4. Identify methodologies, datasets, findings, and limitations when relevant.
5. Be precise and academic in tone."""

QUERY_TEMPLATE = """Context passages:
{context}

Question: {question}

Provide a detailed, citation-backed answer."""


def generate_answer(question: str, chunks: list[dict]) -> dict:
    """Generate an answer using Gemini with retrieved chunks as context."""
    # Format context
    context_parts = []
    for i, chunk in enumerate(chunks):
        source = chunk.get("source", "unknown")
        context_parts.append(f"[Chunk {i+1} | Source: {source}]\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)
    prompt = QUERY_TEMPLATE.format(context=context, question=question)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
        ),
    )

    return {
        "answer": response.text,
        "sources": [
            {
                "chunk_index": c.get("chunk_index", i),
                "source": c.get("source", "unknown"),
                "text": c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
                "score": c.get("score", 0),
            }
            for i, c in enumerate(chunks)
        ],
    }
