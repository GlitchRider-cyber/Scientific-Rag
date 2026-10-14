from google import genai
from config import GEMINI_API_KEY, MAX_CHUNK_SIZE

client = genai.Client(api_key=GEMINI_API_KEY)

CHUNKING_PROMPT = """Split the following academic text into semantically meaningful chunks.
Each chunk should be a self-contained passage about one topic/idea.
Target around {max_words} words per chunk.
Return ONLY the chunks separated by the delimiter: ===CHUNK===
Do not add any numbering, labels, or commentary.

Text:
{text}"""


def semantic_chunk(text: str, source: str = "") -> list[dict]:
    """Use Gemini to split text into semantic chunks."""
    if not text.strip():
        return []

    prompt = CHUNKING_PROMPT.format(max_words=MAX_CHUNK_SIZE, text=text)

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    raw = response.text
    chunks = [c.strip() for c in raw.split("===CHUNK===") if c.strip()]

    return [
        {"text": chunk, "source": source, "chunk_id": i}
        for i, chunk in enumerate(chunks)
    ]
