import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"
INDICES_DIR = DATA_DIR / "indices"

# Create dirs
PAPERS_DIR.mkdir(parents=True, exist_ok=True)
INDICES_DIR.mkdir(parents=True, exist_ok=True)

# API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.6-flash"

# Chunking
MAX_CHUNK_SIZE = 500  # words
CHUNK_OVERLAP = 50    # words

# Retrieval
TOP_K = 5            # chunks per index
RRF_K = 60           # RRF constant
RERANK_ALPHA = 0.6   # weight for RRF vs cosine sim
