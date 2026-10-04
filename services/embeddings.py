import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from utils.helpers import clean_text

_EMBEDDER_MODEL = None
_FAISS_INDEX = None
_INDEXED_ARTICLES = []
_EMBEDDINGS_STATUS = "Not Initialized"


def get_embeddings_status() -> Tuple[bool, str]:
    """
    Return status of embedding model and FAISS vector index.
    """
    global _EMBEDDINGS_STATUS, _FAISS_INDEX
    is_ready = _FAISS_INDEX is not None or _EMBEDDER_MODEL is not None
    return is_ready, _EMBEDDINGS_STATUS


def load_embedder_model():
    """
    Load SentenceTransformer model singleton.
    """
    global _EMBEDDER_MODEL, _EMBEDDINGS_STATUS
    if _EMBEDDER_MODEL is not None:
        return _EMBEDDER_MODEL

    try:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        _EMBEDDINGS_STATUS = "✓ SentenceTransformer (all-MiniLM-L6-v2) Loaded"
        return _EMBEDDER_MODEL
    except Exception as e:
        _EMBEDDINGS_STATUS = f"⚠️ SentenceTransformer unavailable: {str(e)[:50]}. Using TF-IDF fallback."
        return None


def generate_embedding(text: str) -> np.ndarray:
    """
    Generate dense embedding vector for string input.
    """
    model = load_embedder_model()
    if model is not None:
        vec = model.encode(text, convert_to_numpy=True)
        # Normalize vector for cosine similarity
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

    # Lightweight TF-IDF style character/word count fallback vector (128-dim)
    words = clean_text(text).lower().split()
    vec = np.zeros(128, dtype=np.float32)
    for idx, w in enumerate(words[:128]):
        vec[hash(w) % 128] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def build_article_vector_index(articles: List[Dict[str, Any]]) -> Tuple[Any, List[np.ndarray]]:
    """
    Build FAISS index over article texts (title + description + summary).
    """
    global _FAISS_INDEX, _INDEXED_ARTICLES
    _INDEXED_ARTICLES = articles

    embeddings = []
    for art in articles:
        text_to_embed = f"{art.get('title', '')} {art.get('description', '')} {art.get('summary', '')}"
        vec = generate_embedding(text_to_embed)
        embeddings.append(vec)

    if not embeddings:
        return None, []

    matrix = np.vstack(embeddings).astype(np.float32)

    try:
        import faiss
        dimension = matrix.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner Product on normalized vectors = Cosine Similarity
        index.add(matrix)
        _FAISS_INDEX = index
        return index, embeddings
    except Exception:
        _FAISS_INDEX = matrix
        return matrix, embeddings


def search_vector_index(query: str, top_k: int = 3) -> List[Tuple[int, float]]:
    """
    Perform semantic vector search using query string over FAISS index.
    Returns list of (article_index, similarity_score).
    """
    global _FAISS_INDEX, _INDEXED_ARTICLES

    if _FAISS_INDEX is None or not _INDEXED_ARTICLES:
        return []

    query_vec = generate_embedding(query).reshape(1, -1)

    try:
        import faiss
        if isinstance(_FAISS_INDEX, faiss.Index):
            scores, indices = _FAISS_INDEX.search(query_vec, min(top_k, len(_INDEXED_ARTICLES)))
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if idx >= 0:
                    results.append((int(idx), float(score)))
            return results
    except Exception:
        pass

    # Numpy fallback matrix multiplication
    if isinstance(_FAISS_INDEX, np.ndarray):
        sims = np.dot(_FAISS_INDEX, query_vec.T).squeeze()
        if sims.ndim == 0:
            sims = np.array([sims])
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(int(idx), float(sims[idx])) for idx in top_indices]

    return []
