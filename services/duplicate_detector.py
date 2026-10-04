import numpy as np
from typing import List, Dict, Any
from services.embeddings import generate_embedding


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two normalized vectors.
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot_product / (norm1 * norm2))


def detect_duplicate_stories(
    articles: List[Dict[str, Any]],
    similarity_threshold: float = 0.75
) -> List[Dict[str, Any]]:
    """
    Group news articles into semantic coverage clusters based on vector embedding similarity.

    Returns list of clusters:
    [
        {
            "primary_article": dict,
            "related_articles": list of dicts,
            "total_coverage": int,
            "avg_similarity": float
        }
    ]
    """
    if not articles:
        return []

    # Compute embeddings for all articles
    embeddings = []
    for art in articles:
        text = f"{art.get('title', '')} {art.get('description', '')} {art.get('summary', '')}"
        vec = generate_embedding(text)
        embeddings.append(vec)

    visited = set()
    clusters = []

    for i in range(len(articles)):
        if i in visited:
            continue

        visited.add(i)
        primary_art = articles[i]
        primary_vec = embeddings[i]

        related_articles = []
        similarities = []

        for j in range(i + 1, len(articles)):
            if j in visited:
                continue

            sim = compute_cosine_similarity(primary_vec, embeddings[j])
            if sim >= similarity_threshold:
                visited.add(j)
                art_copy = dict(articles[j])
                art_copy["similarity_score"] = round(sim, 3)
                related_articles.append(art_copy)
                similarities.append(sim)

        avg_sim = float(np.mean(similarities)) if similarities else 1.0

        clusters.append({
            "primary_article": primary_art,
            "related_articles": related_articles,
            "total_coverage": 1 + len(related_articles),
            "avg_similarity": round(avg_sim, 3),
        })

    return clusters
