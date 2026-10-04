import os
import requests
from typing import List, Dict, Any, Tuple, Optional
from services.embeddings import search_vector_index
from services.summarizer import summarize_with_fine_tuned_model
from utils.helpers import get_env_var, clean_text


def answer_with_groq(question: str, context: str) -> Tuple[Optional[str], str]:
    """
    Generate grounded QA response using Groq LLM API.
    """
    api_key = get_env_var("GROQ_API_KEY")
    model_name = get_env_var("GROQ_MODEL", "llama-3.1-8b-instant")

    if not api_key:
        return None, "Groq API key missing in environment variables."

    system_prompt = (
        "You are an AI News Intelligence assistant providing accurate, factual answers. "
        "Answer the user's question STRICTLY using the provided news context. "
        "Rules:\n"
        "1. Base your response ONLY on the provided news context.\n"
        "2. If the context does not contain enough facts to answer the question, state: "
        "'Based on the retrieved news articles, there is insufficient information to answer this question.'\n"
        "3. Do not invent or assume external facts.\n"
        "4. Keep the answer clear, professional, and well-structured."
    )

    user_prompt = f"RETRIEVED NEWS CONTEXT:\n{context}\n\nUSER QUESTION: {question}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 400,
    }

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"].strip()
        return answer, "Success"
    except Exception as e:
        return None, f"Groq QA Error: {str(e)}"


def ask_about_news(
    question: str,
    articles: List[Dict[str, Any]],
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Full RAG Pipeline:
    1. Embed User Question
    2. Vector Search (FAISS) over article index
    3. Construct Grounded Context
    4. Pass Context + Question to LLM (Fine-tuned or Groq Fallback)
    5. Return Grounded Answer with Source Citations
    """
    if not question or not question.strip():
        return {
            "answer": "Please enter a valid question to search retrieved news.",
            "sources": [],
            "retrieved_articles": [],
            "model_used": "None",
        }

    if not articles:
        return {
            "answer": "No news articles are currently indexed to answer your question.",
            "sources": [],
            "retrieved_articles": [],
            "model_used": "None",
        }

    # Vector Search over indexed articles
    search_results = search_vector_index(question, top_k=top_k)

    retrieved_articles = []
    sources = []

    if search_results:
        for idx, score in search_results:
            if idx < len(articles):
                art = articles[idx]
                retrieved_articles.append(art)
                sources.append({
                    "title": art.get("title", "Untitled"),
                    "source": art.get("source", "Unknown"),
                    "url": art.get("url", "#"),
                    "relevance_score": round(score, 3),
                })
    else:
        # Fallback to top k articles if index empty
        retrieved_articles = articles[:top_k]
        sources = [
            {
                "title": a.get("title", "Untitled"),
                "source": a.get("source", "Unknown"),
                "url": a.get("url", "#"),
                "relevance_score": 1.0,
            }
            for a in retrieved_articles
        ]

    # Build context string
    context_blocks = []
    for i, art in enumerate(retrieved_articles, 1):
        block = (
            f"Article [{i}]: {art.get('title', '')}\n"
            f"Source: {art.get('source', '')}\n"
            f"Summary/Content: {art.get('summary') or art.get('description') or art.get('content', '')}"
        )
        context_blocks.append(block)

    full_context = "\n\n".join(context_blocks)

    # Attempt LLM Response Generation (Fine-Tuned or Groq Fallback)
    groq_answer, err_msg = answer_with_groq(question, full_context)
    if groq_answer:
        return {
            "answer": groq_answer,
            "sources": sources,
            "retrieved_articles": retrieved_articles,
            "model_used": "⚠️ Groq API Fallback (RAG Grounded)",
        }

    # Extractive answer fallback if LLM is unavailable
    extractive_summary = (
        f"Based on retrieved articles ([{', '.join([s['source'] for s in sources])}]):\n"
        f"Top headline: '{retrieved_articles[0].get('title', '')}'. "
        f"{retrieved_articles[0].get('description', '')}"
    )

    return {
        "answer": extractive_summary,
        "sources": sources,
        "retrieved_articles": retrieved_articles,
        "model_used": "⚠️ Heuristic RAG Extractor",
    }
