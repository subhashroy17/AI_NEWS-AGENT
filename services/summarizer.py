import os
import json
import requests
from typing import Dict, Any, Tuple, Optional
from utils.helpers import get_env_var, clean_text, parse_structured_summary

# Singleton caches for local fine-tuned model loading
_LOCAL_MODEL = None
_LOCAL_TOKENIZER = None
_MODEL_LOAD_ATTEMPTED = False
_MODEL_LOAD_SUCCESS = False
_MODEL_STATUS_MESSAGE = "Not Loaded"


def get_model_status() -> Tuple[bool, str]:
    """
    Returns current status of local fine-tuned summarizer model.
    """
    global _MODEL_LOAD_SUCCESS, _MODEL_STATUS_MESSAGE
    return _MODEL_LOAD_SUCCESS, _MODEL_STATUS_MESSAGE


def load_fine_tuned_model() -> Tuple[Any, Any]:
    """
    Attempt to load local fine-tuned open-source Hugging Face model / LoRA adapter.
    """
    global _LOCAL_MODEL, _LOCAL_TOKENIZER, _MODEL_LOAD_ATTEMPTED, _MODEL_LOAD_SUCCESS, _MODEL_STATUS_MESSAGE

    if _MODEL_LOAD_ATTEMPTED:
        return _LOCAL_MODEL, _LOCAL_TOKENIZER

    _MODEL_LOAD_ATTEMPTED = True
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models", "lora_news_summarizer")
    model_dir = os.path.abspath(model_dir)

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from peft import PeftModel, PeftConfig

        # Check if LoRA adapter directory exists
        if os.path.exists(model_dir) and (
            os.path.exists(os.path.join(model_dir, "adapter_config.json"))
            or os.path.exists(os.path.join(model_dir, "config.json"))
        ):
            if os.path.exists(os.path.join(model_dir, "adapter_config.json")):
                config = PeftConfig.from_pretrained(model_dir)
                base_model_name = config.base_model_name_or_path
                _LOCAL_TOKENIZER = AutoTokenizer.from_pretrained(base_model_name)
                base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, torch_dtype=torch.float32)
                _LOCAL_MODEL = PeftModel.from_pretrained(base_model, model_dir)
            else:
                _LOCAL_TOKENIZER = AutoTokenizer.from_pretrained(model_dir)
                _LOCAL_MODEL = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

            _LOCAL_MODEL.eval()
            _MODEL_LOAD_SUCCESS = True
            _MODEL_STATUS_MESSAGE = "✓ Fine-Tuned Open-Source Model (LoRA Loaded)"
            return _LOCAL_MODEL, _LOCAL_TOKENIZER

        else:
            _MODEL_LOAD_SUCCESS = False
            _MODEL_STATUS_MESSAGE = "⚠️ Fine-Tuned Model directory not found. Using Groq API Fallback."
            return None, None

    except Exception as e:
        _MODEL_LOAD_SUCCESS = False
        _MODEL_STATUS_MESSAGE = f"⚠️ Fine-Tuned Model load unavailable: {str(e)[:60]}. Using Groq Fallback."
        return None, None


def summarize_with_fine_tuned_model(title: str, content: str) -> Optional[Dict[str, Any]]:
    """
    Generate summary using the local fine-tuned Hugging Face LLM (PRIMARY).
    """
    model, tokenizer = load_fine_tuned_model()
    if model is None or tokenizer is None:
        return None

    try:
        import torch

        prompt = (
            f"Summarize news article with title, summary, key points, why it matters, and topics:\n"
            f"Title: {title}\nContent: {content[:1000]}"
        )
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=250,
                num_beams=2,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )

        raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        structured = parse_structured_summary(raw_output, default_title=title)
        return structured
    except Exception:
        return None


def summarize_with_groq_fallback(title: str, content: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Generate summary using Groq API as FALLBACK when local fine-tuned model is unavailable.
    """
    api_key = get_env_var("GROQ_API_KEY")
    model_name = get_env_var("GROQ_MODEL", "llama-3.1-8b-instant")

    if not api_key:
        return None, "Groq API key missing in environment variables."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    system_prompt = (
        "You are an expert news intelligence agent. "
        "Summarize the provided article accurately into clear, structured sections. "
        "Strictly respond in this exact format:\n"
        "Title: <Concise Title>\n"
        "Summary: <2-3 clear sentences summarizing core events>\n"
        "Key Points:\n"
        "- <Bullet 1>\n"
        "- <Bullet 2>\n"
        "- <Bullet 3>\n"
        "Why It Matters: <1-2 sentences explaining real-world impact>\n"
        "Topics: <Comma separated topics>"
    )

    user_prompt = f"Title: {title}\nArticle Content: {content[:1500]}"

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
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
        raw_text = resp.json()["choices"][0]["message"]["content"].strip()
        structured = parse_structured_summary(raw_text, default_title=title)
        return structured, "Success"
    except Exception as e:
        return None, f"Groq request error: {str(e)}"


def generate_heuristic_summary(title: str, content: str) -> Dict[str, Any]:
    """
    Generate a clean heuristic summary if all AI models are unavailable.
    """
    cleaned_content = clean_text(content)
    sentences = [s.strip() for s in cleaned_content.split(".") if len(s.strip()) > 15]

    summary_text = ". ".join(sentences[:3]) + "." if sentences else cleaned_content[:200]
    key_points = [s for s in sentences[:3]] if sentences else ["Key updates reported in full story."]
    why_it_matters = "Stay informed on ongoing developments reported by news sources."

    return {
        "title": title,
        "summary": summary_text,
        "key_points": key_points,
        "why_it_matters": why_it_matters,
        "topics": ["News", "Updates"],
    }


def summarize_article(article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Primary Summarization Pipeline:
    1. Try Fine-tuned LLM (PRIMARY)
    2. If unavailable/error -> Groq API (FALLBACK)
    3. If unavailable/error -> Heuristic Fallback
    """
    title = clean_text(article.get("title", ""))
    content = clean_text(article.get("content") or article.get("description") or title)

    # 1. Try Fine-Tuned Model (PRIMARY)
    fine_tuned_summary = summarize_with_fine_tuned_model(title, content)
    if fine_tuned_summary:
        return {
            **fine_tuned_summary,
            "model_used": "✓ Fine-Tuned Open-Source Model (LoRA)",
            "is_fallback": False,
        }

    # 2. Try Groq API (FALLBACK)
    groq_summary, err_msg = summarize_with_groq_fallback(title, content)
    if groq_summary:
        return {
            **groq_summary,
            "model_used": "⚠️ Groq API Fallback",
            "is_fallback": True,
        }

    # 3. Heuristic Fallback
    heuristic = generate_heuristic_summary(title, content)
    return {
        **heuristic,
        "model_used": "⚠️ Heuristic Fallback",
        "is_fallback": True,
    }
