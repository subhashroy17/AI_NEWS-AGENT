import os
import json
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load environment variables once at module import
load_dotenv()


def get_env_var(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Safely fetch environment variables without exposing sensitive values.
    """
    value = os.getenv(key, default)
    if value:
        value = value.strip()
    return value


def clean_text(text: Optional[str]) -> str:
    """
    Sanitize text input by stripping html tags and excess whitespace.
    """
    if not text:
        return ""
    # Strip HTML tags
    cleaned = re.sub(r"<[^>]+>", "", text)
    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def truncate_text(text: str, max_words: int = 40) -> str:
    """
    Truncate text to a maximum word limit with ellipsis.
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def parse_structured_summary(raw_summary: str, default_title: str = "") -> Dict[str, Any]:
    """
    Parse a raw summary output into structured JSON containing summary, key points,
    why it matters, and topics.
    """
    structured = {
        "title": default_title,
        "summary": "",
        "key_points": [],
        "why_it_matters": "",
        "topics": [],
    }

    if not raw_summary:
        return structured

    # If already JSON string
    if raw_summary.strip().startswith("{") and raw_summary.strip().endswith("}"):
        try:
            parsed = json.loads(raw_summary.strip())
            return {**structured, **parsed}
        except Exception:
            pass

    # Extract sections using simple pattern matching
    lines = [line.strip() for line in raw_summary.split("\n") if line.strip()]

    current_section = "summary"
    summary_lines = []
    key_points = []
    why_it_matters_lines = []
    topics = []

    for line in lines:
        lower_line = line.lower()
        if lower_line.startswith("title:"):
            structured["title"] = line.split(":", 1)[1].strip()
        elif lower_line.startswith("summary:"):
            current_section = "summary"
            summary_lines.append(line.split(":", 1)[1].strip())
        elif lower_line.startswith("key points:") or lower_line.startswith("key point:"):
            current_section = "key_points"
        elif lower_line.startswith("why it matters:") or lower_line.startswith("impact:"):
            current_section = "why_it_matters"
            why_it_matters_lines.append(line.split(":", 1)[1].strip())
        elif lower_line.startswith("topics:") or lower_line.startswith("tags:"):
            current_section = "topics"
            raw_topics = line.split(":", 1)[1].strip()
            topics.extend([t.strip() for t in raw_topics.split(",") if t.strip()])
        else:
            if current_section == "summary":
                summary_lines.append(line)
            elif current_section == "key_points":
                cleaned_item = re.sub(r"^[\*\-\d\.\s]+", "", line).strip()
                if cleaned_item:
                    key_points.append(cleaned_item)
            elif current_section == "why_it_matters":
                why_it_matters_lines.append(line)
            elif current_section == "topics":
                topics.extend([t.strip() for t in line.split(",") if t.strip()])

    structured["summary"] = " ".join(summary_lines).strip() or raw_summary.strip()
    structured["key_points"] = key_points if key_points else ["Key details covered in article."]
    structured["why_it_matters"] = (
        " ".join(why_it_matters_lines).strip() or "Provides important updates on current events and domain developments."
    )
    structured["topics"] = list(set(topics)) if topics else ["News", "General"]

    return structured


def load_json_file(filepath: str) -> Any:
    """
    Safely load a JSON file.
    """
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(data: Any, filepath: str) -> None:
    """
    Safely save data to a JSON file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
