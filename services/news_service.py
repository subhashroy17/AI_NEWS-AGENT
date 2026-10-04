import os
import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote
from typing import List, Dict, Any, Optional
from utils.helpers import get_env_var, clean_text, load_json_file


def ip_geolocate() -> Dict[str, Any]:
    """
    Detect user's approximate location based on IP address.
    """
    try:
        resp = requests.get("https://ipapi.co/json/", timeout=8).json()
        return {
            "lat": resp.get("latitude"),
            "lon": resp.get("longitude"),
            "city": resp.get("city") or "",
            "state": resp.get("region") or "",
            "country": resp.get("country_name") or "",
            "country_code": (resp.get("country_code") or "us").lower(),
        }
    except Exception:
        return {
            "lat": None,
            "lon": None,
            "city": "New York",
            "state": "NY",
            "country": "United States",
            "country_code": "us",
        }


def reverse_geocode(lat: float, lon: float) -> Dict[str, Any]:
    """
    Reverse geocode latitude and longitude using OpenStreetMap Nominatim API.
    """
    if lat is None or lon is None:
        return {}
    try:
        headers = {"User-Agent": "AI-News-Intelligence-Agent/2.0"}
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=10&addressdetails=1"
        r = requests.get(url, headers=headers, timeout=8).json()
        addr = r.get("address", {})
        city = addr.get("city") or addr.get("town") or addr.get("village") or addr.get("hamlet") or ""
        state = addr.get("state") or ""
        country = addr.get("country") or ""
        cc = (addr.get("country_code") or "us").lower()
        return {"city": city, "state": state, "country": country, "country_code": cc}
    except Exception:
        return {}


def geocode_place(place_name: str) -> Dict[str, Any]:
    """
    Geocode a user-entered location query into location metadata.
    """
    if not place_name or not place_name.strip():
        return {}
    try:
        headers = {"User-Agent": "AI-News-Intelligence-Agent/2.0"}
        url = f"https://nominatim.openstreetmap.org/search?format=json&q={quote(place_name.strip())}&limit=1&addressdetails=1"
        r = requests.get(url, headers=headers, timeout=8).json()
        if not r:
            return {}
        hit = r[0]
        lat = float(hit["lat"])
        lon = float(hit["lon"])
        addr = hit.get("address", {})
        city = addr.get("city") or addr.get("town") or addr.get("village") or addr.get("hamlet") or ""
        state = addr.get("state") or ""
        country = addr.get("country") or ""
        cc = (addr.get("country_code") or "us").lower()
        return {"lat": lat, "lon": lon, "city": city, "state": state, "country": country, "country_code": cc}
    except Exception:
        return {}


def format_location(loc: Dict[str, Any]) -> str:
    """
    Format location dictionary into a readable string.
    """
    parts = [p for p in [loc.get("city"), loc.get("state"), loc.get("country")] if p]
    return ", ".join(parts) if parts else "Global / Unknown"


def fetch_google_news_rss(query: str, country_code: str = "us", max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch live news headlines via Google News RSS feed.
    """
    if not query:
        query = "technology AI"
    cc = (country_code or "us").upper()
    url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-{cc}&gl={cc}&ceid={cc}:en"

    items = []
    # Attempt parsing with feedparser first if installed, fallback to ElementTree XML
    try:
        import feedparser
        feed = feedparser.parse(url)
        for entry in feed.entries[:max_results]:
            title = clean_text(entry.get("title", ""))
            desc = clean_text(entry.get("summary") or entry.get("description") or "")
            link = entry.get("link", "#")
            published = entry.get("published", "Recent")
            source = entry.get("source", {}).get("title", "Google News") if isinstance(entry.get("source"), dict) else "Google News"
            if title:
                items.append({
                    "id": f"rss_{hash(link) & 0xffffffff}",
                    "title": title,
                    "description": desc,
                    "content": desc or title,
                    "url": link,
                    "source": source,
                    "published_at": published,
                    "category": query.title(),
                })
    except Exception:
        try:
            xml_data = requests.get(url, timeout=10).text
            root = ET.fromstring(xml_data)
            for idx, item in enumerate(root.findall(".//item")[:max_results]):
                title = clean_text(item.findtext("title") or "")
                link = (item.findtext("link") or "#").strip()
                desc = clean_text(item.findtext("description") or "")
                pub_date = (item.findtext("pubDate") or "Recent").strip()
                if title:
                    items.append({
                        "id": f"rss_{idx}_{hash(link) & 0xffffffff}",
                        "title": title,
                        "description": desc,
                        "content": desc or title,
                        "url": link,
                        "source": "Google News RSS",
                        "published_at": pub_date,
                        "category": query.title(),
                    })
        except Exception:
            pass

    return items


def fetch_gnews_api(query: str, max_results: int = 8) -> List[Dict[str, Any]]:
    """
    Fetch news from GNews API using GNEWS_API_KEY environment variable.
    """
    api_key = get_env_var("GNEWS_API_KEY")
    if not api_key:
        return []

    url = f"https://gnews.io/api/v4/search?lang=en&q={quote(query)}&max={max_results}&token={api_key}"
    try:
        res = requests.get(url, timeout=10).json()
        raw_articles = res.get("articles", []) or []
        items = []
        for idx, a in enumerate(raw_articles):
            title = clean_text(a.get("title", ""))
            desc = clean_text(a.get("description", ""))
            content = clean_text(a.get("content", ""))
            link = a.get("url", "#")
            source_name = a.get("source", {}).get("name", "GNews")
            pub_date = a.get("publishedAt", "Recent")

            if title:
                items.append({
                    "id": f"gnews_{idx}_{hash(link) & 0xffffffff}",
                    "title": title,
                    "description": desc,
                    "content": content or desc or title,
                    "url": link,
                    "source": source_name,
                    "published_at": pub_date,
                    "category": query.title(),
                })
        return items
    except Exception:
        return []


def get_sample_news() -> List[Dict[str, Any]]:
    """
    Load static fallback sample news from data/sample_news.json.
    """
    filepath = os.path.join(os.path.dirname(__file__), "..", "data", "sample_news.json")
    return load_json_file(filepath)


def fetch_all_news(query: str = "AI Technology", location_info: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Unified news collection pipeline combining GNews API, Google News RSS, and sample fallback.
    """
    articles = []
    
    # 1. Attempt GNews API if key exists
    gnews_items = fetch_gnews_api(query)
    if gnews_items:
        articles.extend(gnews_items)

    # 2. Fetch Google News RSS for query
    rss_items = fetch_google_news_rss(query, country_code=location_info.get("country_code", "us") if location_info else "us")
    if rss_items:
        articles.extend(rss_items)

    # 3. Deduplicate articles by title/URL similarity
    unique_articles = []
    seen_titles = set()
    for art in articles:
        normalized_title = art["title"].lower().strip()
        if normalized_title not in seen_titles:
            seen_titles.add(normalized_title)
            unique_articles.append(art)

    # 4. If no live articles returned, fallback to sample news
    if not unique_articles:
        sample = get_sample_news()
        return sample

    return unique_articles
