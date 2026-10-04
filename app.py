import os
import streamlit as st
from urllib.parse import quote

# Import clean modular services and helpers
from services.news_service import (
    ip_geolocate,
    reverse_geocode,
    geocode_place,
    format_location,
    fetch_all_news,
    get_sample_news,
)
from services.summarizer import (
    summarize_article,
    load_fine_tuned_model,
    get_model_status,
)
from services.embeddings import (
    build_article_vector_index,
    get_embeddings_status,
)
from services.duplicate_detector import detect_duplicate_stories
from services.rag import ask_about_news
from utils.helpers import get_env_var

# ----------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------
st.set_page_config(
    page_title="AI News Intelligence Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------
# HIGH-CONTRAST FUTURISTIC DARK THEME CSS
# ----------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #060913 !important;
        color: #F8FAFC !important;
    }

    /* Sidebar Dark Styling */
    [data-testid="stSidebar"] {
        background-color: #0B0F19 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    [data-testid="stSidebar"] * {
        color: #E2E8F0 !important;
    }

    /* Top Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #1E1B4B 0%, #0F172A 50%, #111827 100%);
        border-radius: 20px;
        padding: 2.2rem 2.5rem;
        color: #FFFFFF;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px -5px rgba(99, 102, 241, 0.25);
        border: 1px solid rgba(129, 140, 248, 0.3);
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        background: linear-gradient(90deg, #FFFFFF 0%, #A5B4FC 50%, #38BDF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #CBD5E1;
        font-weight: 500;
        margin-bottom: 1.2rem;
    }
    .hero-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.25);
        color: #A5B4FC;
        border: 1px solid rgba(129, 140, 248, 0.5);
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        transition: all 0.25s ease;
    }
    .metric-card:hover {
        border-color: rgba(129, 140, 248, 0.5);
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.3);
        transform: translateY(-2px);
    }
    .metric-title {
        font-size: 0.82rem;
        color: #94A3B8;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 800;
        color: #F8FAFC;
    }

    /* Status Pills */
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .pill-green {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(52, 211, 153, 0.4);
    }
    .pill-amber {
        background-color: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(251, 191, 36, 0.4);
    }
    .pill-indigo {
        background-color: rgba(99, 102, 241, 0.2);
        color: #818CF8;
        border: 1px solid rgba(129, 140, 248, 0.4);
    }
    .pill-slate {
        background-color: rgba(51, 65, 85, 0.6);
        color: #E2E8F0;
        border: 1px solid rgba(148, 163, 184, 0.3);
    }

    /* News Article Card */
    .article-box {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
        transition: all 0.25s ease;
    }
    .article-box:hover {
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 0 25px rgba(99, 102, 241, 0.2);
    }
    .article-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #F8FAFC;
        line-height: 1.35;
        margin-bottom: 0.6rem;
    }
    .article-meta {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 12px;
        font-size: 0.83rem;
        color: #94A3B8;
        margin-bottom: 1.1rem;
    }
    .summary-text {
        font-size: 0.98rem;
        color: #E2E8F0;
        line-height: 1.6;
        margin-bottom: 1.2rem;
    }
    .why-matters-box {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.1) 0%, rgba(3, 105, 161, 0.2) 100%);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-left: 4px solid #38BDF8;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .why-matters-title {
        font-size: 0.85rem;
        font-weight: 800;
        color: #38BDF8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.2rem;
    }
    .why-matters-content {
        font-size: 0.93rem;
        color: #F0F9FF;
        font-weight: 500;
    }

    /* Sidebar Dark Card */
    .sidebar-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
    }
    .sidebar-title {
        font-size: 0.9rem;
        font-weight: 800;
        color: #818CF8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.8rem;
    }

    /* Streamlit UI Dark Component Overrides */
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
        background-color: #1E293B !important;
        border-color: rgba(255, 255, 255, 0.15) !important;
        color: #F8FAFC !important;
    }
    input, select, textarea {
        color: #F8FAFC !important;
    }

    /* Streamlit Expander Dark Theme */
    div[data-testid="stExpander"] {
        background-color: #0F172A !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
    }
    div[data-testid="stExpander"] summary {
        color: #E2E8F0 !important;
        font-weight: 600 !important;
    }

    /* Streamlit Tabs Dark Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: #0F172A !important;
        color: #94A3B8 !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 0px 20px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
        color: #FFFFFF !important;
        border-color: #818CF8 !important;
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.4) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------
# MODEL INITIALIZATION & STATUS
# ----------------------------------------------------
with st.spinner("Initializing AI Intelligence Pipeline..."):
    load_fine_tuned_model()

ft_loaded, ft_status_msg = get_model_status()
emb_ready, emb_status_msg = get_embeddings_status()
groq_key = get_env_var("GROQ_API_KEY")

# Session State Initialization
if "articles" not in st.session_state:
    st.session_state.articles = get_sample_news()
if "summaries" not in st.session_state:
    st.session_state.summaries = {}
if "selected_article_for_rag" not in st.session_state:
    st.session_state.selected_article_for_rag = ""

# ----------------------------------------------------
# HERO HEADER BANNER
# ----------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-tag">⚡ NEXT-GEN NEWS INTELLIGENCE ENGINE</div>
        <div class="hero-title">AI News Intelligence Agent</div>
        <div class="hero-subtitle">Understand the news, not just read it — Powered by fine-tuned open-source LLMs, FAISS vector search, and RAG.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------
# METRIC HIGHLIGHT CARDS
# ----------------------------------------------------
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Articles Indexed</div>
            <div class="metric-value">{len(st.session_state.articles)} Headlines</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m_col2:
    status_html = '<span class="pill pill-green">✓ Fine-Tuned (LoRA)</span>' if ft_loaded else (
        '<span class="pill pill-amber">⚠️ Groq Fallback</span>' if groq_key else '<span class="pill pill-slate">⚠️ Heuristic</span>'
    )
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Primary Summarizer</div>
            <div class="metric-value">{status_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m_col3:
    emb_html = '<span class="pill pill-indigo">all-MiniLM-L6-v2</span>' if emb_ready else '<span class="pill pill-slate">Standby</span>'
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Embedding Model</div>
            <div class="metric-value">{emb_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m_col4:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-title">Vector Search Engine</div>
            <div class="metric-value"><span class="pill pill-green">FAISS Index Ready</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------------------------------
# CLEAN & RELEVANT SIDEBAR CONTROLS
# ----------------------------------------------------
st.sidebar.markdown(
    """
    <div class="sidebar-card">
        <div class="sidebar-title">📍 Location & Region Filter</div>
    </div>
    """,
    unsafe_allow_html=True,
)

target_location = st.sidebar.text_input(
    "Enter Location / Region:",
    value="Tirupati, India",
    placeholder="e.g. Tirupati, India or New York",
)

if target_location.strip():
    g = geocode_place(target_location.strip())
    location_info = g if g else ip_geolocate()
else:
    location_info = ip_geolocate()

st.sidebar.caption(f"📍 **Active Location:** {format_location(location_info)}")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div class="sidebar-card">
        <div class="sidebar-title">⚙️ RAG & Model Controls</div>
    </div>
    """,
    unsafe_allow_html=True,
)

rag_top_k = st.sidebar.slider("RAG Retrieved Context Sources (Top-K):", min_value=1, max_value=10, value=3, step=1)
model_temp = st.sidebar.slider("AI Temperature (Creativity):", min_value=0.1, max_value=1.0, value=0.2, step=0.1)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div class="sidebar-card">
        <div class="sidebar-title">🤖 Active Model Pipeline</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if ft_loaded:
    st.sidebar.success("✓ Fine-Tuned Model (LoRA) Active")
elif groq_key:
    st.sidebar.warning("⚠️ Groq API Fallback Active")
else:
    st.sidebar.info("⚠️ Heuristic Fallback Active")

# ----------------------------------------------------
# DASHBOARD TABS
# ----------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📰 News Intelligence",
    "👯 Related & Duplicates",
    "💬 Ask About This News (RAG)",
    "⚙️ Model Architecture & LoRA",
])

# ----------------------------------------------------
# TAB 1: NEWS INTELLIGENCE STREAM
# ----------------------------------------------------
with tab1:
    st.markdown("### 🔎 Real-Time Intelligence Search")
    col_search, col_cat, col_btn = st.columns([3, 2, 1])

    with col_search:
        search_query = st.text_input("Keyword / Topic:", value="AI Technology", placeholder="e.g., Quantum Computing, Climate Policy, Finance...")

    with col_cat:
        category_choice = st.selectbox("Filter Domain:", ["All Topics", "Technology", "Business", "Science", "Local News"])

    with col_btn:
        st.write("")
        st.write("")
        fetch_clicked = st.button("🚀 Fetch News", use_container_width=True)

    if fetch_clicked:
        with st.spinner("Fetching, cleaning, and embedding live headlines..."):
            query_to_use = search_query
            if category_choice == "Local News":
                loc_str = format_location(location_info)
                query_to_use = f"{search_query} {loc_str}"
            elif category_choice != "All Topics":
                query_to_use = f"{category_choice} {search_query}"

            fetched = fetch_all_news(query_to_use, location_info=location_info)
            st.session_state.articles = fetched
            st.session_state.summaries = {}
            st.success(f"Fetched {len(fetched)} articles for '{query_to_use}'")

    articles = st.session_state.articles

    # Index articles into vector space
    if articles:
        build_article_vector_index(articles)

    st.markdown("---")

    if not articles:
        st.info("No news articles found. Try adjusting your query.")
    else:
        for idx, art in enumerate(articles):
            art_id = art.get("id", f"art_{idx}")
            title = art.get("title", "Untitled Story")
            source = art.get("source", "News Media")
            pub_date = art.get("published_at", "Recent")
            url = art.get("url", "#")
            desc = art.get("description") or art.get("content", "")

            # Generate or load cached summary
            if art_id not in st.session_state.summaries:
                with st.spinner(f"Analyzing article {idx+1}/{len(articles)}..."):
                    summary_obj = summarize_article(art)
                    st.session_state.summaries[art_id] = summary_obj
            else:
                summary_obj = st.session_state.summaries[art_id]

            model_badge = (
                '<span class="pill pill-green">✓ Fine-Tuned LoRA Model</span>' if "Fine-Tuned" in summary_obj.get("model_used", "")
                else '<span class="pill pill-amber">⚠️ Groq Fallback</span>'
            )

            topics_html = " ".join([f'<span class="pill pill-slate">{t}</span>' for t in summary_obj.get("topics", ["News"])])

            # Modern Article Card UI
            st.markdown(
                f"""
                <div class="article-box">
                    <div class="article-title">{title}</div>
                    <div class="article-meta">
                        <span class="pill pill-slate">📰 {source}</span>
                        <span class="pill pill-slate">🕒 {pub_date}</span>
                        {model_badge}
                    </div>
                    <div class="summary-text"><strong>AI Intelligence Summary:</strong> {summary_obj.get('summary', desc)}</div>
                    <div class="why-matters-box">
                        <div class="why-matters-title">⚡ Why It Matters</div>
                        <div class="why-matters-content">{summary_obj.get('why_it_matters', 'Pivotal development in this sector.')}</div>
                    </div>
                    <div style="margin-top: 0.8rem;">{topics_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_kp, col_actions = st.columns([3, 2])
            with col_kp:
                with st.expander("📌 View Key Bullet Points"):
                    for kp in summary_obj.get("key_points", []):
                        st.markdown(f"- {kp}")

            with col_actions:
                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    st.markdown(f'<a href="{url}" target="_blank" style="text-decoration:none;"><button style="width:100%; border-radius:10px; border:1px solid rgba(255,255,255,0.2); background:#1E293B; color:#F8FAFC; padding:8px; font-weight:600; cursor:pointer;">🔗 Original Article</button></a>', unsafe_allow_html=True)
                with col_a2:
                    if st.button("💬 Ask RAG Agent", key=f"rag_btn_{idx}", use_container_width=True):
                        st.session_state.selected_article_for_rag = title
                        st.success("Selected for RAG Q&A! Switch to 'Ask About This News' tab.")

            st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------------------------------
# TAB 2: SEMANTIC DUPLICATE CLUSTERING
# ----------------------------------------------------
with tab2:
    st.markdown("### 👯 Semantic Coverage & Duplicate Story Clusters")
    st.markdown(
        "Powered by `all-MiniLM-L6-v2` dense vector embeddings. Detects when multiple media outlets "
        "are covering the same underlying event."
    )

    sim_threshold = st.slider("Similarity Threshold:", min_value=0.50, max_value=0.95, value=0.75, step=0.05)

    if not articles:
        st.warning("No articles currently loaded.")
    else:
        clusters = detect_duplicate_stories(articles, similarity_threshold=sim_threshold)

        st.markdown(f"**Identified {len(clusters)} Distinct Story Clusters**")

        for idx, cl in enumerate(clusters):
            primary = cl["primary_article"]
            related = cl["related_articles"]

            with st.expander(f"Cluster #{idx+1}: {primary.get('title')} ({cl['total_coverage']} Sources)", expanded=(idx == 0)):
                st.markdown(f"**Primary Outlet:** `{primary.get('source')}`")
                st.markdown(f"**Headline:** {primary.get('title')}")

                if related:
                    st.markdown("#### Related Coverage Across Publishers:")
                    for rel in related:
                        st.markdown(
                            f"- **{rel.get('title')}** — `{rel.get('source')}` (Similarity Score: `{rel.get('similarity_score')}`)"
                        )
                else:
                    st.caption("No duplicate coverage found for this headline at the current similarity threshold.")

# ----------------------------------------------------
# TAB 3: GROUNDED RAG Q&A
# ----------------------------------------------------
with tab3:
    st.markdown("### 💬 RAG Agent: Ask About This News")
    st.markdown(
        "Ask questions about indexed news. The agent retrieves relevant story snippets via **FAISS Vector Search** "
        "and generates a grounded answer."
    )

    preset_q = st.session_state.selected_article_for_rag
    user_question = st.text_input(
        "Your Question:",
        value=f"What are the main details about: {preset_q}" if preset_q else "",
        placeholder="e.g., What breakthrough was announced regarding semiconductors?",
    )

    if st.button("🚀 Ask RAG Agent", use_container_width=True):
        if not user_question.strip():
            st.warning("Please enter a valid question.")
        else:
            with st.spinner("Searching FAISS vector index & generating grounded answer..."):
                rag_response = ask_about_news(user_question, articles, top_k=rag_top_k)

                st.markdown(
                    f"""
                    <div class="article-box" style="border-left: 4px solid #818CF8;">
                        <div style="font-size:1.1rem; font-weight:800; color:#818CF8; margin-bottom:0.4rem;">🤖 Grounded Answer</div>
                        <div style="font-size:1rem; color:#F8FAFC; line-height:1.6;">{rag_response.get('answer')}</div>
                        <div style="margin-top:0.8rem;"><span class="pill pill-slate">Model Engine: {rag_response.get('model_used')}</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("#### 📚 Retrieved Context Sources:")
                for src in rag_response.get("sources", []):
                    st.markdown(f"- **[{src.get('title')}]({src.get('url')})** — `{src.get('source')}` (Relevance Score: `{src.get('relevance_score')}`)")

# ----------------------------------------------------
# TAB 4: ARCHITECTURE & LoRA TRAINING
# ----------------------------------------------------
with tab4:
    st.markdown("### ⚙️ System Architecture & Model Pipeline")

    st.markdown(
        """
        ```
                     NEWS ARTICLE
                          ↓
                   Preprocessing
                          ↓
               Fine-tuned Open-Source LLM
                  (LoRA / QLoRA)
                          ↓
                 News Intelligence Summary
                          ↓
               Sentence Embeddings & FAISS
                          ↓
                Streamlit Intelligence UI
        ```
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div class="article-box">
                <div style="font-weight:800; font-size:1.1rem; color:#F8FAFC; margin-bottom:0.4rem;">1. PRIMARY MODEL</div>
                <div>Local Fine-Tuned LLM (LoRA)</div>
                <div style="margin-top:0.6rem;">
                    {'<span class="pill pill-green">✓ ACTIVE & LOADED</span>' if ft_loaded else '<span class="pill pill-amber">⚠️ NOT LOADED</span>'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="article-box">
                <div style="font-weight:800; font-size:1.1rem; color:#F8FAFC; margin-bottom:0.4rem;">2. FALLBACK MODEL</div>
                <div>Groq API (Cloud Inference)</div>
                <div style="margin-top:0.6rem;">
                    {'<span class="pill pill-green">✓ CONFIGURED</span>' if groq_key else '<span class="pill pill-slate">⚠️ NO API KEY</span>'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🏋️ LoRA Fine-Tuning Execution Commands")
    st.code(
        """
# Step 1: Prepare news instruction dataset
python training/prepare_dataset.py

# Step 2: Train LoRA adapter parameters
python training/train_lora.py

# Step 3: Evaluate fine-tuned model performance
python training/evaluate.py
        """,
        language="bash",
    )
