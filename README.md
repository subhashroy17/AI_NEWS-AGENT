# AI News Intelligence Agent 📰🤖

An end-to-end AI-powered news intelligence dashboard built in Python and Streamlit. The application collects current news from live APIs and RSS feeds, summarizes articles using a **domain-specific fine-tuned open-source LLM (LoRA)**, detects duplicate/related coverage across news outlets using **dense vector embeddings**, and enables **Retrieval-Augmented Generation (RAG)** so users can interactively ask questions grounded in real-time news data.

---

## 1. Project Overview

Modern readers are overwhelmed by duplicate news headlines and fragmented reporting across publishers. The **AI News Intelligence Agent** addresses news overload by delivering structured, key-point summaries, grouping semantically duplicate stories, and offering grounded Q&A over current events.

> **IMPORTANT MODEL ARCHITECTURE**:
> - **PRIMARY MODEL**: Fine-tuned open-source Hugging Face LLM (PEFT / LoRA).
> - **FALLBACK MODEL**: Groq API (`llama-3.1-8b-instant`), triggered ONLY when local model is unavailable or hardware constraints prevent local inference.

---

## 2. Key Features

- **Multi-Source News Collection**: Live news fetching via Google News RSS, GNews API, and location-based geocoding.
- **Domain-Specific Fine-Tuned Summarization**: Generates structured summaries (Title, Summary, Key Points, Why It Matters, Topics) using a fine-tuned open-source model.
- **Semantic Duplicate Detection**: Uses dense vector embeddings to cluster related stories across publishers with configurable similarity thresholds.
- **RAG ("Ask About This News")**: Vector search via **FAISS** over current articles with grounded, non-hallucinated QA responses.
- **Real-Time System & Model Status**: Live visual indicators showing whether the Fine-Tuned Model or Groq Fallback is active.
- **Offline Fallback Capability**: Embedded sample news data ensures the project runs smoothly even without network connectivity or API keys.

---

## 3. System Architecture

```
                       NEWS ARTICLE
                            ↓
                     Preprocessing
                            ↓
               Fine-tuned Open-Source LLM
                     (LoRA / QLoRA)
                            ↓
                    AI News Summary
                            ↓
               Sentence Embeddings & FAISS
                            ↓
                  Duplicate Detection + RAG
                            ↓
                     Streamlit UI
```

---

## 4. Technology Stack

- **Frontend**: Streamlit
- **Fine-Tuning & LLM Inference**: PyTorch, Hugging Face Transformers, PEFT (LoRA), Accelerate
- **Embeddings & Vector Search**: Sentence-Transformers (`all-MiniLM-L6-v2`), FAISS (`faiss-cpu`)
- **API Fallback**: Groq API (`llama-3.1-8b-instant`), GNews API
- **News Collection & Geocoding**: Feedparser, Requests, Nominatim OpenStreetMap, ipapi.co

---

## 5. Primary vs Fallback Model Logic

1. **PRIMARY (Fine-Tuned Model)**: The application looks for adapter weights in `models/lora_news_summarizer/`. If found, it runs local inference using the fine-tuned model.
2. **FALLBACK (Groq API)**: If the fine-tuned model is not present, fails to load, or runs out of GPU memory, the system seamlessly switches to Groq API.
3. **HEURISTIC FALLBACK**: If API keys are also missing or network requests fail, a clean heuristic extraction fallback ensures zero application crashes.

---

## 6. Project Structure

```
AI_NEWS_AGENT/
│
├── app.py                     # Main Streamlit Dashboard UI
├── requirements.txt           # Verified project dependencies
├── .env.example               # Safe environment variable template
├── .gitignore                 # Excludes secrets, models, and virtualenvs
├── README.md                  # Complete project documentation
├── INTERVIEW_NOTES.md         # Comprehensive interview QA guide
│
├── services/
│   ├── news_service.py        # Location geocoding & multi-source news fetchers
│   ├── summarizer.py          # Primary LoRA + Groq Fallback summarization engine
│   ├── embeddings.py          # Sentence Transformers & FAISS index search
│   ├── duplicate_detector.py  # Semantic similarity clustering logic
│   └── rag.py                 # Retrieval-Augmented Generation Q&A engine
│
├── training/
│   ├── prepare_dataset.py     # Generates news instruction dataset
│   ├── train_lora.py          # Trains PEFT LoRA adapter on base model
│   └── evaluate.py            # Evaluates model outputs vs reference summaries
│
├── models/
│   └── README.md              # Instructions for adapter directory
│
├── data/
│   ├── sample_news.json       # Fallback sample news dataset
│   └── news_summary_dataset.json
│
└── utils/
    └── helpers.py             # Text cleaning, parsing, and env helpers
```

---

## 7. Installation & Setup

### Prerequisites
- Python 3.9+ installed

### Step 1: Clone Repository & Create Virtual Environment
```bash
python -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on Linux/Mac:
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Create a `.env` file in the root directory based on `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys (optional for fallback usage):
```env
GROQ_API_KEY=your_actual_groq_api_key
GROQ_MODEL=llama-3.1-8b-instant
GNEWS_API_KEY=your_actual_gnews_api_key
```

*Note: Never commit your `.env` file containing secrets! `.env` is listed in `.gitignore`.*

---

## 8. Running the Application

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`.

---

## 9. Domain-Specific Fine-Tuning Instructions

To train your own LoRA adapter on news summarization data:

1. **Prepare Training Dataset**:
   ```bash
   python training/prepare_dataset.py
   ```
2. **Train LoRA Adapter**:
   ```bash
   python training/train_lora.py
   ```
   *This saves adapter weights to `models/lora_news_summarizer/`.*

3. **Evaluate Fine-Tuned Model**:
   ```bash
   python training/evaluate.py
   ```

4. **Restart App**: The application automatically detects the newly trained adapter and uses it as the **PRIMARY** model!

---

## 10. Fine-Tuning vs. RAG Concept Matrix

| Dimension | Fine-Tuning (LoRA) | Retrieval-Augmented Generation (RAG) |
|---|---|---|
| **Purpose** | Teaches model *style*, *format*, and *task-specific summarization skills*. | Provides model with *fresh, real-time facts* and context. |
| **Knowledge Updates** | Static (frozen at training time). | Dynamic (queries live vector index at runtime). |
| **Primary Component** | Model weights (adapter parameters). | Vector Index (FAISS) + Embedding Model. |
| **Hallucination Prevention** | Moderate (improves output structure). | High (answers grounded strictly in retrieved context). |

---

## 11. Limitations & Future Scope

- **Hardware Memory**: Loading larger base models (e.g. 7B parameter models) locally requires GPU memory; smaller models like `flan-t5` run efficiently on CPU.
- **Future Enhancements**: Integration with vector databases like Qdrant/Chroma, automated RSS cron scraping, and multi-lingual summarization.
