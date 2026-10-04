import os
import sys

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)

def build_pdf_notes(output_filename="AI_News_Intelligence_Agent_Notes.pdf"):
    pdf_path = os.path.abspath(output_filename)
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    PRIMARY_DARK = colors.HexColor("#0F172A")
    ACCENT_PURPLE = colors.HexColor("#4F46E5")
    ACCENT_CYAN = colors.HexColor("#0284C7")
    TEXT_MAIN = colors.HexColor("#1E293B")
    BG_LIGHT = colors.HexColor("#F8FAFC")
    BORDER_COLOR = colors.HexColor("#E2E8F0")

    # Custom Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=PRIMARY_DARK,
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=ACCENT_PURPLE,
        spaceAfter=12,
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=17,
        textColor=ACCENT_PURPLE,
        spaceBefore=12,
        spaceAfter=6,
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=ACCENT_CYAN,
        spaceBefore=8,
        spaceAfter=4,
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=TEXT_MAIN,
        spaceAfter=5,
    )

    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=TEXT_MAIN,
        leftIndent=12,
        spaceAfter=4,
    )

    story = []

    # Title Section
    story.append(Paragraph("AI News Intelligence Agent", title_style))
    story.append(Paragraph("Complete Project Guide & Interview Notes (Simple English Edition)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=ACCENT_PURPLE, spaceAfter=12))

    # SECTION 1: THE BIG PICTURE
    story.append(Paragraph("1. The Big Picture: What Problem Does This Project Solve?", h1_style))
    p1 = (
        "<b>Real-World Problem:</b> Every morning, tech professionals and researchers read news from dozens of websites "
        "(TechCrunch, Reuters, Bloomberg, etc.). They face 3 major problems:<br/>"
        "1. <b>Headline Overload:</b> Too many articles to read in full.<br/>"
        "2. <b>Duplicate News:</b> 10 different websites report on the exact same press release.<br/>"
        "3. <b>Superficial Content:</b> Articles are full of promotional filler and miss <i>why it matters</i>.<br/><br/>"
        "<b>Our Solution:</b> The <b>AI News Intelligence Agent</b> automates the entire news lifecycle. "
        "It fetches current news, uses a <b>Fine-Tuned Open-Source AI Model</b> to write crisp bulleted summaries, "
        "groups duplicate stories using vector math, and lets you ask questions directly to the news using <b>RAG (Retrieval-Augmented Generation)</b>."
    )
    story.append(Paragraph(p1, body_style))

    # SECTION 2: TECH STACK & WHY WE CHOSE IT
    story.append(Spacer(1, 8))
    story.append(Paragraph("2. Tech Stack & Why Each Technology Was Chosen", h1_style))

    tech_data = [
        [Paragraph("<b>Technology</b>", body_style), Paragraph("<b>Role in Project</b>", body_style), Paragraph("<b>Why Chosen? (Simple English)</b>", body_style)],
        [
            Paragraph("<b>Streamlit</b>", body_style),
            Paragraph("Frontend Dashboard", body_style),
            Paragraph("Allows building interactive web dashboards purely in Python without writing HTML/JS/CSS boilerplate.", body_style),
        ],
        [
            Paragraph("<b>PyTorch & Transformers</b>", body_style),
            Paragraph("Local AI Model Engine", body_style),
            Paragraph("Industry standard open-source framework for running machine learning models locally.", body_style),
        ],
        [
            Paragraph("<b>PEFT (LoRA)</b>", body_style),
            Paragraph("Model Fine-Tuning", body_style),
            Paragraph("Fine-tunes open-source LLMs by training only 0.4% adapter parameters, making training fast on standard PCs.", body_style),
        ],
        [
            Paragraph("<b>Sentence Transformers</b><br/><i>(all-MiniLM-L6-v2)</i>", body_style),
            Paragraph("Text Embeddings", body_style),
            Paragraph("Converts text into 384-dimensional mathematical vectors to measure concept meaning and similarity.", body_style),
        ],
        [
            Paragraph("<b>FAISS</b>", body_style),
            Paragraph("Vector Database Search", body_style),
            Paragraph("Facebook AI Similarity Search — performs sub-millisecond vector retrieval for grounded Q&A.", body_style),
        ],
        [
            Paragraph("<b>Groq API</b>", body_style),
            Paragraph("Cloud Fallback Engine", body_style),
            Paragraph("Acts as a fallback LLM if the user's local PC lacks GPU memory to run the model locally.", body_style),
        ],
    ]

    t = Table(tech_data, colWidths=[110, 110, 320])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EEF2FF")),
        ('TEXTCOLOR', (0, 0), (-1, 0), PRIMARY_DARK),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t)

    # SECTION 3: CORE AI CONCEPTS EXPLAINED SIMPLY
    story.append(Spacer(1, 8))
    story.append(Paragraph("3. Core AI Concepts Explained in Simple Analogies", h1_style))

    story.append(Paragraph("A. What is Fine-Tuning (LoRA)?", h2_style))
    p_lora = (
        "<b>Analogy:</b> Imagine hiring a general college graduate who speaks fluent English. They can write long essays, "
        "but they don't know how to write concise 3-bullet medical summaries. Fine-tuning is giving them specialized training.<br/>"
        "<b>What is LoRA?</b> Instead of retraining their whole brain (costing millions of dollars), "
        "LoRA attaches a small specialized notebook (adapter weights). They consult the notebook to output "
        "perfectly formatted news summaries every time. We train only <b>0.4% parameters</b>!"
    )
    story.append(Paragraph(p_lora, body_style))

    story.append(Paragraph("B. What are Embeddings & Cosine Similarity?", h2_style))
    p_emb = (
        "<b>Analogy:</b> Embeddings map concepts on a 2D map. 'Apple' and 'Microsoft' are placed close together "
        "under 'Tech', while 'Banana' is far away under 'Fruit'.<br/>"
        "Our model converts news articles into 384 numbers (vectors). "
        "<b>Cosine Similarity</b> measures the angle between two article vectors. "
        "If the angle is close to 0°, the articles report on the exact same news event!"
    )
    story.append(Paragraph(p_emb, body_style))

    story.append(Paragraph("C. What is RAG (Retrieval-Augmented Generation)?", h2_style))
    p_rag = (
        "<b>Analogy:</b> An <i>Open-Book Exam</i>.<br/>"
        "Without RAG, if you ask an AI model about today's news, it guesses or invents facts (hallucination).<br/>"
        "With RAG, when you ask a question, the FAISS vector database retrieves the top 3 relevant news articles "
        "and places them in front of the AI model. The AI reads those exact articles and answers your question "
        "with 100% grounded facts."
    )
    story.append(Paragraph(p_rag, body_style))

    # SECTION 4: ARCHITECTURE PIPELINE
    story.append(Spacer(1, 8))
    story.append(Paragraph("4. System Architecture Pipeline (How It Runs Step-by-Step)", h1_style))

    pipeline_text = (
        "1. <b>News Collection:</b> User enters a query -> System pulls headlines via Google News RSS & GNews API.<br/>"
        "2. <b>Summarization:</b> Article content passes through local Fine-Tuned Model (LoRA) -> Structured summary.<br/>"
        "3. <b>Embedding & Indexing:</b> Text is converted into vectors and indexed into FAISS vector database.<br/>"
        "4. <b>Duplicate Detection:</b> Pairwise cosine similarity groups matching headlines across news outlets.<br/>"
        "5. <b>RAG Q&A:</b> User asks a question -> FAISS retrieves context -> Model answers with source links."
    )
    story.append(Paragraph(pipeline_text, body_style))

    # SECTION 5: INTERVIEW QA QUICK REFERENCE
    story.append(Spacer(1, 8))
    story.append(Paragraph("5. How to Answer Key Interview Questions", h1_style))

    q1 = "<b>Q: How did you build this project?</b><br/>" \
         "<i>A: I built a modular Python Streamlit dashboard. I used HuggingFace PEFT to fine-tune an open-source model using LoRA for structured news summarization. I integrated Sentence Transformers and FAISS vector search for semantic duplicate detection and grounded RAG Q&A.</i>"
    story.append(Paragraph(q1, bullet_style))

    q2 = "<b>Q: Why do you have both Fine-Tuning AND RAG? Aren't they redundant?</b><br/>" \
         "<i>A: No! Fine-tuning teaches the model <b>HOW</b> to format and structure news summaries. RAG gives the model <b>WHAT</b> real-time fresh news facts to read today. Fine-tuning handles format/style, while RAG handles up-to-date facts.</i>"
    story.append(Paragraph(q2, bullet_style))

    q3 = "<b>Q: What happens if the local fine-tuned model cannot run on a weak PC?</b><br/>" \
         "<i>A: The architecture uses a resilient multi-tier fallback mechanism. If the local PyTorch model fails to load, it automatically routes inference to Groq API cloud inference. If API keys are absent, it uses a clean heuristic fallback so the app never crashes.</i>"
    story.append(Paragraph(q3, bullet_style))

    # SECTION 6: SUMMARY OF FEATURES
    story.append(Spacer(1, 8))
    story.append(Paragraph("6. Summary of Key Project Features", h1_style))
    story.append(Paragraph("• <b>Live Multi-Source News Engine:</b> Pulls real-time headlines with geolocation support.", bullet_style))
    story.append(Paragraph("• <b>Domain-Specific LoRA Summarizer:</b> 3-bullet points + 'Why It Matters' impact analysis.", bullet_style))
    story.append(Paragraph("• <b>Semantic Duplicate Clustering:</b> Groups matching coverage across publishers.", bullet_style))
    story.append(Paragraph("• <b>Grounded RAG Q&A:</b> Interactive question answering with source citations.", bullet_style))
    story.append(Paragraph("• <b>Futuristic Dark UI:</b> Modern glassmorphism dashboard with real-time model status badges.", bullet_style))

    # Build PDF
    doc.build(story)
    print(f"[OK] PDF Notes successfully generated at: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    build_pdf_notes()
