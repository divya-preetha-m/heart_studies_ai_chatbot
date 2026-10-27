"""
app.py — the Streamlit application (the only file you launch).

RUN IT WITH:
    streamlit run app.py

WHAT THIS FILE DOES (in simple English)
----------------------------------------
It is the user interface and the "traffic controller":
  * the sidebar lets the student choose a page (Home / PDF Chat / Heart
    Anatomy / Quiz Practice),
  * the Home page explains how the app works,
  * the PDF Chat page collects PDFs, asks rag.py to index them, and shows the
    conversation with source references,
  * the Heart Anatomy page gives general explanations (clearly not from PDFs),
  * the Quiz page generates a multiple-choice quiz via quiz.py and scores it.

Everything heavy lifting (PDF parsing, embeddings, FAISS, Gemini calls) lives
in rag.py / quiz.py / utils.py so this file stays readable.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from quiz import format_review, generate_quiz, grade_quiz, score_message, topic_query
from rag import (
    PDFProcessingError,
    answer_from_documents,
    answer_general,
    build_documents,
    build_vector_store,
    extract_text_per_page,
    retrieve,
    split_documents,
)
from utils import (
    APP_ICON,
    APP_TAGLINE,
    APP_TITLE,
    DEFAULT_QUIZ_QUESTIONS,
    DISCLAIMER,
    HEART_TOPICS,
    MAX_QUIZ_QUESTIONS,
    STATIC_LESSONS,
    get_google_api_key,
    inject_theme,
    load_environment,
    render_chat_message,
    safe_filename,
    validate_pdf_uploads,
)

# ---------------------------------------------------------------------------
# PAGE SETUP + SESSION STATE
# ---------------------------------------------------------------------------

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
inject_theme()

PAGES = ["\U0001f3e0 Home", "\U0001f4c4 PDF Chat", "\U0001fac0 Heart Anatomy", "\U0001f4dd Quiz Practice"]


def init_session_state() -> None:
    """Create empty session variables the first time each browser tab runs."""
    defaults = {
        "messages": [],            # PDF-chat history: {"role", "content", "sources"}
        "anatomy_messages": [],    # Heart Anatomy history (kept separate on purpose)
        "vector_store": None,      # the FAISS index for this session
        "doc_stats": [],           # [{name, pages, text_pages, chunks}]
        "quiz": None,              # generated questions
        "quiz_result": None,       # grading result after submission
        "quiz_meta": "",           # label like "Intermediate • 5 questions • PDF mode"
        "bundled_attempted": False,  # auto-load of sample_pdfs/ already tried this session?
        "bundled_loaded": False,     # did that auto-load succeed?
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


SAMPLE_DIR = Path(__file__).resolve().parent / "sample_pdfs"


def auto_load_bundled_samples() -> None:
    """Index the PDFs shipped in sample_pdfs/ so every NEW session starts with
    documents already available — no manual upload needed.

    Runs at most once per session (guarded by `bundled_attempted`, even when it
    fails, so a broken file cannot trigger a rebuild on every rerun) and never
    replaces an index the student built themselves (guarded by `vector_store`).
    Every failure is swallowed: the app still starts, and manual upload works.
    """
    if st.session_state.vector_store is not None or st.session_state.bundled_attempted:
        return
    st.session_state.bundled_attempted = True

    try:
        pdf_paths = sorted(SAMPLE_DIR.glob("*.pdf")) if SAMPLE_DIR.is_dir() else []
        documents, stats = [], []
        for path in pdf_paths:
            name = safe_filename(path.name)
            try:
                pages = extract_text_per_page(path.read_bytes(), name)
            except PDFProcessingError:
                continue  # one broken bundled file must not block the others
            docs = build_documents(pages, name)
            if not docs:
                continue
            documents.extend(docs)
            stats.append({"name": name, "pages": len(pages), "text_pages": len(docs), "chunks": 0})
        if not documents:
            return

        chunks = split_documents(documents)
        if not chunks:
            return

        per_file: dict[str, int] = {}
        for c in chunks:
            src = c.metadata.get("source", "unknown")
            per_file[src] = per_file.get(src, 0) + 1
        for s in stats:
            s["chunks"] = per_file.get(s["name"], 0)

        with st.spinner("📚 Loading bundled study material so your session starts ready to chat…"):
            store = build_vector_store(chunks)
    except Exception:  # noqa: BLE001 - never block the app; manual upload still works
        return

    st.session_state.vector_store = store
    st.session_state.doc_stats = stats
    st.session_state.messages = []          # fresh session => fresh conversation
    st.session_state.bundled_loaded = True


init_session_state()
load_environment()
api_key = get_google_api_key()
auto_load_bundled_samples()


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(f"## {APP_ICON} {APP_TITLE}")
    st.caption(APP_TAGLINE)
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.divider()

    # Status panel — shows state, never any secret values.
    if api_key:
        st.success("\u2705 Gemini API key loaded \u00b7 model ready")
    else:
        st.error("\U0001f511 No Gemini API key found")
        with st.expander("How to add your key"):
            st.markdown(
                "1. Get a free key at <https://aistudio.google.com/apikey>\n"
                "2. Copy `.env.example` to `.env`\n"
                "3. Paste the key into `GOOGLE_API_KEY=...`\n"
                "4. Restart the app"
            )

    indexed = st.session_state.doc_stats
    if indexed:
        total_chunks = sum(d.get("chunks", 0) for d in indexed)
        st.info(f"\U0001f4c1 {len(indexed)} document(s) \u00b7 {total_chunks} chunks indexed")
    else:
        st.info("\U0001f4c1 No documents indexed yet")

    st.divider()
    st.caption("\u26a0\ufe0f For **education only** — not for diagnosis or treatment.")


# ---------------------------------------------------------------------------
# PAGE 1: HOME
# ---------------------------------------------------------------------------


def render_home() -> None:
    st.markdown(
        f'<div style="text-align:center; font-size:3rem;"><span class="heart-pulse">{APP_ICON}</span></div>',
        unsafe_allow_html=True,
    )
    st.title(APP_TITLE)
    st.markdown(
        f"<p style='text-align:center; color:#777;'>{APP_TAGLINE}</p>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="info-card"><h4>\U0001f4c4 1. Upload PDFs</h4>'
            "<p>Add your lecture notes or textbook chapters. The text is split "
            "into chunks and converted into embeddings (numbers that represent "
            "meaning) inside a FAISS index.</p></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="info-card"><h4>\U0001f4ac 2. Ask questions</h4>'
            "<p>The chatbot finds the chunks most similar to your question "
            "(<b>retrieval</b>) and Gemini answers using only that material "
            "(<b>augmented generation</b>) — with page references.</p></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="info-card"><h4>\U0001f4dd 3. Practise</h4>'
            "<p>Auto-generated multiple-choice quizzes test what you learned, "
            "show instant feedback and a review of the answers you missed.</p></div>",
            unsafe_allow_html=True,
        )

    st.subheader("\U0001f9ed How RAG works — step by step")
    st.markdown(
        "1. **Extract** — `pypdf` pulls the text from every page of your PDFs.\n"
        "2. **Chunk** — a LangChain splitter cuts long text into ~1000-character pieces.\n"
        "3. **Embed** — `all-MiniLM-L6-v2` turns each chunk into 384 numbers that capture its meaning.\n"
        "4. **Index** — the vectors go into a **FAISS** index for fast similarity search.\n"
        "5. **Retrieve** — your question is embedded the same way; the closest chunks are selected.\n"
        "6. **Generate** — Gemini writes an answer from *those chunks only*, or honestly "
        "says the materials do not cover your question."
    )

    st.subheader("\U0001f6ab Before you start")
    st.info(DISCLAIMER)
    if not api_key:
        st.warning(
            "\U0001f511 **No Gemini API key detected.** Chat, learning mode and quizzes need "
            "one; PDF upload and indexing still work. See the sidebar for setup steps."
        )
    with st.expander("\U0001f4a1 Ideas for questions"):
        st.markdown(
            "- *Which valve closes when blood leaves the right ventricle?*\n"
            "- *Why is the left ventricle wall thicker than the right?*\n"
            "- *Trace the path of a red blood cell from the body back to the body.*\n"
            "- *Explain the difference between preload and afterload simply.*"
        )


# ---------------------------------------------------------------------------
# PAGE 2: PDF CHAT
# ---------------------------------------------------------------------------


def process_uploads(files) -> None:
    """Run the whole ingestion pipeline with visible progress and honest errors."""
    progress = st.progress(0.0, text="Reading your PDFs\u2026")
    documents = []
    stats = []

    for i, f in enumerate(files):
        name = safe_filename(f.name)
        try:
            pages = extract_text_per_page(f.getvalue(), name)
            docs = build_documents(pages, name)
            documents.extend(docs)
            stats.append({"name": name, "pages": len(pages), "text_pages": len(docs), "chunks": 0})
        except PDFProcessingError as exc:
            st.warning(str(exc))
        progress.progress((i + 1) / len(files), text=f"Reading {name}\u2026")

    if not documents:
        progress.empty()
        st.error(
            "\u274c **No usable text was extracted from any of the uploaded files.** "
            "Upload PDFs that contain selectable text (not scans or photos)."
        )
        return

    progress.progress(1.0, text="Splitting text into chunks\u2026")
    chunks = split_documents(documents)
    if not chunks:
        progress.empty()
        st.error("\u274c The extracted text was too short to create chunks.")
        return

    # Count chunks per file so the student sees exactly what was indexed.
    per_file: dict[str, int] = {}
    for c in chunks:
        src = c.metadata.get("source", "unknown")
        per_file[src] = per_file.get(src, 0) + 1
    for s in stats:
        s["chunks"] = per_file.get(s["name"], 0)

    try:
        with st.spinner(
            "Creating embeddings with all-MiniLM-L6-v2 and building the FAISS index "
            "(the first run downloads the ~90 MB model)…"
        ):
            store = build_vector_store(chunks)
    except Exception as exc:  # noqa: BLE001 - surface model/download problems honestly
        progress.empty()
        st.error(f"\u274c **Could not build the vector index:** {exc}")
        return

    progress.empty()
    st.session_state.vector_store = store
    st.session_state.doc_stats = stats
    st.session_state.messages = []  # new documents => start a fresh conversation
    st.session_state.bundled_loaded = False  # the student's own files replaced the samples

    total_chunks = sum(s["chunks"] for s in stats)
    st.success(
        f"\u2705 Indexed **{len(stats)}** document(s), **{total_chunks}** chunks. Ask a question below!"
    )
    for s in stats:
        st.markdown(
            f"- **{s['name']}** — {s['pages']} page(s), "
            f"{s['text_pages']} with text, {s['chunks']} chunk(s)"
        )


def _history_pairs(messages: list[dict]) -> list[tuple[str, str]]:
    """Turn [{role, content}, ...] into [(student, tutor), ...] pairs for the prompt."""
    pairs, pending = [], None
    for m in messages:
        if m["role"] == "user":
            pending = m["content"]
        elif m["role"] == "assistant" and pending is not None:
            pairs.append((pending, m["content"]))
            pending = None
    return pairs


def render_pdf_chat() -> None:
    st.header("\U0001f4c4 PDF Chat")
    st.caption("Upload your study materials, then ask questions — answers cite the file and page.")
    if st.session_state.bundled_loaded:
        st.caption(
            "📚 Sample study material was loaded automatically — your session starts ready. "
            "Upload your own PDFs below to replace it."
        )

    uploads = st.file_uploader(
        "Upload heart study PDFs (max 5 files, 20 MB each)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Text-based PDFs only — scanned/image PDFs cannot be read without OCR.",
    )

    if uploads:
        accepted, errors = validate_pdf_uploads(uploads)
        for message in errors:
            st.error(message)

        if accepted:
            names_indexed = {s["name"] for s in st.session_state.doc_stats}
            names_uploaded = {safe_filename(f.name) for f in accepted}
            needs_processing = names_uploaded != names_indexed or st.session_state.vector_store is None

            if needs_processing:
                if st.button("\u2699\ufe0f Process documents", type="primary"):
                    process_uploads(accepted)
            else:
                st.caption("\u2705 These files are already indexed. Upload different files to replace the index.")

    st.divider()

    # ---- Conversation ----
    if st.session_state.messages:
        col_chat, col_clear = st.columns([4, 1])
        with col_clear:
            if st.button("\U0001f9f9 Clear chat"):
                st.session_state.messages = []
                st.rerun()

    for message in st.session_state.messages:
        render_chat_message(message["role"], message["content"])
        if message.get("sources"):
            st.caption("\U0001f4ce " + "  \u00b7  ".join(message["sources"]))
        if message.get("fallback"):
            st.caption("\u2753 Not found in your documents")

    question = st.chat_input("Ask a question about your study materials\u2026")
    if question:
        if not api_key:
            st.warning(
                "\U0001f511 Add GOOGLE_API_KEY to your `.env` file to ask questions. "
                "See the sidebar for instructions."
            )
            st.stop()
        if st.session_state.vector_store is None:
            st.warning("\U0001f4c1 Upload and process at least one PDF first (button above).")
            st.stop()

        # Show the student's message immediately, then compute the answer.
        st.session_state.messages.append({"role": "user", "content": question, "sources": []})
        render_chat_message("user", question)

        with st.spinner("Searching your documents\u2026"):
            result = answer_from_documents(
                question=question,
                vector_store=st.session_state.vector_store,
                history=_history_pairs(st.session_state.messages),
            )

        if result["error"]:
            st.error(result["error"])
            st.session_state.messages.pop()  # drop the user turn so history pairs stay aligned
        else:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["text"],
                    "sources": result["sources"],
                    "fallback": not result["retrieved"],
                }
            )
            st.rerun()  # re-render the full conversation with sources


# ---------------------------------------------------------------------------
# PAGE 3: HEART ANATOMY LEARNING MODE
# ---------------------------------------------------------------------------


def render_heart_anatomy() -> None:
    st.header("\U0001fac0 Heart Anatomy Learning Mode")
    st.info(
        "\U0001f4d6 **General learning mode** — answers here come from Gemini's general "
        "medical knowledge, **not** from your uploaded PDFs. Use *PDF Chat* when you want "
        "answers grounded in your own materials."
    )

    topic = st.selectbox("Choose a topic", HEART_TOPICS, key="anatomy_topic")
    custom = st.text_input(
        "Or type your own question (optional)",
        placeholder="e.g. Why does the left ventricle have a thicker wall?",
    )
    level = st.radio(
        "Explain it at this level", ["Beginner", "Intermediate", "Advanced"], horizontal=True
    )

    with st.expander(f"\U0001f4d6 Quick reference card: {topic} (works offline)"):
        st.markdown(STATIC_LESSONS[topic])

    if st.button("\U0001f50d Explain", type="primary", key="anatomy_go"):
        question = custom.strip() or f"Explain {topic} for a {level.lower()} student."
        if not api_key:
            st.warning(
                "\U0001f511 This mode needs GOOGLE_API_KEY — see the sidebar. "
                "The reference card above works without it."
            )
        else:
            with st.spinner("Preparing an explanation\u2026"):
                result = answer_general(
                    question=question,
                    history=_history_pairs(st.session_state.anatomy_messages),
                    level=level,
                )
            if result["error"]:
                st.error(result["error"])
            else:
                st.session_state.anatomy_messages.append(
                    {"role": "user", "content": question, "sources": []}
                )
                st.session_state.anatomy_messages.append(
                    {
                        "role": "assistant",
                        "content": result["text"],
                        "sources": ["General learning mode (not from your PDFs)"],
                    }
                )
                st.rerun()

    if st.session_state.anatomy_messages:
        st.subheader("Your learning conversation")
        for message in st.session_state.anatomy_messages:
            render_chat_message(message["role"], message["content"])
            if message.get("sources"):
                st.caption("\U0001f4ce " + "  \u00b7  ".join(message["sources"]))
        if st.button("\U0001f9f9 Clear conversation", key="anatomy_clear"):
            st.session_state.anatomy_messages = []
            st.rerun()


# ---------------------------------------------------------------------------
# PAGE 4: QUIZ PRACTICE
# ---------------------------------------------------------------------------


def _option_index(label: str | None, question: dict) -> int | None:
    """Map the selected 'B. option text' label back to its 0-3 index."""
    if label is None:
        return None
    for i, option in enumerate(question["options"]):
        if label == f"{chr(65 + i)}. {option}":
            return i
    return None


def _render_quiz_questions(questions: list[dict]) -> None:
    """Draw one card per question with a 4-option selector."""
    for i, q in enumerate(questions):
        st.markdown(
            f'<div class="quiz-card"><b>Q{i + 1}.</b> {q["question"]}</div>',
            unsafe_allow_html=True,
        )
        labels = [f"{chr(65 + j)}. {option}" for j, option in enumerate(q["options"])]
        st.radio(
            f"Choose an answer for Q{i + 1}",
            labels,
            index=None,
            key=f"quiz_choice_{i}",
            label_visibility="collapsed",
            disabled=st.session_state.quiz_result is not None,
            horizontal=True,
        )
        st.write("")  # spacing between cards


def _render_quiz_results() -> None:
    """Score summary, per-question feedback and the review of wrong answers."""
    result = st.session_state.quiz_result

    st.subheader("Your result")
    m1, m2, m3 = st.columns(3)
    m1.metric("Score", f"{result['correct']} / {result['total']}")
    m2.metric("Percentage", f"{result['percent']:.0f}%")
    m3.metric("Missed", f"{len(result['wrong'])}")
    st.progress(result["percent"] / 100.0)
    st.markdown(f"**{score_message(result['percent'])}**")

    st.subheader("Question-by-question feedback")
    for i, r in enumerate(result["results"], start=1):
        if r["is_correct"]:
            st.markdown(f"**\u2705 Q{i}.** {r['question']}")
        else:
            chosen = (
                f"{chr(65 + r['chosen'])}. {r['options'][r['chosen']]}"
                if r["chosen"] is not None
                else "— not answered"
            )
            correct = f"{chr(65 + r['correct_index'])}. {r['options'][r['correct_index']]}"
            st.markdown(
                f"**\u274c Q{i}.** {r['question']}  \n"
                f"Your answer: {chosen} \u00b7 Correct: **{correct}**"
            )
        st.caption(f"\U0001f4a1 {r['explanation']}")

    st.subheader(f"\U0001f4da Review of your {len(result['wrong'])} incorrect answer(s)")
    st.markdown(format_review(result))


def render_quiz_practice() -> None:
    st.header("\U0001f4dd Quiz Practice")
    st.caption("Multiple-choice questions with instant feedback, a score and a review.")

    has_docs = st.session_state.vector_store is not None
    mode = st.radio(
        "Question source",
        ["From my uploaded PDFs", "General heart anatomy topic"],
        horizontal=True,
        help="PDF mode asks only about facts found in your documents; topic mode uses general knowledge.",
    )
    if mode == "From my uploaded PDFs" and not has_docs:
        st.warning(
            "\U0001f4c1 No documents indexed yet — process a PDF on the **PDF Chat** page first, "
            "or switch to *General heart anatomy topic*."
        )

    topic = st.selectbox("Topic", HEART_TOPICS, key="quiz_topic")
    difficulty = st.radio("Difficulty", ["Beginner", "Intermediate", "Advanced"], horizontal=True)
    count = st.slider("Number of questions", 3, MAX_QUIZ_QUESTIONS, DEFAULT_QUIZ_QUESTIONS)

    if st.button("\U0001f3b2 Generate quiz", type="primary", key="quiz_generate"):
        if not api_key:
            st.warning("\U0001f511 Quizzes need GOOGLE_API_KEY — see the sidebar for setup.")
        elif mode == "From my uploaded PDFs" and not has_docs:
            st.warning("Process at least one PDF first.")
        else:
            context_blocks = None
            if mode == "From my uploaded PDFs":
                with st.spinner("Retrieving relevant material\u2026"):
                    results = retrieve(topic_query(topic), st.session_state.vector_store, k=8)
                if not results:
                    st.error(
                        f"\u274c I couldn't find enough material about **{topic}** in your uploaded "
                        "documents to build a fair quiz. Try another topic or switch to general mode."
                    )
                    st.stop()
                context_blocks = [
                    (
                        f"{doc.metadata.get('source', 'document')}, page {doc.metadata.get('page', '?')}",
                        doc.page_content,
                    )
                    for doc, _score in results
                ]

            with st.spinner(f"Generating {count} {difficulty.lower()} questions\u2026"):
                questions, error = generate_quiz(
                    topic=topic,
                    difficulty=difficulty,
                    num_questions=count,
                    context_blocks=context_blocks,
                )
            if error:
                st.error(error)
            else:
                st.session_state.quiz = questions
                st.session_state.quiz_result = None
                st.session_state.quiz_meta = (
                    f"{topic} \u00b7 {difficulty} \u00b7 {len(questions)} questions \u00b7 "
                    + ("PDF-based" if context_blocks else "general knowledge")
                )
                st.rerun()

    # ---- Active quiz ----
    if st.session_state.quiz:
        st.divider()
        st.subheader(f"\U0001f3af {st.session_state.quiz_meta}")
        _render_quiz_questions(st.session_state.quiz)

        if st.session_state.quiz_result is None:
            unanswered = [
                i + 1
                for i in range(len(st.session_state.quiz))
                if st.session_state.get(f"quiz_choice_{i}") is None
            ]
            if st.button("\u2705 Submit answers", type="primary", key="quiz_submit"):
                if unanswered:
                    st.warning(
                        "Please choose an answer for "
                        + ", ".join(f"Q{i}" for i in unanswered)
                        + " before submitting."
                    )
                else:
                    answers = [
                        _option_index(
                            st.session_state.get(f"quiz_choice_{i}"),
                            st.session_state.quiz[i],
                        )
                        for i in range(len(st.session_state.quiz))
                    ]
                    st.session_state.quiz_result = grade_quiz(st.session_state.quiz, answers)
                    st.rerun()
        else:
            _render_quiz_results()

        if st.button("\U0001f501 New quiz", key="quiz_new"):
            st.session_state.quiz = None
            st.session_state.quiz_result = None
            st.session_state.quiz_meta = ""
            st.rerun()


# ---------------------------------------------------------------------------
# ROUTER
# ---------------------------------------------------------------------------

if page.startswith("\U0001f3e0"):
    render_home()
elif page.startswith("\U0001f4c4"):
    render_pdf_chat()
elif page.startswith("\U0001fac0"):
    render_heart_anatomy()
else:
    render_quiz_practice()

# Small footer on every page
st.divider()
st.caption(
    f"{APP_TITLE} \u00b7 educational use only \u00b7 never a substitute for professional medical "
    "advice. Verify important facts with authoritative sources and your instructors."
)
