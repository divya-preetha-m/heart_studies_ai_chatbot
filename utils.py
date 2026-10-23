"""
utils.py — shared configuration, validation and reusable helpers.

PURPOSE (in simple English)
---------------------------
Every module in this project needs the same small pieces of behaviour:
  * where to find the Gemini API key,
  * which limits to enforce (file size, number of files, quiz size),
  * how to validate what the student uploads,
  * how to recognise emergency-symptom messages,
  * how to create the chat model, and how to call it safely,
  * how the app looks (CSS) and how chat bubbles are drawn.

Keeping all of that in ONE file means app.py, rag.py and quiz.py stay short
and focused on their own jobs.
"""

from __future__ import annotations

import html
import os
import re
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv

# Load variables from the .env file into os.environ (once, at import time).
load_dotenv()

# ---------------------------------------------------------------------------
# 1. CONFIGURATION — every "magic number" in the app lives here.
# ---------------------------------------------------------------------------

APP_TITLE = "Heart Studies AI"
APP_ICON = "\U0001fac0"  # 🫀 anatomical heart
APP_TAGLINE = "Your AI study buddy for heart anatomy & cardiovascular physiology"

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Upload safety limits
MAX_FILES_PER_BATCH = 5
MAX_PDF_SIZE_MB = 20

# Chunking settings (how we cut long PDF text into pieces for retrieval)
CHUNK_SIZE = 1000        # characters per chunk (~150-250 words)
CHUNK_OVERLAP = 150      # characters repeated between chunks so meaning is not cut in half
RETRIEVAL_K = 4          # how many chunks we retrieve per question

# Minimum similarity (cosine, -1..1) for a retrieved chunk to count as
# "relevant enough" to answer from. Below this, the bot refuses to answer.
MIN_RELEVANCE = 0.30

# Quiz limits
MAX_QUIZ_QUESTIONS = 10
DEFAULT_QUIZ_QUESTIONS = 5

# How many previous conversation turns the model sees as context
HISTORY_TURNS = 4

# The exact fallback sentence the chatbot must use when documents do not
# contain the answer (required by the project spec).
FALLBACK_MESSAGE = (
    "I couldn't find sufficient information in your uploaded study materials "
    "to answer this question reliably."
)

EMERGENCY_ADVISORY = (
    "\u26a0\ufe0f **Important — this may be an emergency.** If you or someone else "
    "has symptoms such as severe chest pain, collapse, or serious difficulty "
    "breathing, **call your local emergency number right now** (911 in the US) or go "
    "to the nearest emergency department. Do not wait for an answer from this app."
)

DISCLAIMER = (
    "Heart Studies AI is an **educational study aid** for students learning heart "
    "anatomy and cardiovascular physiology. It is **not** a medical device and does "
    "not provide diagnosis or treatment advice. Always verify important information "
    "with your textbooks, lectures and qualified instructors."
)

# ---------------------------------------------------------------------------
# 2. HEART ANATOMY TOPICS — used by the learning mode and the quiz page.
# ---------------------------------------------------------------------------

HEART_TOPICS = [
    "Four chambers of the heart",
    "Heart valves and their functions",
    "Blood flow through the heart",
    "Pulmonary and systemic circulation",
    "Major blood vessels of the heart",
    "Cardiac conduction system",
    "Basic cardiovascular physiology",
]

DIFFICULTY_LEVELS = {
    "Beginner": (
        "introductory level — simple everyday words, short sentences, one clear "
        "example per idea; assumes the student is just starting"
    ),
    "Intermediate": (
        "undergraduate level — uses standard medical terminology but still defines "
        "each term; connects structure to function"
    ),
    "Advanced": (
        "exam-preparation level — precise physiological detail, mechanisms, "
        "clinically relevant relationships; no hand-holding"
    ),
}

# Short offline fallback lessons (shown only when no API key is configured,
# so the learning page is still useful without Gemini).
STATIC_LESSONS = {
    "Four chambers of the heart": (
        "The heart has **four chambers** (rooms):\n"
        "- **Right atrium** — receives used (oxygen-poor) blood from the body.\n"
        "- **Right ventricle** — pumps that blood to the lungs.\n"
        "- **Left atrium** — receives fresh (oxygen-rich) blood from the lungs.\n"
        "- **Left ventricle** — the strongest chamber; pumps oxygen-rich blood out "
        "to the whole body.\n\n"
        "Simple analogy: the heart is a double pump — the right side is the "
        "'lung pump', the left side is the 'body pump'."
    ),
    "Heart valves and their functions": (
        "Valves are **one-way doors** that keep blood moving in the right direction:\n"
        "- **Tricuspid valve** — right atrium → right ventricle.\n"
        "- **Pulmonary (semilunar) valve** — right ventricle → pulmonary artery.\n"
        "- **Mitral (bicuspid) valve** — left atrium → left ventricle.\n"
        "- **Aortic (semilunar) valve** — left ventricle → aorta.\n\n"
        "When valves close you hear the 'lub-dub' heart sounds."
    ),
    "Blood flow through the heart": (
        "Follow the blood: **Body → right atrium → tricuspid valve → right "
        "ventricle → pulmonary valve → lungs → left atrium → mitral valve → "
        "left ventricle → aortic valve → aorta → body.**\n\n"
        "Simple analogy: blood always returns to the heart before it is sent out again, "
        "like a train that passes through four stations in order."
    ),
    "Pulmonary and systemic circulation": (
        "- **Pulmonary circulation**: right ventricle → lungs → left atrium. Picks up "
        "oxygen, drops off carbon dioxide.\n"
        "- **Systemic circulation**: left ventricle → body → right atrium. Delivers "
        "oxygen and nutrients, collects waste.\n\n"
        "The two loops run **in series**, like two circuit tracks joined together."
    ),
    "Major blood vessels of the heart": (
        "- **Superior & inferior vena cava** — bring oxygen-poor blood to the right atrium.\n"
        "- **Pulmonary artery** — the only artery carrying oxygen-poor blood, to the lungs.\n"
        "- **Pulmonary veins** — bring oxygen-rich blood to the left atrium.\n"
        "- **Aorta** — the largest artery; distributes oxygen-rich blood to the body.\n"
        "- **Coronary arteries** — sit on the outside of the heart and feed the heart "
        "muscle itself."
    ),
    "Cardiac conduction system": (
        "The heart beats by itself using an electrical system:\n"
        "**SA node** (the natural pacemaker, in the right atrium) → **AV node** "
        "(slows the signal) → **Bundle of His** → **left & right bundle branches** → "
        "**Purkinje fibres** (spread the signal through the ventricles).\n\n"
        "Simple analogy: the SA node is the conductor of an orchestra — it sets the "
        "tempo and the other musicians follow."
    ),
    "Basic cardiovascular physiology": (
        "- **Cardiac output** = heart rate × stroke volume (blood pumped per minute).\n"
        "- **Blood pressure** = cardiac output × peripheral resistance.\n"
        "- **Blood vessels**: arteries carry blood away from the heart, veins bring it "
        "back, capillaries are the thin exchange sites.\n"
        "- Nerves and hormones adjust heart rate and vessel width to meet the body's needs."
    ),
}

# ---------------------------------------------------------------------------
# 3. ENVIRONMENT / API KEY
# ---------------------------------------------------------------------------


def load_environment() -> None:
    """Re-load .env (useful if the student edits .env while the app is open)."""
    load_dotenv()


def get_google_api_key() -> Optional[str]:
    """Return the Gemini API key from the environment, or None if missing.

    We accept both GOOGLE_API_KEY (documented in the spec) and GEMINI_API_KEY
    (Google's own alternative name), never printing the value anywhere.
    """
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value and value != "your-gemini-api-key-here":
            return value
    return None


def get_model_name() -> str:
    """Which Gemini model to use — overridable via the GEMINI_MODEL env var."""
    return os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_GEMINI_MODEL


# ---------------------------------------------------------------------------
# 4. UPLOAD VALIDATION
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = (".pdf",)


def validate_pdf_uploads(uploaded_files) -> tuple[list, list]:
    """Check the student's uploads before we touch them.

    Returns (accepted, errors) where `accepted` is the list of safe-to-process
    files and `errors` is a list of human-readable problem descriptions.
    """
    accepted, errors = [], []

    if not uploaded_files:
        return accepted, errors

    if len(uploaded_files) > MAX_FILES_PER_BATCH:
        errors.append(
            f"Too many files ({len(uploaded_files)}). "
            f"Please upload at most {MAX_FILES_PER_BATCH} PDFs at a time."
        )
        uploaded_files = uploaded_files[:MAX_FILES_PER_BATCH]

    for f in uploaded_files:
        name = getattr(f, "name", "unknown file")
        # Prefer the real size attribute; only fall back to reading the bytes
        # when the attribute is genuinely missing (NOT when it is 0 — a
        # 0-byte file must stay 0 so it is rejected as empty).
        size = getattr(f, "size", None)
        if size is None:
            try:
                size = len(f.getvalue())
            except Exception:  # noqa: BLE001 - no bytes available to measure
                size = 0

        if not str(name).lower().endswith(ALLOWED_EXTENSIONS):
            errors.append(f"\u274c **{name}** is not a PDF. Only .pdf files are supported.")
            continue
        if size == 0:
            errors.append(f"\u274c **{name}** is empty (0 bytes).")
            continue
        if size > MAX_PDF_SIZE_MB * 1024 * 1024:
            mb = size / (1024 * 1024)
            errors.append(
                f"\u274c **{name}** is {mb:.1f} MB — larger than the "
                f"{MAX_PDF_SIZE_MB} MB limit."
            )
            continue
        accepted.append(f)

    return accepted, errors


# ---------------------------------------------------------------------------
# 5. MEDICAL SAFETY — emergency symptom detection (keyword based, always on).
# ---------------------------------------------------------------------------

_EMERGENCY_PATTERNS = [
    r"chest (pain|pressure|tightness|crush)",
    r"(can'?t|cannot|can not|hard to|difficulty|trouble|struggling to)\s+breath",
    r"short(ness)? of breath",
    r"heart attack",
    r"cardiac arrest",
    r"collapsed?",
    r"passed? out|fainted|losing consciousness|unconscious",
    r"choking",
    r"severe bleeding|coughing up blood",
    r"stroke",
    r"suicid|self.?harm",
    r"poison",
]
_EMERGENCY_RE = re.compile("|".join(_EMERGENCY_PATTERNS), re.IGNORECASE)


def is_emergency_message(text: str) -> bool:
    """True when the student's message describes possible emergency symptoms.

    This is a simple keyword screen (fast, offline, no API call). It exists to
    show a prominent 'call emergency services' banner — it never diagnoses.
    """
    return bool(text and _EMERGENCY_RE.search(text))


# ---------------------------------------------------------------------------
# 6. LLM FACTORY + SAFE INVOCATION
# ---------------------------------------------------------------------------


class MissingAPIKeyError(RuntimeError):
    """Raised when GOOGLE_API_KEY is not configured (handled gracefully downstream)."""


def get_llm(temperature: float = 0.2):
    """Create a LangChain chat model for Google Gemini.

    The API key comes from the environment (never from source code). We import
    the class lazily so the app can still start (and show a helpful message)
    even if the LangChain packages are broken or missing.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = get_google_api_key()
    if not api_key:
        # Fail with our own clear error *before* pydantic raises an opaque one.
        raise MissingAPIKeyError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your Gemini key."
        )
    return ChatGoogleGenerativeAI(
        model=get_model_name(),
        temperature=temperature,
        google_api_key=api_key,  # NB: the field on ChatGoogleGenerativeAI is `google_api_key`
        timeout=60,
        max_retries=2,
    )


def friendly_api_error(exc: Exception) -> str:
    """Turn a raw Gemini/LangChain exception into a message a beginner understands."""
    if isinstance(exc, MissingAPIKeyError):
        return (
            "\U0001f511 **No Gemini API key configured.** Copy `.env.example` to `.env`, "
            "set `GOOGLE_API_KEY=your-key` (get one at https://aistudio.google.com/apikey) "
            "and restart the app."
        )
    text = str(exc)
    lowered = text.lower()
    if "api key" in lowered or "unauthenticated" in lowered or "401" in text or "permission" in lowered:
        return (
            "\U0001f510 **Gemini rejected the API key.** Check that GOOGLE_API_KEY in "
            "your `.env` file is correct, has no extra spaces or quotes, and restart "
            "the app afterwards."
        )
    if "429" in text or "rate limit" in lowered or "quota" in lowered:
        return (
            "\u23f3 **Gemini rate limit or quota reached.** Wait about a minute and try "
            "again. Free API keys have daily limits — check https://aistudio.google.com/usage"
        )
    if "503" in text or "unavailable" in lowered or "overloaded" in lowered:
        return "\U0001f6e1\ufe0f **The Gemini service is temporarily overloaded.** Please try again in a minute."
    if "timeout" in lowered or "timed out" in lowered or "deadline" in lowered:
        return "\u23f1\ufe0f **The request timed out.** Check your internet connection and try again."
    if "connection" in lowered or "network" in lowered or "dns" in lowered:
        return "\U0001f4f6 **Network problem while reaching Gemini.** Check your internet connection and try again."
    # Unknown problems: be honest but do not leak internals.
    return f"\U0001f6a8 **Something went wrong while contacting Gemini.** Details: {text[:300]}"


def invoke_llm_prompt(
    prompt: str, temperature: float = 0.2
) -> tuple[Optional[str], Optional[str]]:
    """Build the LLM and invoke it in ONE protected step.

    Wrapping construction matters: ChatGoogleGenerativeAI itself raises when no
    API key exists, and that must surface as a friendly message, not a crash.

    Returns (text, None) on success or (None, friendly_error) on failure.
    """
    try:
        llm = get_llm(temperature=temperature)
    except Exception as exc:  # noqa: BLE001
        return None, friendly_api_error(exc)
    return safe_invoke(llm, prompt)


def safe_invoke(llm, prompt: str) -> tuple[Optional[str], Optional[str]]:
    """Call the LLM exactly once and never raise.

    Returns (answer_text, None) on success or (None, friendly_error) on failure.
    All chat/quiz features go through this function so that API problems,
    rate limits and network errors are handled gracefully everywhere.
    """
    try:
        response = llm.invoke(prompt)
    except Exception as exc:  # noqa: BLE001 - we translate every failure for the user
        return None, friendly_api_error(exc)

    text = getattr(response, "content", response)
    if isinstance(text, list):  # some LangChain versions return content blocks
        text = "".join(block if isinstance(block, str) else getattr(block, "text", "") for block in text)
    text = str(text).strip()
    if not text:
        return None, "\U0001f6a8 Gemini returned an empty response. Please try again."
    return text, None


# ---------------------------------------------------------------------------
# 7. SMALL HELPERS
# ---------------------------------------------------------------------------


def format_bytes(num_bytes: int) -> str:
    """1234567 -> '1.2 MB' (used in the upload summary)."""
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def safe_filename(name: str) -> str:
    """Strip any path components a browser might sneak into a file name."""
    return os.path.basename(str(name)).replace("\x00", "")


def format_chat_history(history: list[tuple[str, str]], limit: int = HISTORY_TURNS) -> str:
    """Turn [(user, assistant), ...] into a short text block for the prompt.

    Tolerates None/empty history (a brand-new conversation).
    """
    lines = []
    for user_msg, bot_msg in (history or [])[-limit:]:
        lines.append(f"Student: {user_msg}")
        lines.append(f"Tutor: {bot_msg[:600]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 8. UI HELPERS — theme CSS and chat bubbles.
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* ---------- Heart-inspired theme: white background, subtle red accents ---------- */
.stApp { background-color: #ffffff; }
header[data-testid="stHeader"] { background: #ffffff; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fdf4f4 0%, #ffffff 100%);
    border-right: 1px solid #f3d4d4;
}
h1, h2, h3 { color: #b3261e !important; font-family: "Segoe UI", -apple-system, sans-serif; }
p, li, label { color: #333333; }

/* Primary buttons: soft red outline, fills red on hover */
.stButton > button {
    border: 1px solid #e57373; color: #b3261e; background: #ffffff;
    border-radius: 8px; font-weight: 600; transition: all .18s ease;
}
.stButton > button:hover { background: #b3261e; color: #ffffff; border-color: #b3261e; }
.stButton > button:focus { box-shadow: 0 0 0 2px rgba(179,38,30,.25); }

/* Pulsing heart used on the Home page */
@keyframes heartbeat {
    0%, 100% { transform: scale(1); }
    14% { transform: scale(1.18); }
    28% { transform: scale(1); }
    42% { transform: scale(1.12); }
    56% { transform: scale(1); }
}
.heart-pulse { display: inline-block; animation: heartbeat 1.8s infinite ease-in-out; }

/* Feature cards on the Home page */
.info-card {
    border: 1px solid #f3d4d4; border-radius: 12px; padding: 18px 20px;
    background: #fffafa; margin-bottom: 8px;
}
.info-card h4 { color: #b3261e; margin: 0 0 6px 0; font-size: 1.05rem; }
.info-card p { color: #555555; margin: 0; font-size: .92rem; }

/* Chat bubbles */
.chat-row { display: flex; margin: 10px 0; }
.chat-row.user { justify-content: flex-end; }
.chat-bubble {
    max-width: 78%; border-radius: 14px; padding: 12px 16px;
    font-size: .95rem; line-height: 1.55; white-space: pre-wrap; word-wrap: break-word;
}
.chat-row.user .chat-bubble { background: #b3261e; color: #ffffff; border-bottom-right-radius: 4px; }
.chat-row.assistant .chat-bubble {
    background: #f7f7f7; color: #2b2b2b; border: 1px solid #ececec; border-bottom-left-radius: 4px;
}
.chat-meta { font-size: .72rem; opacity: .7; margin-bottom: 4px; font-weight: 600; letter-spacing: .04em; }

/* Quiz question cards */
.quiz-card {
    border: 1px solid #f3d4d4; border-left: 5px solid #e57373;
    border-radius: 10px; padding: 16px 18px; background: #fffdfd; margin: 12px 0;
}
</style>
"""


def inject_theme() -> None:
    """Add the custom theme to the Streamlit page (call once at the top of app.py)."""
    import streamlit as st

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_chat_message(role: str, content: str, meta: str = "") -> None:
    """Draw one chat bubble (user = red, assistant = grey). HTML is escaped."""
    import streamlit as st

    role_class = "user" if role == "user" else "assistant"
    label = "You" if role == "user" else "Heart Studies AI"
    meta_html = f'<div class="chat-meta">{html.escape(meta or label)}</div>'
    bubble = f'<div class="chat-bubble">{html.escape(content)}</div>'
    st.markdown(
        f'<div class="chat-row {role_class}">{meta_html}{bubble}</div>',
        unsafe_allow_html=True,
    )
