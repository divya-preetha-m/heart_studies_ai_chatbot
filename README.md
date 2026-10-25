# 🫀 Heart Studies AI

An AI-powered learning platform for students studying **heart anatomy and cardiovascular
physiology**. Upload your lecture notes as PDFs, ask questions and get grounded answers with
page references, explore general heart topics, and practise with automatically generated
multiple-choice quizzes.

Built with **Streamlit + LangChain + Google Gemini + sentence-transformers + FAISS**.

> ⚠️ **Educational use only.** This app is a study aid, not a medical device. It does not
> diagnose or treat disease. Always verify important information with your textbooks,
> lectures and qualified instructors.

---

## Table of contents

1. [What the app does](#1-what-the-app-does)
2. [How the RAG pipeline works](#2-how-the-rag-pipeline-works)
3. [Project structure](#3-project-structure)
4. [Requirements](#4-requirements)
5. [Setup on Windows (PowerShell)](#5-setup-on-windows-powershell)
6. [Getting a Gemini API key](#6-getting-a-gemini-api-key)
7. [Running the application](#7-running-the-application)
8. [Using the app](#8-using-the-app)
9. [How the files work together](#9-how-the-files-work-together)
10. [Testing](#10-testing)
11. [Security and medical safety](#11-security-and-medical-safety)
12. [Troubleshooting (real error messages)](#12-troubleshooting-real-error-messages)
13. [Deploying to Streamlit Community Cloud](#13-deploying-to-streamlit-community-cloud)

---

## 1. What the app does

| Page | What you can do |
|---|---|
| 🏠 **Home** | Explains the app and the RAG pipeline step by step. |
| 📄 **PDF Chat** | Upload 1–5 PDFs (max 20 MB each), index them, then ask questions. Answers cite **file name + page number**. If your documents don't cover the question, the bot honestly says so instead of guessing. |
| 🫀 **Heart Anatomy** | General learning mode: chambers, valves, blood flow, pulmonary/systemic circulation, major vessels, conduction system and basic physiology — clearly labelled as *not* from your PDFs. Includes offline quick-reference cards. |
| 📝 **Quiz Practice** | Generate 3–10 multiple-choice questions (4 options each) from **your PDFs** or from a **general topic**, at Beginner / Intermediate / Advanced level. Pick answers, submit, see your score, percentage, per-question explanations and a review of the ones you missed. |

---

## 2. How the RAG pipeline works

**RAG = Retrieval-Augmented Generation.** Instead of letting the LLM guess from memory, we
first *retrieve* the relevant parts of *your* documents and make the model answer from them.

```
PDF upload
   │
   ▼
① EXTRACT    pypdf pulls text from every page          (rag.py: extract_text_per_page)
   │
   ▼
② CHUNK      LangChain splitter cuts text into         (rag.py: split_documents)
             ~1000-character chunks with 150-char overlap
   │
   ▼
③ EMBED      all-MiniLM-L6-v2 converts each chunk      (rag.py: get_embeddings)
             into 384 numbers ("embedding") that
             capture its meaning
   │
   ▼
④ INDEX      Embeddings go into a FAISS index           (rag.py: FAISSStore)
             (in-memory, per browser session)
   │
   ▼
⑤ RETRIEVE   Your question is embedded the same way;    (rag.py: retrieve)
             FAISS returns the 4 closest chunks.
             Chunks below cosine similarity 0.30 are
             discarded — that's the "relevance gate".
   │
   ▼
⑥ GENERATE   Gemini answers using ONLY those chunks     (rag.py: answer_from_documents)
             + short conversation history. If nothing
             passed the gate, it replies with the fixed
             fallback sentence instead of inventing
             facts. Sources are listed from the actual
             retrieved chunks only.
```

Key safety property: **retrieved PDF text is treated as untrusted reference material, never
as instructions** (see the system prompt in `rag.py`), so a malicious PDF cannot inject
prompts into the tutor.

---

## 3. Project structure

```
heart-studies-chatbot/            (this folder)
├── app.py                        Streamlit UI + navigation (the only file you launch)
├── rag.py                        RAG pipeline: extraction, chunking, embeddings,
│                                 FAISS index, retrieval, grounded answers, general mode
├── quiz.py                       Quiz generation, JSON validation, scoring, review
├── utils.py                      Config, limits, upload validation, emergency detection,
│                                 LLM factory + safe invocation, CSS theme, chat bubbles
├── requirements.txt              Exact verified dependency versions
├── .env.example                  Template for your API key (safe to commit — placeholder only)
├── .gitignore                    Excludes .env, .venv, caches, indexes, logs
├── README.md                     This file
├── sample_pdfs/
│   └── sample_heart_anatomy.pdf  Small sample document to try the app immediately
├── make_sample_pdf.py            Dev utility that (re)generates the sample PDF
└── test_pipeline.py              Dev utility: 56 end-to-end checks of the whole pipeline
```

> **Note:** `.env` (your real key) exists locally and is git-ignored. `.env.example` only
> ever contains the placeholder `your-gemini-api-key-here`.

---

## 4. Requirements

- **Python 3.11, 3.12 or 3.13** — ⚠️ **Python 3.14/3.15 do not work yet**: PyTorch and
  faiss-cpu do not publish wheels for them. Check with `python --version`.
- ~1.5 GB free disk (PyTorch CPU + the embedding model).
- Internet on the first run (downloads the ~90 MB `all-MiniLM-L6-v2` model, then it is cached).
- A free Google Gemini API key.

---

## 5. Setup on Windows (PowerShell)

Open **PowerShell**, go to this project folder, and run each block in order.

### 5.1 (Only if you have Python 3.14+ or no Python) install Python 3.13

```powershell
# See what you have (e.g. "Python 3.15.0" or "Python 3.13.15")
python --version

# If it is NOT 3.11–3.13, install 3.13 for your user account (no admin rights needed):
# download the installer from https://www.python.org/downloads/ and in the first
# installer screen UNCHECK "Add python.exe to PATH", then click Install.
# It lands in %LOCALAPPDATA%\Programs\Python\Python313
```

### 5.2 Create a virtual environment and install dependencies

```powershell
# Pick the right interpreter:
#   3.13 ->  . "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" is not needed;
#            just call it directly if plain "python" is 3.14+:
$PY = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
if (-not (Test-Path $PY)) { $PY = "python" }   # fall back to whatever is on PATH

# Create the virtual environment
& $PY -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install the verified dependency set (takes a few minutes;
# PyTorch is ~124 MB)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation with `running scripts is disabled on this system`:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 5.3 Create your `.env` file

```powershell
Copy-Item .env.example .env
notepad .env     # paste your real key after GOOGLE_API_KEY=  and save
```

---

## 6. Getting a Gemini API key

1. Open <https://aistudio.google.com/apikey> and sign in with a Google account.
2. Click **Create API key** and copy it.
3. Paste it into your `.env` file: `GOOGLE_API_KEY=your-real-key-here`
   (no quotes, no spaces around the `=`).
4. Restart the app after changing `.env`.

The sidebar shows **✅ Gemini API key loaded · model ready** when everything is configured.
The key is read only from the environment — it is never hard-coded, never logged and never
sent anywhere except to Google's API.

---

## 7. Running the application

With the virtual environment activated (`.venv\Scripts\Activate.ps1`):

```powershell
streamlit run app.py
```

Streamlit prints a local URL, usually <http://localhost:8765>. Ctrl+C in the terminal stops
the server.

---

## 8. Using the app

1. **PDF Chat** → drop one or more text-based PDFs into the uploader → click
   **⚙️ Process documents**. You'll see a progress bar for extraction, chunking, embedding
   and indexing, then a per-file summary (pages, chunks).
2. Type a question in the chat box at the bottom. Answers show 📎 source references
   like `lecture3.pdf, page 4`. If nothing relevant was found you'll see the fixed
   fallback message plus a "❓ Not found in your documents" tag.
3. **Heart Anatomy** → pick a topic and a level, click **🔍 Explain**. The quick-reference
   cards work even without an API key.
4. **Quiz Practice** → choose *From my uploaded PDFs* (requires indexing first) or
   *General heart anatomy topic*, pick difficulty and question count, **🎲 Generate quiz**,
   answer every question, **✅ Submit answers** → score, explanations and a review of
   incorrect answers.

**Limits** (defined in `utils.py`): max 5 PDFs per batch, 20 MB per file, 3–10 questions
per quiz, last 4 conversation turns used as context, minimum cosine similarity 0.30 for a
chunk to count as relevant.

---

## 9. How the files work together

```
              ┌────────────┐
              │   app.py   │  Streamlit pages + routing
              └─────┬──────┘
       ┌────────────┼───────────────┐
       ▼            ▼               ▼
   rag.py        quiz.py        utils.py
 (pipeline)   (quiz engine)  (shared toolbox)
       │            │               ▲
       └────────────┴───────────────┘
              every module imports utils
```

- **`utils.py`** has no dependencies on the other modules. It owns every constant
  (chunk size, limits, fallback sentence), the API-key handling, upload validation,
  emergency-symptom detection, the `ChatGoogleGenerativeAI` factory, the never-raise
  `safe_invoke()` wrapper, and the CSS theme.
- **`rag.py`** implements the seven pipeline steps described in section 2 and both answer
  modes (`answer_from_documents` = grounded in PDFs, `answer_general` = general learning).
- **`quiz.py`** builds the quiz prompt, parses Gemini's JSON reply, **validates every
  question** (exactly 4 unique options, exactly one correct index, non-empty explanation —
  malformed items are dropped, never shown), grades answers and formats the review.
- **`app.py`** is the only executable entry point. It keeps per-page conversation history
  in `st.session_state`, and the FAISS index lives in the session too — it disappears when
  you close the browser tab (by design: no server-side storage of your documents).

---

## 10. Testing

`test_pipeline.py` runs **56 end-to-end checks** without needing an API key: PDF
extraction, chunking, embedding generation, FAISS search quality, the relevance gate,
fallback behaviour, quiz JSON validation, grading, upload validation and emergency
detection.

```powershell
python test_pipeline.py
```

Expected output ends with: `RESULT: 56/56 passed`.

To rebuild the sample PDF:

```powershell
python make_sample_pdf.py
```

> The chat, learning mode and quiz *generation* need a real API key, so test those in the
> browser. Everything else is covered by the script.

---

## 11. Security and medical safety

- **API key**: only read from `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) via `.env`, which is
  git-ignored. `.env.example` contains a placeholder only. Never commit `.env`.
- **Uploads**: type-checked (`.pdf` only), size-capped (20 MB), count-capped (5). Corrupted,
  password-protected, scanned/image-only and empty PDFs produce clear error messages, never
  crashes. Uploaded text is treated as **data, not instructions** — the system prompt tells
  Gemini to ignore any instructions found inside PDFs, and no code from PDFs is ever executed.
- **Medical safety**: an offline keyword screen detects possible emergency symptoms
  (severe chest pain, collapse, difficulty breathing, stroke, etc.) and shows a
  "call emergency services now" banner *before* any AI answer. The app never diagnoses,
  never recommends medication or treatment, and refuses to answer when the uploaded
  materials don't support an answer.

---

## 12. Troubleshooting (real error messages)

### 🔑 "No Gemini API key found" (red box in the sidebar)

Chat, learning mode and quizzes need a key; PDF upload/indexing still works without one.
Fix: copy `.env.example` → `.env`, set `GOOGLE_API_KEY=...` (section 6), **restart** the
app. Check for trailing spaces or stray quotes in `.env`.

### 🐍 `error: externally-managed-environment` or packages install into the wrong Python

You created the venv with one Python version and are running another. Recreate it
(section 5.2) and make sure `python --version` prints 3.11–3.13 **inside the activated
venv**.

### 📦 `Could not find a version that satisfies the requirement torch` (or `faiss-cpu`)

Your Python is too new (3.14/3.15 have no wheels for these packages). Install Python
3.13 (section 5.1) and recreate the venv with it.

### 🪟 Windows: `WinError 4551` / "This file does not have an app associated with it" /
`ModuleNotFoundError` for `torch` or `regex` right after install

This is **Smart App Control** or antivirus blocking unsigned DLLs — not a code problem.
Check `Settings → Privacy & security → Windows security → App & browser control →
Smart App Control`. If it is enforcing, torch cannot load until it is set to "Off" or
"Evaluation" (turning it off is a one-way switch on some Windows builds — decide whether
that tradeoff is acceptable to you). Corporate/managed devices may not allow it.

### 🔇 Log full of `ModuleNotFoundError: No module named 'torchvision'`

**This is harmless noise, not an error.** Streamlit's auto-reload file watcher probes every
`transformers` submodule; the *image-processing* ones want `torchvision` (an
image library this text-only app never uses). Streamlit catches the exception itself and
continues. You can ignore it completely. (If it bothers you: `mkdir .streamlit`, create
`.streamlit/config.toml` with `[server]` / `fileWatcherType = "none"` — that also disables
auto-reload on code changes.)

### 📄 "**…has no extractable text. It is probably a scanned or image-only PDF**"

The PDF is a photo/scan. This app has no OCR. Re-export the document with selectable text
(most word processors and reference managers can do this) or use the original digital file.

### 🔐 "**…is password-protected**"

Remove the password in a PDF tool and upload again.

### ⚠️ "Gemini rate limit or quota reached" / HTTP 429

Free API keys have per-minute and per-day limits. Wait a minute and retry; check usage at
<https://aistudio.google.com/usage>. The app already retries twice with backoff before
showing this message, and because Gemini counts quotas **per model**, it automatically
retries on a second model first (see the fallback note in the 404 section below) — so you
usually never see this.

### 🚨 "Gemini returned an empty response"

Gemini replied, but with no text. This happened with the newer `gemini-3.x` models, which
send their reply as structured content blocks instead of a plain string; the app now
flattens every reply format and retries once before showing this. If it still appears, the
service is under heavy load — wait a moment and press the button again.

### 🛡️ `404 NOT_FOUND … 'models/gemini-X.Y-flash' is no longer available to new users`

Google retires older Gemini models periodically. The app's built-in default is kept
current (see `DEFAULT_GEMINI_MODEL` in `utils.py`), but if you pinned a model with
`GEMINI_MODEL` in `.env` and it disappears, either remove that line to use the default
or point it at a model your key can access. Transient `503 UNAVAILABLE` / "high demand"
errors on a valid model are Google-side capacity issues — wait a minute and retry.

**Automatic model fallback:** when the default model is overloaded, times out, returns
nothing or hits its quota, the app silently makes one attempt on `FALLBACK_GEMINI_MODEL`
(same file) before giving you an error, because Gemini quotas are counted per model. This
fallback is skipped when you set `GEMINI_MODEL` yourself — your choice is respected.

### ⚠️ "Gemini rejected the API key" / HTTP 401

The key in `.env` is wrong, truncated, or belongs to a different Google project. Copy it
again from <https://aistudio.google.com/apikey>, save, restart.

### 🌐 "Network problem while reaching Gemini"

Check internet connectivity and any VPN/proxy. The embedding model and Gemini both need
outbound HTTPS.

### ❓ Answers always say "I couldn't find sufficient information…"

Either the indexed PDFs really don't contain the answer (the honest behaviour!), or the
relevance gate (0.30 cosine) is filtering everything — common for very short questions about
long unrelated documents. Try phrasing the question with terms that actually appear in the
document, or index a more relevant PDF.

### 🐢 First PDF upload is very slow

The first run downloads the ~90 MB `all-MiniLM-L6-v2` model from Hugging Face (cached
afterwards in `~/.cache/huggingface`). Embedding then runs on CPU — a few seconds per page
is normal. Subsequent uploads are much faster.

### 🖥️ Port 8765 already in use

```powershell
streamlit run app.py --server.port 8787
```

### 🔁 Streamlit says "Session state mismatch" or stale data after edits

Stop the server (Ctrl+C) and run it again. Streamlit reloads `app.py` on save, but changes
in imported modules (`rag.py`, `quiz.py`, `utils.py`) sometimes need a full restart.

---

## 13. Deploying to Streamlit Community Cloud

**Why not Vercel?** Vercel's Python runtime only runs ASGI/WSGI apps (FastAPI, Flask,
Django) as short-lived serverless functions — Streamlit is a long-running server with its
own WebSocket sessions, so it cannot run there. Even after a rewrite, the PyPI `torch`
wheel for Linux is 555 MB and pulls ~3 GB of NVIDIA CUDA libraries, well past Vercel's
function size limits. Community Cloud is built for exactly this stack.

**Steps** (the repo must already be on GitHub):

1. Go to <https://share.streamlit.io> → **New app** → pick this repository, branch `main`,
   main file `app.py`.
2. Open **Advanced settings** and set:
   - **Python version** — `3.13` matches local development, but the cloud default of
     `3.14` also works (verified on a live deploy: `torch` has a 3.14 CPU wheel and
     `faiss-cpu` is an `abi3` build). Only 3.15+ is unsafe — no torch wheels yet.
   - **Secrets** → paste (same names as your `.env`):

     ```toml
     GOOGLE_API_KEY = "paste-your-gemini-key-here"
     # optional — override the default model
     # GEMINI_MODEL = "gemini-3.5-flash"
     ```

     The app reads `.env` when running locally and `st.secrets` when deployed, so no code
     change is needed. Never commit a real key — `.streamlit/secrets.toml` is already in
     `.gitignore`.
3. Click **Deploy**. The first build takes a few minutes (it installs the 196 MB CPU-only
   `torch` wheel), and the first run downloads the ~90 MB embedding model — both cached
   afterwards.

**Keep the CPU-torch lines in `requirements.txt`.** The `--extra-index-url` line and the
`torch==2.14.1+cpu` pin are what make the app fit Community Cloud's 1 GB resource limit;
plain `torch` would install the CUDA build and fail the deploy.

### 📦 Build log says `error: No solution found when resolving dependencies` (then succeeds)

**This is harmless.** Community Cloud first tries `uv`, whose default "first index" safety
rule sees `requests` on `download.pytorch.org/whl/cpu` (it only carries an old version) and
gives up. The platform then automatically retries with plain **pip**, which consults both
indexes correctly — look a few lines further down for
`🐍 Python dependencies were installed … using pip` followed by `📦 Processed dependencies!`
and `Uvicorn server started`. If those are present, the build succeeded.

**Free-tier limits to know:** ~1 GB RAM per app, 3 apps per account, and apps sleep after
a period of inactivity (the first request after waking takes ~30 seconds).

---

## License / attribution

Built as an educational project. The sample PDF in `sample_pdfs/` was generated by
`make_sample_pdf.py` and contains original summary text of standard heart anatomy facts.
