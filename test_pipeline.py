"""Temporary verification script (run once, then deleted).

Tests every part of the pipeline that does NOT need a real Gemini API key:
PDF extraction (+ error paths), chunking, embeddings, FAISS retrieval with the
relevance gate, the refusal fallback, LLM error handling, and quiz
parse/grade logic.
"""

from __future__ import annotations

import os
import sys

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


# ---------------------------------------------------------------------------
print("\n[1] Module imports")
import utils
import rag
import quiz

check("utils, rag, quiz all import", True)

# ---------------------------------------------------------------------------
print("\n[2] PDF extraction (happy path)")
with open(os.path.join("sample_pdfs", "sample_heart_anatomy.pdf"), "rb") as fh:
    sample_bytes = fh.read()

pages = rag.extract_text_per_page(sample_bytes, "sample_heart_anatomy.pdf")
check("extracts 2 pages", len(pages) == 2, f"got {len(pages)}")
check("page 1 has substantial text", len(pages[0][1]) > 500, f"got {len(pages[0][1])} chars")
check("page numbering starts at 1", pages[0][0] == 1)
check("valve facts present on page 1", "tricuspid" in pages[0][1].lower())
check("conduction facts present on page 2", "purkinje" in pages[1][1].lower())

print("\n[3] PDF extraction (error paths)")
try:
    rag.extract_text_per_page(b"this is definitely not a PDF", "broken.pdf")
    check("corrupted file raises PDFProcessingError", False)
except rag.PDFProcessingError:
    check("corrupted file raises PDFProcessingError", True)

# Near-empty PDF (simulates an image-only/scanned page): blank page via pypdf
from pypdf import PdfWriter
import io

buf = io.BytesIO()
blank_writer = PdfWriter()
blank_writer.add_blank_page(width=200, height=200)
blank_writer.write(buf)
try:
    rag.extract_text_per_page(buf.getvalue(), "scanned.pdf")
    check("blank/scanned PDF raises PDFProcessingError", False)
except rag.PDFProcessingError as exc:
    check("blank/scanned PDF raises PDFProcessingError", "no extractable text" in str(exc))

print("\n[4] Documents + chunking")
documents = rag.build_documents(pages, "sample_heart_anatomy.pdf")
check("one document per non-empty page", len(documents) == 2, f"got {len(documents)}")
check("metadata keeps source + page", documents[0].metadata == {"source": "sample_heart_anatomy.pdf", "page": 1})

chunks = rag.split_documents(documents)
check("chunking produced multiple chunks", len(chunks) >= 4, f"got {len(chunks)}")
check(
    "all chunks <= chunk_size + slack",
    all(len(c.page_content) <= utils.CHUNK_SIZE + 10 for c in chunks),
)
check("chunk metadata preserved", all(c.metadata.get("source") for c in chunks))

print("\n[5] Embeddings (first call downloads the ~90 MB MiniLM model)")
emb = rag.get_embeddings()
vec = emb.embed_query("the left ventricle pumps blood to the body")
check("embedding dimension is 384", len(vec) == 384, f"got {len(vec)}")
check("embedding is normalised (|v| ~ 1)", abs(sum(x * x for x in vec) ** 0.5 - 1.0) < 0.01)

print("\n[6] FAISS index + retrieval with relevance gate")
store = rag.build_vector_store(chunks)
check("index contains every chunk", store.ntotal == len(chunks), f"{store.ntotal} vs {len(chunks)}")

# 6a. distance -> cosine mapping: querying with a chunk's FULL own text must give ~1.0
own_results = rag.retrieve(chunks[0].page_content, store, k=1)
check("self-query cosine ~ 1.0", own_results and own_results[0][1] > 0.99,
      f"got {own_results[0][1] if own_results else 'no results'}")

# 6a2. a 200-char prefix of the same chunk still scores high but below 1.0
prefix_results = rag.retrieve(chunks[0].page_content[:200], store, k=1)
check("prefix-query cosine sane (0.5-0.99)",
      prefix_results and 0.5 < prefix_results[0][1] < 0.99,
      f"got {prefix_results[0][1] if prefix_results else 'no results'}")

# 6b. relevant query returns the right material
results = rag.retrieve("Which valve stops blood flowing back into the left ventricle?", store, k=4)
check("relevant query returns chunks", len(results) >= 1, f"got {len(results)}")
top_text = results[0][0].page_content.lower() if results else ""
check("top hit mentions valves", "valve" in top_text, f"top text starts: {top_text[:80]!r}")

# 6c. unrelated query must be gated out (this powers the honest refusal)
unrelated = rag.retrieve("What is the recipe for tomato soup?", store, k=4)
check("unrelated query is filtered out (refusal gate)", len(unrelated) == 0,
      f"got {[round(s, 2) for _, s in unrelated]}")

# 6d. sources are formatted and only come from retrieved chunks
labels = rag.collect_sources(results)
check("source labels look like 'file, page N'", labels and all("sample_heart_anatomy.pdf, page" in l for l in labels), str(labels))
check("collect_sources on empty returns []", rag.collect_sources([]) == [])

print("\n[7] Grounded answer + graceful API failure (no key set)")
utils.os.environ.pop("GOOGLE_API_KEY", None)
utils.os.environ.pop("GEMINI_API_KEY", None)

# 7a. no relevant chunks -> fallback WITHOUT calling any model
res = rag.answer_from_documents("What is the recipe for tomato soup?", store)
check("unsupported question returns exact fallback sentence", res["text"] == utils.FALLBACK_MESSAGE, repr(res["text"][:80]))
check("unsupported question has no sources", res["sources"] == [])
check("unsupported question flagged as not retrieved", res["retrieved"] is False)

# 7b. relevant chunks but no API key -> friendly error, no crash, no fabrication
res2 = rag.answer_from_documents("Which valve prevents backflow into the left ventricle?", store)
check("missing API key handled without exception", res2["error"] is not None or bool(res2["text"]), str(res2))
if res2["error"]:
    print(f"        (friendly error: {res2['error'][:100]}...)")

# 7c. emergency keyword detection
check("chest pain flagged as emergency", utils.is_emergency_message("I have severe chest pain right now"))
check("breathing difficulty flagged", utils.is_emergency_message("my dad can't breathe and feels dizzy"))
check("normal question not flagged", not utils.is_emergency_message("how does the mitral valve work?"))

print("\n[8] LLM construction + error translation")
os.environ["GOOGLE_API_KEY"] = "fake-key-for-construction-test"
llm = utils.get_llm()
check("ChatGoogleGenerativeAI constructs with google_api_key", llm is not None)
text, err = utils.safe_invoke(llm, "Say only: OK")
check("safe_invoke never raises (fake key -> friendly error)", err is not None or text is not None)
if err:
    print(f"        (friendly error: {err[:110]}...)")
check("429 translated to rate-limit message",
      "rate limit" in utils.friendly_api_error(Exception("429 RESOURCE_EXHAUSTED quota")).lower())
check("401 translated to API key message",
      "api key" in utils.friendly_api_error(Exception("401 UNAUTHENTICATED")).lower())
del os.environ["GOOGLE_API_KEY"]

print("\n[9] General learning mode (no key -> friendly error, no crash)")
res3 = rag.answer_general("Explain the four chambers of the heart")
check("answer_general returns dict with keys", set(res3) == {"text", "emergency", "error"})
check("missing key handled gracefully", res3["error"] is not None or bool(res3["text"]))

print("\n[10] Quiz parsing + validation")
valid_json = """
[
  {"question": "Which chamber pumps blood to the whole body?",
   "options": ["Right atrium", "Right ventricle", "Left atrium", "Left ventricle"],
   "correct_index": 3,
   "explanation": "The left ventricle has the thickest wall because it pumps to the whole body."},
  {"question": "Which valve lies between the left atrium and left ventricle?",
   "options": ["Tricuspid", "Pulmonary", "Mitral", "Aortic"],
   "correct_index": 2,
   "explanation": "The mitral valve separates the left atrium from the left ventricle."}
]
"""
parsed = quiz.parse_quiz(valid_json, 10)
check("parses two valid questions", len(parsed) == 2, f"got {len(parsed)}")
check("question structure complete", all(
    set(q) == {"question", "options", "correct_index", "explanation"} and len(q["options"]) == 4
    for q in parsed))

# markdown-fenced JSON still parses
fenced = "```json\n" + valid_json + "\n```"
check("parses fenced JSON", len(quiz.parse_quiz(fenced, 10)) == 2)

# malformed items are dropped
bad = """
[{"question": "too vague?", "options": ["a", "b"], "correct_index": 0, "explanation": "x"},
 {"question": "Which chamber pumps oxygenated blood to the aorta?",
  "options": ["Left ventricle", "Left ventricle", "Right ventricle", "Right atrium"],
  "correct_index": 0, "explanation": "dup options"},
 {"question": "Which chamber pumps oxygenated blood to the aorta?",
  "options": ["Left ventricle", "Right ventricle", "Right atrium", "Left atrium"],
  "correct_index": 9, "explanation": "index out of range"},
 {"question": "Which chamber pumps oxygenated blood to the aorta and does so during systole?",
  "options": ["Left ventricle", "Right ventricle", "Right atrium", "Left atrium"],
  "correct_index": 0, "explanation": ""}]
"""
check("drops all malformed questions", quiz.parse_quiz(bad, 10) == [],
      f"got {len(quiz.parse_quiz(bad, 10))}")
check("returns [] for non-JSON text", quiz.parse_quiz("I cannot do that.", 10) == [])
check("respects max_questions", len(quiz.parse_quiz(valid_json, 1)) == 1)

print("\n[11] Quiz grading + review")
questions = parsed
answers = [3, 0]  # Q1 correct, Q2 wrong
result = quiz.grade_quiz(questions, answers)
check("score counts correctly", result["correct"] == 1 and result["total"] == 2)
check("percentage correct", abs(result["percent"] - 50.0) < 0.01)
check("wrong list has 1 item", len(result["wrong"]) == 1)
check("wrong item has correct answer reference", result["wrong"][0]["correct_index"] == 2)
review = quiz.format_review(result)
check("review includes correct answer text", "Mitral" in review)
check("review includes explanation", "mitral valve" in review.lower())
none_result = quiz.grade_quiz(questions, [None, None])
check("unanswered questions score 0", none_result["correct"] == 0)
check("score messages for each band", all(
    quiz.score_message(p) for p in (0, 45, 65, 85, 100)))

print("\n[12] Upload validation")
class FakeUpload:
    def __init__(self, name, size, data=b"x"):
        self.name, self.size = name, size
        self._data = data
    def getvalue(self):
        return self._data

acc, errs = utils.validate_pdf_uploads([
    FakeUpload("notes.pdf", 1000),
    FakeUpload("image.png", 1000),
    FakeUpload("huge.pdf", 30 * 1024 * 1024),
    FakeUpload("empty.pdf", 0),
])
check("accepts one good PDF", len(acc) == 1 and acc[0].name == "notes.pdf")
check("rejects non-PDF", any("not a PDF" in e for e in errs))
check("rejects oversized PDF", any("limit" in e for e in errs))
check("rejects empty PDF", any("empty" in e for e in errs))
check("rejects too many files",
      len(utils.validate_pdf_uploads([FakeUpload(f"f{i}.pdf", 10) for i in range(9)])[1]) == 1)
check("safe_filename strips paths", utils.safe_filename("../../etc/passwd") == "passwd")

# ---------------------------------------------------------------------------
print(f"\n{'='*60}\nRESULT: {PASS} passed, {FAIL} failed\n{'='*60}")
sys.exit(1 if FAIL else 0)
