"""
quiz.py — AI quiz generation, validation, scoring and review.

PURPOSE (in simple English)
---------------------------
This module turns the student's study material (or a chosen heart topic) into a
multiple-choice quiz, checks their answers, and produces a friendly review.

    generate_quiz()  asks Gemini for a JSON quiz, then VALIDATES every question
                     (exactly 4 options, exactly one correct answer, non-empty
                     text) and drops anything malformed — students should never
                     see a broken or ambiguous question.
    grade_quiz()     compares the student's selected options with the correct
                     ones and returns score, percentage and a review list of the
                     questions they got wrong.

TWO SOURCES OF QUESTIONS
------------------------
* PDF mode      — questions are grounded in chunks retrieved from the student's
                  uploaded documents (via rag.retrieve), so facts trace back to
                  their study material.
* Topic mode    — questions come from Gemini's general heart-anatomy knowledge,
                  clearly labelled as general learning.

FILES: app.py calls generate_quiz()/grade_quiz(); rag.py provides the LLM,
retrieval and shared prompts; utils.py supplies limits and error helpers.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from utils import (
    DIFFICULTY_LEVELS,
    DISCLAIMER,
    MAX_QUIZ_QUESTIONS,
    invoke_llm_prompt,
)

# ---------------------------------------------------------------------------
# PROMPT CONSTRUCTION
# ---------------------------------------------------------------------------

QUIZ_SYSTEM_RULES = f"""You are a quiz writer for a heart anatomy and cardiovascular physiology \
study app for students. You produce multiple-choice questions that are:

- Factually correct and unambiguous: one option is clearly the single best answer,
  the other three are plausible but clearly wrong (no "all of the above", no \
  two options that could both be right),
- Supported ONLY by the study material provided below (PDF mode) or by standard, \
  well-established heart anatomy and physiology facts (topic mode) — never invent \
  obscure or contested claims,
- Educational: they test understanding, not trick wording,
- Free of diagnosis, treatment recommendations or patient-specific advice.

Format rules (critical):
- Reply with ONLY a JSON array. No markdown fences, no commentary before or after.
- Each element is an object with exactly these keys:
  "question" (string), "options" (array of EXACTLY 4 strings),
  "correct_index" (integer 0-3, the position of the correct option),
  "explanation" (string: 1-2 short sentences explaining WHY the correct answer is right).

{DISCLAIMER}
"""


def build_quiz_prompt(
    topic: str,
    difficulty: str,
    num_questions: int,
    context_blocks: Optional[list[tuple[str, str]]] = None,
) -> str:
    """Assemble the quiz request sent to Gemini.

    `context_blocks` is a list of (label, text) retrieved chunks — when present
    we are in PDF mode and the questions must come from that material.
    """
    level = DIFFICULTY_LEVELS.get(difficulty, DIFFICULTY_LEVELS["Beginner"])
    num_questions = max(1, min(num_questions, MAX_QUIZ_QUESTIONS))

    lines = [
        QUIZ_SYSTEM_RULES,
        f"Create a multiple-choice quiz about: {topic}",
        f"Number of questions: {num_questions}",
        f"Student level: {difficulty} ({level})",
    ]

    if context_blocks:
        joined = "\n\n".join(
            f"[{label}]\n{text}" for label, text in context_blocks
        )
        lines.append(
            "MODE: PDF-BASED. Ask questions ONLY about facts that appear in the "
            "STUDY MATERIAL below, and make sure the correct answers can be found there.\n\n"
            f"STUDY MATERIAL:\n----------------\n{joined}\n----------------"
        )
    else:
        lines.append(
            "MODE: GENERAL TOPIC. Use standard, well-established heart anatomy and "
            "physiology knowledge."
        )

    lines.append("Return ONLY the JSON array now.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PARSING + VALIDATION — never show a broken question to a student.
# ---------------------------------------------------------------------------


def _extract_json_array(raw: str) -> Optional[list]:
    """Find the first JSON array in the model's reply (tolerates ```json fences)."""
    text = raw.strip()
    # Strip markdown code fences if the model added them anyway.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None


def validate_question(item: dict) -> Optional[dict]:
    """Return a clean question dict, or None if this item is unusable."""
    if not isinstance(item, dict):
        return None

    question = str(item.get("question", "")).strip()
    options = item.get("options")
    correct = item.get("correct_index")

    if not question or len(question) < 8:
        return None
    if not isinstance(options, list) or len(options) != 4:
        return None
    options = [str(o).strip() for o in options]
    if any(not o for o in options) or len(set(o.lower() for o in options)) != 4:
        return None  # empty or duplicated options are ambiguous
    if isinstance(correct, bool) or not isinstance(correct, int) or not (0 <= correct <= 3):
        return None

    explanation = str(item.get("explanation", "")).strip()
    if not explanation:
        return None  # we require an explanation for the feedback screen

    return {
        "question": question,
        "options": options,
        "correct_index": correct,
        "explanation": explanation,
    }


def parse_quiz(raw: str, max_questions: int) -> list[dict]:
    """Parse Gemini's reply into a list of validated question dicts (may be empty)."""
    items = _extract_json_array(raw)
    if items is None:
        return []
    questions = []
    for item in items:
        clean = validate_question(item)
        if clean:
            questions.append(clean)
        if len(questions) >= max_questions:
            break
    return questions


# ---------------------------------------------------------------------------
# GENERATION
# ---------------------------------------------------------------------------


def generate_quiz(
    topic: str,
    difficulty: str,
    num_questions: int,
    context_blocks: Optional[list[tuple[str, str]]] = None,
) -> tuple[list[dict], Optional[str]]:
    """Generate a validated quiz.

    Returns (questions, error):
      * success           -> ([q1, q2, ...], None)
      * Gemini/API error  -> ([], friendly error message)
      * bad output twice  -> ([], friendly error message)

    A retry with a stricter instruction is attempted once if the first reply
    is not valid JSON.
    """
    num_questions = max(1, min(int(num_questions), MAX_QUIZ_QUESTIONS))
    prompt = build_quiz_prompt(topic, difficulty, num_questions, context_blocks)

    # temperature 0.6: a little creativity keeps quizzes interesting
    raw, error = invoke_llm_prompt(prompt, temperature=0.6)
    if error:
        return [], error

    questions = parse_quiz(raw, num_questions)
    if questions:
        return questions, None

    # One corrective retry: sometimes the model adds prose around the JSON.
    retry_prompt = (
        f"{prompt}\n\nIMPORTANT: Your previous reply could not be parsed. "
        "Reply with ONLY a valid JSON array of objects, no markdown, no text."
    )
    raw, error = invoke_llm_prompt(retry_prompt, temperature=0.6)
    if error:
        return [], error
    questions = parse_quiz(raw, num_questions)
    if questions:
        return questions, None

    return [], (
        "\U0001f9d1\u200d\U0001f4bb **I could not build a valid quiz from the model's reply.** "
        "Please try again — or pick a different topic or difficulty."
    )


# ---------------------------------------------------------------------------
# GRADING AND REVIEW
# ---------------------------------------------------------------------------


def grade_quiz(
    questions: list[dict],
    answers: list[Optional[int]],
) -> dict:
    """Score the quiz and build the review of wrong answers.

    `answers[i]` is the option index the student picked (or None if skipped).

    Returns:
      {
        "total": int, "correct": int, "percent": float,
        "results": [ {"question", "options", "correct_index", "chosen",
                      "is_correct", "explanation"}, ... ],
        "wrong": [same dicts, wrong ones only],
      }
    """
    results = []
    for i, q in enumerate(questions):
        chosen = answers[i] if i < len(answers) else None
        is_correct = chosen == q["correct_index"]
        results.append(
            {
                "question": q["question"],
                "options": q["options"],
                "correct_index": q["correct_index"],
                "chosen": chosen,
                "is_correct": is_correct,
                "explanation": q["explanation"],
            }
        )

    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    percent = (correct / total * 100) if total else 0.0
    return {
        "total": total,
        "correct": correct,
        "percent": percent,
        "results": results,
        "wrong": [r for r in results if not r["is_correct"]],
    }


def score_message(percent: float) -> str:
    """A short, encouraging verdict for the score screen (no medical claims)."""
    if percent == 100:
        return "\U0001f31f Perfect score — excellent recall!"
    if percent >= 80:
        return "\U0001f44d Strong result — you know this material well."
    if percent >= 60:
        return "\U0001f4a1 Good foundation — review the missed questions below."
    if percent >= 40:
        return "\U0001f501 Keep practising — re-read the relevant sections and try again."
    return "\U0001f4d6 This topic needs more study time — revisit your material, then retry."


def format_review(result: dict) -> str:
    """Render the 'review your wrong answers' section as simple Markdown."""
    if not result["wrong"]:
        return "No incorrect answers to review — well done!"
    lines = []
    for i, r in enumerate(result["wrong"], start=1):
        chosen = (
            f"{chr(65 + r['chosen'])}. {r['options'][r['chosen']]}"
            if r["chosen"] is not None
            else "Not answered"
        )
        correct = f"{chr(65 + r['correct_index'])}. {r['options'][r['correct_index']]}"
        lines += [
            f"**{i}. {r['question']}**",
            f"- Your answer: {chosen} \u274c",
            f"- Correct answer: {correct} \u2705",
            f"- Why: {r['explanation']}",
            "",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------


def topic_query(topic: str) -> str:
    """A good retrieval query for 'quiz me about <topic>' in PDF mode."""
    return f"Key facts about {topic}"
