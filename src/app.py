from __future__ import annotations

import base64
import html
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

import chromadb
import gradio as gr
from dotenv import load_dotenv
from groq import (
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    BadRequestError,
    Groq,
    GroqError,
    PermissionDeniedError,
    RateLimitError,
)
from PIL import Image as PILImage
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
LOCALE_DIR = BASE_DIR / "locales"
TEXT_MODEL = "llama-3.3-70b-versatile"
VISION_MODEL = "llama-3.2-11b-vision-preview"
MAX_IMAGE_BYTES = 5 * 1024 * 1024
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
SUPPORTED_LANGUAGES = ("en", "hi", "te")
SPEECH_LANGUAGE_MAP = {"en": "en-US", "hi": "hi-IN", "te": "te-IN"}
LANGUAGE_NAMES = {"en": "English", "hi": "Hindi", "te": "Telugu"}
DEFAULT_THEME = "light"
HINDI_HINTS = {
    "क",
    "ख",
    "ग",
    "घ",
    "च",
    "छ",
    "ज",
    "झ",
    "ट",
    "ठ",
    "ड",
    "ढ",
    "ण",
    "त",
    "थ",
    "द",
    "ध",
    "न",
    "प",
    "फ",
    "ब",
    "भ",
    "म",
    "य",
    "र",
    "ल",
    "व",
    "श",
    "ष",
    "स",
    "ह",
}

TELUGU_RANGE = (0x0C00, 0x0C7F)

APP_CSS = """
:root {
    --page-bg: #f4f7fb;
    --page-bg-2: #ffffff;
    --surface: rgba(255, 255, 255, 0.84);
    --surface-strong: #ffffff;
    --surface-border: rgba(15, 23, 42, 0.1);
    --text-main: #0f172a;
    --text-muted: #475569;
    --primary: #2563eb;
    --primary-hover: #1d4ed8;
    --accent: #0f766e;
    --danger: #b91c1c;
    --shadow: 0 22px 60px rgba(15, 23, 42, 0.12);
    --shadow-soft: 0 10px 30px rgba(15, 23, 42, 0.08);
    --input-bg: rgba(255, 255, 255, 0.96);
    --input-border: rgba(148, 163, 184, 0.34);
}

html[data-theme='dark'] {
    --page-bg: #07111f;
    --page-bg-2: #0b1628;
    --surface: rgba(15, 23, 42, 0.84);
    --surface-strong: #0f172a;
    --surface-border: rgba(148, 163, 184, 0.18);
    --text-main: #e2e8f0;
    --text-muted: #94a3b8;
    --primary: #60a5fa;
    --primary-hover: #3b82f6;
    --accent: #2dd4bf;
    --danger: #f87171;
    --shadow: 0 22px 60px rgba(2, 6, 23, 0.5);
    --shadow-soft: 0 10px 28px rgba(2, 6, 23, 0.35);
    --input-bg: rgba(15, 23, 42, 0.94);
    --input-border: rgba(148, 163, 184, 0.24);
}

body {
    background:
        radial-gradient(circle at top left, rgba(37, 99, 235, 0.12), transparent 30%),
        radial-gradient(circle at top right, rgba(14, 165, 233, 0.10), transparent 24%),
        linear-gradient(180deg, var(--page-bg), var(--page-bg-2));
    color: var(--text-main);
    transition: background 260ms ease, color 260ms ease;
}

.gradio-container {
    max-width: 1240px !important;
    margin: 0 auto !important;
}

#app-shell {
    gap: 1rem;
    padding: 1rem 0 1.5rem;
}

.hero-panel,
.surface-card,
.output-card,
.status-card {
    background: var(--surface);
    border: 1px solid var(--surface-border);
    border-radius: 24px;
    box-shadow: var(--shadow);
    backdrop-filter: blur(16px);
}

.hero-panel {
    padding: 1.25rem 1.35rem;
}

.surface-card,
.output-card,
.status-card {
    padding: 1rem 1.1rem;
}

.hero-title h1,
.hero-title .prose h1 {
    margin-bottom: 0.35rem;
    font-size: clamp(2rem, 4vw, 3rem) !important;
    line-height: 1.05 !important;
    letter-spacing: -0.03em;
}

.hero-description,
.hero-description .prose,
.hero-description p {
    color: var(--text-muted);
    font-size: 0.98rem;
}

.toolbar-row {
    align-items: center;
    gap: 0.75rem;
}

.toolbar-row > * {
    flex: 1 1 auto;
}

.toolbar-actions {
    display: flex;
    gap: 0.75rem;
    justify-content: flex-end;
    flex-wrap: wrap;
}

.control-row {
    gap: 0.8rem;
    align-items: end;
}

.control-row .gr-button {
    min-height: 3.2rem;
    border-radius: 999px !important;
    box-shadow: var(--shadow-soft);
    transition: transform 180ms ease, box-shadow 180ms ease, background 180ms ease;
}

.control-row .gr-button:hover,
.toolbar-actions .gr-button:hover {
    transform: translateY(-1px);
}

.gr-button.primary,
.gr-button.secondary {
    border-radius: 999px !important;
}

.gr-textbox textarea,
.gr-textbox input,
.gr-dropdown,
.gr-dropdown input,
.gr-image,
.gr-file,
.gr-markdown,
.gr-html {
    transition: background 220ms ease, color 220ms ease, border-color 220ms ease, box-shadow 220ms ease;
}

.gr-textbox textarea,
.gr-textbox input,
.gr-dropdown input,
.gr-dropdown,
.gr-image input,
.gr-textbox {
    border-radius: 18px !important;
}

.gr-textbox textarea,
.gr-textbox input {
    background: var(--input-bg) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--input-border) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
}

.gr-dropdown {
    background: var(--input-bg) !important;
    border: 1px solid var(--input-border) !important;
    color: var(--text-main) !important;
}

.gr-textbox textarea:focus,
.gr-textbox input:focus,
.gr-dropdown:focus-within {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.16);
}

#app-status {
    min-height: 3.5rem;
}

#app-status .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    width: 100%;
    padding: 0.9rem 1rem;
    border-radius: 18px;
    border: 1px solid var(--surface-border);
    background: linear-gradient(180deg, rgba(37, 99, 235, 0.08), rgba(15, 118, 110, 0.05));
    color: var(--text-main);
    box-shadow: var(--shadow-soft);
}

#app-status .status-badge {
    width: 0.7rem;
    height: 0.7rem;
    border-radius: 999px;
    background: var(--primary);
    box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.34);
    animation: pulseDot 1.8s infinite;
}

#app-status .status-badge.processing {
    background: var(--accent);
}

#app-status .status-badge.error {
    background: var(--danger);
}

.answer-markdown,
.context-box,
.message-markdown {
    border-radius: 18px;
}

.answer-markdown {
    padding: 0.25rem 0.25rem 0.75rem;
}

.answer-markdown h1,
.answer-markdown h2,
.answer-markdown h3 {
    letter-spacing: -0.02em;
}

.answer-markdown table {
    width: 100%;
    border-collapse: collapse;
    overflow: hidden;
    border-radius: 14px;
}

.answer-markdown th,
.answer-markdown td {
    border: 1px solid var(--surface-border);
    padding: 0.65rem 0.75rem;
}

.answer-markdown th {
    background: rgba(37, 99, 235, 0.12);
}

.loading-line {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
}

.loading-dots {
    display: inline-flex;
    gap: 0.28rem;
}

.loading-dots span {
    width: 0.45rem;
    height: 0.45rem;
    border-radius: 999px;
    background: var(--primary);
    animation: bounce 1.15s infinite ease-in-out;
}

.loading-dots span:nth-child(2) { animation-delay: 0.12s; }
.loading-dots span:nth-child(3) { animation-delay: 0.24s; }

@keyframes bounce {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.45; }
    40% { transform: translateY(-4px); opacity: 1; }
}

@keyframes pulseDot {
    0% { box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.34); }
    70% { box-shadow: 0 0 0 12px rgba(37, 99, 235, 0); }
    100% { box-shadow: 0 0 0 0 rgba(37, 99, 235, 0); }
}

@media (max-width: 900px) {
    .hero-panel,
    .surface-card,
    .output-card,
    .status-card {
        border-radius: 20px;
    }

    .toolbar-actions {
        justify-content: stretch;
    }

    .toolbar-actions > * {
        flex: 1 1 140px;
    }

    .control-row {
        gap: 0.6rem;
    }
}
"""


def load_locale(code: str) -> dict[str, str]:
    locale_path = LOCALE_DIR / f"{code}.json"
    fallback_path = LOCALE_DIR / "en.json"
    try:
        with locale_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        with fallback_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)


LOCALES = {code: load_locale(code) for code in SUPPORTED_LANGUAGES}


def t(lang: str, key: str, **kwargs: Any) -> str:
    locale = LOCALES.get(lang, LOCALES["en"])
    fallback = LOCALES["en"]
    text = locale.get(key, fallback.get(key, key))
    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception:
            return text
    return text


def status_html(lang: str, state_key: str, detail: str = "") -> str:
    state = html.escape(t(lang, f"status_{state_key}"))
    badge_class = ""
    if state_key == "processing":
        badge_class = "processing"
    elif state_key == "error":
        badge_class = "error"
    detail_html = ""
    if detail:
        detail_html = f"<span style='margin-left: 0.5rem; opacity: 0.85;'>{html.escape(detail)}</span>"
    return (
        f"<div class='status-pill'><span class='status-badge {badge_class}'></span><strong>{state}</strong>{detail_html}</div>"
    )


def language_name(lang: str) -> str:
    return LANGUAGE_NAMES.get(lang, LANGUAGE_NAMES["en"])


def detect_text_language(text: str) -> str:
    sample = text or ""
    if any("\u0C00" <= character <= "\u0C7F" for character in sample):
        return "te"
    if any(character in HINDI_HINTS for character in sample):
        return "hi"
    return "en"


def safe_response_text(response: Any) -> str:
    try:
        choices = getattr(response, "choices", None)
        if not choices:
            return ""
        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is None:
            return ""
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()
        return ""
    except Exception:
        return ""


def translate_text(text: str, target_language: str, source_language: str | None = None) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if target_language not in SUPPORTED_LANGUAGES:
        target_language = "en"

    source_language = source_language or detect_text_language(text)
    if source_language == target_language:
        return text

    prompt = f"""
Translate the following text.

Source language: {language_name(source_language)}
Target language: {language_name(target_language)}

Rules:
- Preserve meaning.
- Preserve names, dataset references, and numbers.
- Return only the translated text.

Text:
{text}
"""

    response = call_groq_chat(
        TEXT_MODEL,
        [{"role": "user", "content": prompt}],
        temperature=0,
    )
    translated = safe_response_text(response)
    return translated or text


def translate_text_safe(text: str, target_language: str, source_language: str | None = None) -> tuple[str, str]:
    try:
        return translate_text(text, target_language, source_language), ""
    except (APITimeoutError, APIConnectionError, APIStatusError, BadRequestError, PermissionDeniedError, RateLimitError, GroqError, Exception):
        return text, t(target_language if target_language in SUPPORTED_LANGUAGES else "en", "message_translation_failed")


def theme_toggle_label(theme_code: str, language_code: str) -> str:
    if theme_code == "dark":
        return t(language_code, "theme_light_mode")
    return t(language_code, "theme_dark_mode")


def apply_theme_state(theme_code: str) -> str:
    return theme_code if theme_code in {"light", "dark"} else DEFAULT_THEME


def theme_data_uri(theme_code: str) -> str:
    return f"data-theme='{apply_theme_state(theme_code)}'"


def compare_mode(question: str, route: str) -> bool:
    lowered = (question or "").lower()
    return route == "both" or any(keyword in lowered for keyword in ("compare", "comparison", "versus", "vs"))


def keyword_route_query(question: str) -> str:
    question = question.lower()

    cricket_keywords = [
        "cricket",
        "batsman",
        "bowler",
        "wicket",
        "odi",
        "test",
        "ipl",
        "virat",
        "dhoni",
        "sachin",
    ]

    olympic_keywords = [
        "olympic",
        "medal",
        "athletics",
        "javelin",
        "wrestling",
        "boxing",
        "neeraj",
        "pv sindhu",
        "mary kom",
    ]

    cricket = any(word in question for word in cricket_keywords)
    olympics = any(word in question for word in olympic_keywords)

    if cricket and olympics:
        return "both"
    if cricket:
        return "cricket"
    if olympics:
        return "olympics"
    return "unknown"


load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    embedding_model = None

try:
    db = chromadb.PersistentClient(path=str(REPO_DIR / "vector_db"))
    cricket_collection = db.get_collection("cricket")
    olympics_collection = db.get_collection("olympics")
except Exception:
    db = None
    cricket_collection = None
    olympics_collection = None


def call_groq_chat(model: str, messages: list[dict[str, Any]], temperature: float, timeout: float = 30.0):
    return client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
        timeout=timeout,
    )


def route_query(question: str) -> str:
    question = (question or "").strip()
    if not question:
        return "unknown"

    prompt = f"""
You are an intelligent routing assistant.

Available datasets:

1. World_Cricketers.xlsx
2. Indian_Olympic_Players.xlsx

Choose ONLY ONE of these words:

cricket
olympics
both
unknown

Rules:

- Cricket questions -> cricket
- Olympic questions -> olympics
- Comparison questions -> both
- Unrelated questions -> unknown

Return ONLY one word.

Question:
{question}
"""

    try:
        response = call_groq_chat(
            TEXT_MODEL,
            [{"role": "user", "content": prompt}],
            temperature=0,
        )
        route = safe_response_text(response).lower()
        if route in {"cricket", "olympics", "both", "unknown"}:
            return route
    except (APITimeoutError, APIConnectionError, APIStatusError, BadRequestError, PermissionDeniedError, RateLimitError, GroqError, Exception):
        pass

    return keyword_route_query(question)


def route_query_with_language(question: str, ui_language: str) -> tuple[str, str, str]:
    detected_language = detect_text_language(question)
    english_question = question
    translation_message = ""

    if detected_language in {"hi", "te"}:
        english_question, translation_message = translate_text_safe(question, "en", detected_language)

    route = route_query(english_question)
    return route, english_question, translation_message


def retrieve(collection, question: str, k: int = 3):
    if collection is None or embedding_model is None:
        return []

    embedding = embedding_model.encode(question).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=k)
    documents = results.get("documents", []) if isinstance(results, dict) else []
    if not documents:
        return []
    first_documents = documents[0] or []
    return [str(document) for document in first_documents if document]


def generate(question: str, context: str, language_code: str = "en") -> str:
    language = language_name(language_code)
    not_found_message = t(language_code, "message_not_found")
    prompt = f"""
You are an expert Sports Assistant.

Answer ONLY using the information provided below.
Do NOT make up facts.
Respond in {language}.

If the answer cannot be found in the context, reply exactly:

{not_found_message}

Context:

{context}

Question:

{question}

Answer:
"""

    response = call_groq_chat(
        TEXT_MODEL,
        [{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    answer = safe_response_text(response)
    return answer or t(language_code, "message_api_bad_response")


def generate_english_answer(question: str, context: str) -> str:
    is_comparison = compare_mode(question, keyword_route_query(question))
    prompt = f"""
You are an expert Sports Assistant.

Answer ONLY using the information provided below.
Do NOT make up facts.
Answer in English.

Format the response in clean Markdown.
Use short headings and bullet points where helpful.
If this is a comparison question, use a concise Markdown table when appropriate.

If the answer cannot be found in the context, reply exactly:

I could not find the answer in the provided datasets.

Context:

{context}

Question:

{question}

Comparison question:
{is_comparison}

Answer:
"""

    response = call_groq_chat(
        TEXT_MODEL,
        [{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    answer = safe_response_text(response)
    return answer or "I could not find the answer in the provided datasets."


def translate_answer_for_ui(answer: str, ui_language: str) -> tuple[str, str]:
    if ui_language == "en":
        return answer, ""
    try:
        translated = translate_text(answer, ui_language, "en")
        return translated, ""
    except (APITimeoutError, APIConnectionError, APIStatusError, BadRequestError, PermissionDeniedError, RateLimitError, GroqError, Exception):
        return answer, t(ui_language, "message_translation_failed")


def format_answer_markdown(answer: str) -> str:
    cleaned = (answer or "").strip()
    if not cleaned:
        return ""
    if cleaned.startswith("#") or cleaned.startswith("-") or "|" in cleaned:
        return cleaned
    paragraphs = [segment.strip() for segment in cleaned.split("\n\n") if segment.strip()]
    if len(paragraphs) == 1:
        return f"### Answer\n\n{cleaned}"
    return "\n\n".join(["### Answer", *paragraphs])


def validate_image_file(image_path: str | None, language_code: str) -> tuple[bool, str]:
    if not image_path:
        return False, t(language_code, "message_invalid_image_missing")

    path = Path(image_path)
    if not path.exists():
        return False, t(language_code, "message_invalid_image_missing")

    if path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        return False, t(language_code, "message_invalid_image_type")

    if path.stat().st_size > MAX_IMAGE_BYTES:
        return False, t(language_code, "message_invalid_image_size")

    try:
        with PILImage.open(path) as image_file:
            image_file.verify()
    except Exception:
        return False, t(language_code, "message_invalid_image_corrupt")

    return True, ""


def image_to_data_url(image_path: str) -> str:
    path = Path(image_path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    with path.open("rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def analyze_image(image_path: str, question: str, language_code: str) -> str:
    user_prompt = question.strip() if question else ""
    language = language_name(language_code)
    if user_prompt:
        prompt = f"""
You are an expert visual assistant.

Answer the user's question using the uploaded image.
Respond in {language}.

Question:
{user_prompt}
"""
    else:
        prompt = f"""
You are an expert visual assistant.

Describe the uploaded image in {language}.
Focus on any visible sports-related details.
"""

    response = call_groq_chat(
        VISION_MODEL,
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_path), "detail": "auto"}},
                ],
            }
        ],
        temperature=0.2,
    )
    answer = safe_response_text(response)
    return answer or t(language_code, "message_api_bad_response")


def sports_router(question: str, language_code: str = "en"):
    question = (question or "").strip()
    if not question:
        return "", "", t(language_code, "message_empty_question")

    route, english_question, translation_message = route_query_with_language(question, language_code)
    if route == "unknown":
        return (
            t(language_code, "route_unknown"),
            t(language_code, "message_no_documents"),
            t(language_code, "message_out_of_scope"),
        )

    retrieved_docs: list[str] = []
    if route == "cricket":
        retrieved_docs = retrieve(cricket_collection, english_question)
    elif route == "olympics":
        retrieved_docs = retrieve(olympics_collection, english_question)
    elif route == "both":
        cricket_docs = retrieve(cricket_collection, english_question)
        olympic_docs = retrieve(olympics_collection, english_question)
        retrieved_docs = cricket_docs + olympic_docs

    context = "\n\n".join(retrieved_docs)
    english_answer = generate_english_answer(english_question, context)
    translated_answer, answer_translation_message = translate_answer_for_ui(english_answer, language_code)
    messages = "\n".join(
        message for message in [translation_message, answer_translation_message] if message
    )
    return t(language_code, f"route_{route}"), context, translated_answer, messages


def set_processing(language_code: str):
    return status_html(language_code, "thinking"), t(language_code, "message_thinking")


def validate_uploaded_image(image_path: str | None, language_code: str):
    valid, error_message = validate_image_file(image_path, language_code)
    if not valid:
        return None, error_message
    return image_path, t(language_code, "message_image_received")


def process_submission(question: str, image_path: str | None, language_code: str):
    language_code = language_code if language_code in SUPPORTED_LANGUAGES else "en"
    question = (question or "").strip()

    try:
        if not question and not image_path:
            yield "", "", "", status_html(language_code, "error"), t(language_code, "message_no_input")
            return

        yield "", "", "", status_html(language_code, "thinking"), t(language_code, "message_thinking")

        if image_path:
            valid, error_message = validate_image_file(image_path, language_code)
            if not valid:
                yield "", "", "", status_html(language_code, "error"), error_message
                return

            yield t(language_code, "route_image"), t(language_code, "vision_context"), "", status_html(language_code, "processing"), t(language_code, "message_generating")

            answer = analyze_image(image_path, question, language_code)
            if not answer.strip():
                yield (
                    t(language_code, "route_image"),
                    t(language_code, "vision_context"),
                    "",
                    status_html(language_code, "error"),
                    t(language_code, "message_api_bad_response"),
                )
                return

            yield (
                t(language_code, "route_image"),
                t(language_code, "vision_context"),
                format_answer_markdown(answer),
                status_html(language_code, "completed"),
                "",
            )
            return

        route, english_question, translation_message = route_query_with_language(question, language_code)
        if route == "unknown":
            yield (
                t(language_code, "route_unknown"),
                t(language_code, "message_no_documents"),
                t(language_code, "message_out_of_scope"),
                status_html(language_code, "completed"),
                translation_message,
            )
            return

        yield t(language_code, f"route_{route}"), "", "", status_html(language_code, "retrieving"), t(language_code, "message_retrieving")

        try:
            if route == "cricket":
                retrieved_docs = retrieve(cricket_collection, english_question)
            elif route == "olympics":
                retrieved_docs = retrieve(olympics_collection, english_question)
            else:
                cricket_docs = retrieve(cricket_collection, english_question)
                olympic_docs = retrieve(olympics_collection, english_question)
                retrieved_docs = cricket_docs + olympic_docs
        except Exception:
            yield (
                t(language_code, "route_unknown"),
                "",
                "",
                status_html(language_code, "error"),
                t(language_code, "message_db_error"),
            )
            return

        context = "\n\n".join(retrieved_docs)
        yield t(language_code, f"route_{route}"), context, "", status_html(language_code, "processing"), t(language_code, "message_generating")
        english_answer = generate_english_answer(english_question, context)
        answer, answer_translation_message = translate_answer_for_ui(english_answer, language_code)
        combined_message = "\n".join(
            message for message in [translation_message, answer_translation_message] if message
        )
        if not answer.strip():
            yield (
                t(language_code, f"route_{route}"),
                context,
                "",
                status_html(language_code, "error"),
                t(language_code, "message_api_bad_response"),
            )
            return

        yield (
            t(language_code, f"route_{route}"),
            context,
            format_answer_markdown(answer),
            status_html(language_code, "completed"),
            combined_message,
        )
        return
    except APITimeoutError:
        yield "", "", "", status_html(language_code, "error"), t(language_code, "message_api_timeout")
        return
    except (APIConnectionError, APIStatusError, BadRequestError, PermissionDeniedError, RateLimitError, GroqError):
        yield "", "", "", status_html(language_code, "error"), t(language_code, "message_api_connection")
        return
    except Exception:
        yield "", "", "", status_html(language_code, "error"), t(language_code, "message_unknown_error")
        return


def build_language_updates(language_code: str, theme_code: str):
    language_code = language_code if language_code in SUPPORTED_LANGUAGES else "en"
    theme_code = apply_theme_state(theme_code)
    return (
        gr.update(value=language_code),
        gr.update(value=theme_toggle_label(theme_code, language_code)),
        f"# {t(language_code, 'app_title')}",
        f"{t(language_code, 'app_description')}\n\n{t(language_code, 'app_subdescription')}",
        gr.update(label=t(language_code, "question_label"), placeholder=t(language_code, "question_placeholder")),
        gr.update(value=t(language_code, "submit_button")),
        gr.update(value=t(language_code, "voice_button")),
        gr.update(label=t(language_code, "image_label"), placeholder=t(language_code, "image_placeholder")),
        gr.update(label=t(language_code, "route_label")),
        gr.update(label=t(language_code, "context_label")),
        gr.update(label=t(language_code, "answer_label")),
        gr.update(label=t(language_code, "message_label")),
        status_html(language_code, "idle"),
    )


def initialize_ui(language_code: str, theme_code: str):
    language_code = language_code if language_code in SUPPORTED_LANGUAGES else "en"
    theme_code = apply_theme_state(theme_code)
    return (
        gr.update(value=language_code),
        gr.update(value=theme_toggle_label(theme_code, language_code)),
        f"# {t(language_code, 'app_title')}",
        f"{t(language_code, 'app_description')}\n\n{t(language_code, 'app_subdescription')}",
        gr.update(label=t(language_code, "question_label"), placeholder=t(language_code, "question_placeholder")),
        gr.update(value=t(language_code, "submit_button")),
        gr.update(value=t(language_code, "voice_button")),
        gr.update(label=t(language_code, "image_label"), placeholder=t(language_code, "image_placeholder")),
        gr.update(label=t(language_code, "route_label")),
        gr.update(label=t(language_code, "context_label")),
        gr.update(label=t(language_code, "answer_label")),
        status_html(language_code, "idle"),
        gr.update(label=t(language_code, "message_label"), value=""),
    )


def build_voice_js() -> str:
    speech_map = json.dumps(SPEECH_LANGUAGE_MAP, ensure_ascii=False)
    bundle = {
        "en": {
            "status_idle": t("en", "status_idle"),
            "status_listening": t("en", "status_listening"),
            "status_processing": t("en", "status_processing"),
            "status_completed": t("en", "status_completed"),
            "status_error": t("en", "status_error"),
            "message_unsupported_browser": t("en", "message_unsupported_browser"),
            "message_permission_denied": t("en", "message_permission_denied"),
            "message_empty_speech": t("en", "message_empty_speech"),
            "message_recognition_timeout": t("en", "message_recognition_timeout"),
            "message_recognition_failed": t("en", "message_recognition_failed"),
            "message_offline": t("en", "message_offline"),
        },
        "hi": {
            "status_idle": t("hi", "status_idle"),
            "status_listening": t("hi", "status_listening"),
            "status_processing": t("hi", "status_processing"),
            "status_completed": t("hi", "status_completed"),
            "status_error": t("hi", "status_error"),
            "message_unsupported_browser": t("hi", "message_unsupported_browser"),
            "message_permission_denied": t("hi", "message_permission_denied"),
            "message_empty_speech": t("hi", "message_empty_speech"),
            "message_recognition_timeout": t("hi", "message_recognition_timeout"),
            "message_recognition_failed": t("hi", "message_recognition_failed"),
            "message_offline": t("hi", "message_offline"),
        },
        "te": {
            "status_idle": t("te", "status_idle"),
            "status_listening": t("te", "status_listening"),
            "status_processing": t("te", "status_processing"),
            "status_completed": t("te", "status_completed"),
            "status_error": t("te", "status_error"),
            "message_unsupported_browser": t("te", "message_unsupported_browser"),
            "message_permission_denied": t("te", "message_permission_denied"),
            "message_empty_speech": t("te", "message_empty_speech"),
            "message_recognition_timeout": t("te", "message_recognition_timeout"),
            "message_recognition_failed": t("te", "message_recognition_failed"),
            "message_offline": t("te", "message_offline"),
        },
    }
    bundle_json = json.dumps(bundle, ensure_ascii=False)
    return f"""
async (languageCode) => {{
  const speechMap = {speech_map};
  const localized = {bundle_json};
  const bundle = localized[languageCode] || localized.en;
  const statusElement = document.getElementById("app-status");

  const setStatus = (stateKey, detail = "") => {{
    if (!statusElement) return;
    const label = bundle[stateKey] || stateKey;
    const detailHtml = detail ? `<span style="margin-left: 0.5rem; opacity: 0.85;">${{detail}}</span>` : "";
    statusElement.innerHTML = `<div style="padding: 0.7rem 0.85rem; border-radius: 0.85rem; border: 1px solid rgba(0,0,0,0.12); background: rgba(0,0,0,0.03); font-size: 0.95rem; line-height: 1.35;"><strong>${{label}}</strong>${{detailHtml}}</div>`;
  }};

  try {{
    if (!navigator.onLine) {{
      setStatus("status_error", bundle.message_offline);
      return ["", bundle.message_offline];
    }}

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {{
      setStatus("status_error", bundle.message_unsupported_browser);
      return ["", bundle.message_unsupported_browser];
    }}

    const recognition = new SpeechRecognition();
    recognition.lang = speechMap[languageCode] || "en-US";
    recognition.interimResults = false;
    recognition.continuous = false;
    recognition.maxAlternatives = 1;

    setStatus("status_listening");

    const result = await new Promise((resolve) => {{
      let finished = false;
      const timeoutId = window.setTimeout(() => {{
        if (finished) return;
        finished = true;
        try {{ recognition.abort(); }} catch (error) {{}}
        setStatus("status_error", bundle.message_recognition_timeout);
        resolve({{ text: "", error: bundle.message_recognition_timeout }});
      }}, 10000);

      recognition.onresult = (event) => {{
        if (finished) return;
        finished = true;
        window.clearTimeout(timeoutId);
        const text = Array.from(event.results)
          .map((result) => result[0].transcript)
          .join(" ")
          .trim();
        if (!text) {{
          setStatus("status_error", bundle.message_empty_speech);
          resolve({{ text: "", error: bundle.message_empty_speech }});
          return;
        }}
        setStatus("status_processing");
        resolve({{ text, error: "" }});
      }};

      recognition.onerror = (event) => {{
        if (finished) return;
        finished = true;
        window.clearTimeout(timeoutId);
        const code = event && event.error ? event.error : "error";
        let errorMessage = bundle.message_recognition_failed;
        if (code === "not-allowed" || code === "service-not-allowed" || code === "permission-denied") {{
          errorMessage = bundle.message_permission_denied;
        }} else if (code === "no-speech") {{
          errorMessage = bundle.message_empty_speech;
        }} else if (code === "network") {{
          errorMessage = bundle.message_offline;
        }}
        setStatus("status_error", errorMessage);
        resolve({{ text: "", error: errorMessage }});
      }};

      recognition.onend = () => {{
        if (!finished) {{
          finished = true;
          window.clearTimeout(timeoutId);
          setStatus("status_error", bundle.message_recognition_failed);
          resolve({{ text: "", error: bundle.message_recognition_failed }});
        }}
      }};

      try {{
        recognition.start();
      }} catch (error) {{
        finished = true;
        window.clearTimeout(timeoutId);
        setStatus("status_error", bundle.message_recognition_failed);
        resolve({{ text: "", error: bundle.message_recognition_failed }});
      }}
    }});

    if (result.text) {{
      setStatus("status_completed");
      return [result.text, ""];
    }}

    return ["", result.error || bundle.message_recognition_failed];
  }} catch (error) {{
    const message = error && error.message ? error.message : bundle.message_recognition_failed;
    setStatus("status_error", message);
    return ["", message];
  }}
}}
"""


def build_theme_init_js() -> str:
    return """
async () => {
    const storedTheme = localStorage.getItem('agentic_sports_router_theme') || 'light';
    document.documentElement.setAttribute('data-theme', storedTheme);
    document.documentElement.style.colorScheme = storedTheme;
}
"""


def build_theme_toggle_js() -> str:
    bundle = {
        "en": {"light": t("en", "theme_light_mode"), "dark": t("en", "theme_dark_mode")},
        "hi": {"light": t("hi", "theme_light_mode"), "dark": t("hi", "theme_dark_mode")},
        "te": {"light": t("te", "theme_light_mode"), "dark": t("te", "theme_dark_mode")},
    }
    bundle_json = json.dumps(bundle, ensure_ascii=False)
    return f"""
async (themeCode, languageCode) => {{
    const localized = {bundle_json};
    const bundle = localized[languageCode] || localized.en;
    const nextTheme = themeCode === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', nextTheme);
    document.documentElement.style.colorScheme = nextTheme;
    localStorage.setItem('agentic_sports_router_theme', nextTheme);
    return [nextTheme, bundle[nextTheme]];
}}
"""


with gr.Blocks(css=APP_CSS, theme=gr.themes.Soft()) as demo:
    browser_language = gr.BrowserState(default_value="en", storage_key="agentic_sports_router_lang")
    browser_theme = gr.BrowserState(default_value=DEFAULT_THEME, storage_key="agentic_sports_router_theme")

    with gr.Column(elem_id="app-shell"):
        with gr.Column(elem_classes=["hero-panel"]):
            title_md = gr.Markdown(elem_classes=["hero-title"])
            description_md = gr.Markdown(elem_classes=["hero-description"])
            with gr.Row(elem_classes=["toolbar-row"]):
                language_selector = gr.Dropdown(
                    choices=[("English", "en"), ("हिन्दी", "hi"), ("తెలుగు", "te")],
                    value="en",
                    label=t("en", "language_label"),
                    scale=1,
                    min_width=180,
                )
                with gr.Row(elem_classes=["toolbar-actions"]):
                    theme_button = gr.Button(
                        value=theme_toggle_label(DEFAULT_THEME, "en"),
                        variant="secondary",
                        scale=1,
                        min_width=140,
                    )

        with gr.Column(elem_classes=["surface-card"]):
            with gr.Row(elem_classes=["control-row"]):
                question_box = gr.Textbox(
                    lines=2,
                    placeholder=t("en", "question_placeholder"),
                    label=t("en", "question_label"),
                    scale=8,
                )
                ask_button = gr.Button(
                    value=t("en", "submit_button"),
                    variant="primary",
                    scale=1,
                    min_width=120,
                )
                voice_button = gr.Button(
                    value=t("en", "voice_button"),
                    variant="secondary",
                    scale=1,
                    min_width=120,
                )

        status_box = gr.HTML(value=status_html("en", "idle"), elem_id="app-status")

        with gr.Column(elem_classes=["surface-card"]):
            image_input = gr.Image(
                type="filepath",
                sources=["upload"],
                label=t("en", "image_label"),
                placeholder=t("en", "image_placeholder"),
                height=240,
                show_label=True,
            )
            message_box = gr.Markdown(
                value="",
                label=t("en", "message_label"),
                elem_classes=["message-markdown"],
            )

        with gr.Column(elem_classes=["output-card"]):
            route_box = gr.Textbox(label=t("en", "route_label"), lines=1, interactive=False)
            answer_box = gr.Markdown(value="", label=t("en", "answer_label"), elem_classes=["answer-markdown"])
            with gr.Accordion(label=t("en", "context_label"), open=False):
                context_box = gr.Textbox(lines=12, interactive=False, label=t("en", "context_label"))

    language_selector.change(
        build_language_updates,
        inputs=[language_selector, browser_theme],
        outputs=[
            browser_language,
            theme_button,
            title_md,
            description_md,
            question_box,
            ask_button,
            voice_button,
            image_input,
            route_box,
            context_box,
            answer_box,
            message_box,
            status_box,
        ],
        queue=False,
    )

    demo.load(
        initialize_ui,
        inputs=[browser_language, browser_theme],
        outputs=[
            browser_language,
            theme_button,
            title_md,
            description_md,
            question_box,
            ask_button,
            voice_button,
            image_input,
            route_box,
            context_box,
            answer_box,
            status_box,
            message_box,
        ],
        queue=False,
    )

    demo.load(
        fn=None,
        inputs=browser_theme,
        outputs=[],
        js=build_theme_init_js(),
        queue=False,
    )

    theme_button.click(
        fn=None,
        inputs=[browser_theme, browser_language],
        outputs=[browser_theme, theme_button],
        js=build_theme_toggle_js(),
        queue=False,
    )

    voice_button.click(
        fn=None,
        inputs=language_selector,
        outputs=[question_box, message_box],
        js=build_voice_js(),
        queue=False,
    )

    image_input.change(
        validate_uploaded_image,
        inputs=[image_input, language_selector],
        outputs=[image_input, message_box],
        queue=False,
    )

    submit_chain = ask_button.click(
        set_processing,
        inputs=language_selector,
        outputs=[status_box, message_box],
        queue=False,
    )

    submit_chain.then(
        process_submission,
        inputs=[question_box, image_input, language_selector],
        outputs=[route_box, context_box, answer_box, status_box, message_box],
    )

    question_box.submit(
        set_processing,
        inputs=language_selector,
        outputs=[status_box, message_box],
        queue=False,
    ).then(
        process_submission,
        inputs=[question_box, image_input, language_selector],
        outputs=[route_box, context_box, answer_box, status_box, message_box],
    )

    demo.queue()


if __name__ == "__main__":
    demo.launch()