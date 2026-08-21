import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import re
import html
from time import perf_counter
from datetime import datetime, timezone
from uuid import uuid4
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Minimum cosine similarity required before Version 3 will return a matched
# answer. Below this, the agent reports low confidence instead of guessing.
SIMILARITY_THRESHOLD = 0.15

try:
    from st_keyup import st_keyup
except ImportError:
    st_keyup = None

st.set_page_config(page_title="Multi-Algorithm IT Chatbot", page_icon="🤖", layout="wide")

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,500&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg: #F5F4EE;
        --panel: #FAF9F5;
        --panel-raised: #FFFFFF;
        --border: rgba(30, 27, 22, 0.09);
        --border-strong: rgba(30, 27, 22, 0.16);
        --text: #2D2A26;
        --text-muted: #75716A;
        --accent: #D97757;
        --accent-hover: #C4653F;
        --accent-soft: rgba(217, 119, 87, 0.12);
        --bubble: #ECEAE3;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }

    h1, h2, h3, .stMarkdown h3 { color: var(--text); font-family: 'Inter', sans-serif; }

    /* ---------- Greeting ---------- */
    .greeting-wrap {
        padding: 1rem 0 1.75rem 0;
        text-align: center;
    }

    .greeting-wrap.is-fresh {
        min-height: 46vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding-bottom: 2.25rem;
    }

    .greeting-wrap .greeting-subtitle {
        margin-left: auto;
        margin-right: auto;
    }

    .greeting-wrap .status-row {
        justify-content: center;
    }

    .greeting-title {
        font-family: 'Source Serif 4', Georgia, serif;
        font-weight: 500;
        font-size: 2.15rem;
        line-height: 1.25;
        color: var(--text);
        margin: 0 0 0.4rem 0;
    }

    .greeting-title .accent-dot {
        color: var(--accent);
    }

    .greeting-subtitle {
        color: var(--text-muted);
        font-size: 1rem;
        margin: 0 0 1.1rem 0;
        max-width: 600px;
    }

    .status-row { display: flex; flex-wrap: wrap; gap: 0.5rem; }

    .status-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        background: var(--panel-raised);
        border: 1px solid var(--border);
        color: var(--text-muted);
        font-size: 0.8rem;
        font-weight: 500;
    }

    .status-chip b { color: var(--text); font-weight: 600; }
    .status-chip .swatch { width: 6px; height: 6px; border-radius: 50%; display: inline-block; background: var(--accent); }

    /* ---------- Agent switcher: segmented pill control ---------- */
    .agent-segment {
        display: inline-flex;
        background: var(--panel-raised);
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: 0.25rem;
        gap: 0.25rem;
        margin-bottom: 0.35rem;
    }

    div[data-testid="column"] div[data-testid="stButton"] > button {
        border-radius: 999px !important;
    }

    .agent-desc-text {
        color: var(--text-muted);
        font-size: 0.86rem;
        margin: 0.5rem 0 1.25rem 0;
        text-align: center;
    }

    /* ---------- Buttons ---------- */
    div[data-testid="stButton"] > button {
        border-radius: 10px;
        border: 1px solid var(--border-strong);
        background: var(--panel-raised);
        color: var(--text);
        padding: 0.55rem 1rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        box-shadow: none;
    }

    div[data-testid="stButton"] > button:hover {
        border-color: var(--accent);
        color: var(--accent-hover);
    }

    div[data-testid="stButton"] > button[kind="primary"] {
        background: var(--accent);
        color: #FFFFFF;
        border: none;
    }

    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background: var(--accent-hover);
        color: #FFFFFF;
    }

    /* ---------- Chat ---------- */
    div[data-testid="stChatMessage"] {
        background: transparent;
        border: none;
        padding-left: 0;
        padding-right: 0;
    }

    div[data-testid="stChatMessage"] > div:first-child {
        display: none !important;
    }

    div[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {
        flex-direction: row;
        justify-content: flex-end;
    }

    div[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) > div:last-child {
        flex: 1 1 auto;
        width: 100% !important;
        max-width: 100%;
        margin-left: auto !important;
    }

    div[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stMarkdownContainer"] {
        text-align: right !important;
    }

    .assistant-bubble {
        background: transparent;
    }

    .user-bubble {
        background: var(--bubble);
        border-radius: 16px;
        padding: 0.65rem 1rem;
        display: block;
        width: fit-content;
        margin-left: auto !important;
        margin-right: 0 !important;
        text-align: left;
    }

    .chat-caption {
        font-size: 0.78rem;
        color: var(--text-muted);
        margin-top: 0.65rem;
        padding-top: 0.55rem;
        border-top: 1px solid var(--border);
        display: flex;
        align-items: center;
        gap: 0.55rem;
        flex-wrap: wrap;
    }

    .chat-caption .topic-tag {
        background: var(--accent-soft);
        color: var(--accent-hover);
        padding: 0.1rem 0.5rem;
        border-radius: 999px;
        font-weight: 500;
    }

    .feedback-label {
        color: var(--text-muted);
        font-size: 0.78rem;
        margin-top: 0.75rem;
    }

    .feedback-row div[data-testid="stButton"] > button {
        min-height: 2rem;
        padding: 0.25rem 0.7rem;
        font-size: 0.9rem;
    }

    /* confidence meter: slim rounded progress bar */
    .confidence-track {
        display: inline-block;
        width: 56px;
        height: 5px;
        border-radius: 999px;
        background: var(--border-strong);
        overflow: hidden;
        vertical-align: middle;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 999px;
        background: var(--meter-color, var(--accent));
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: var(--panel);
        border-right: 1px solid var(--border);
        color: var(--text);
    }

    section[data-testid="stSidebar"] * {
        color: var(--text);
    }

    section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small {
        color: var(--text-muted) !important;
    }

    section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4 {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--text-muted);
    }

    div[data-testid="stMetric"] {
        background: var(--panel-raised);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.5rem 0.7rem;
    }

    div[data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif;
        color: var(--accent);
        font-size: 1.1rem;
        font-weight: 600;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    section[data-testid="stSidebar"] div[data-testid="stExpander"] {
        background: var(--panel-raised);
        border: 1px solid var(--border);
        border-radius: 10px;
    }

    /* session list rows */
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
        text-align: left;
        justify-content: flex-start;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.86rem;
        background: transparent;
        border: 1px solid transparent;
        border-radius: 8px;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
        background: var(--panel-raised);
        border-color: var(--border);
        color: var(--accent-hover);
    }

    input, textarea {
        font-family: 'Inter', sans-serif !important;
    }

    /* focus visibility */
    button:focus-visible, input:focus-visible {
        outline: 2px solid var(--accent) !important;
        outline-offset: 2px;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

AGENT_OPTIONS = [
    "Version 1: Exact Match Agent",
    "Version 2: Pattern Matching Agent",
    "Version 3: Machine Learning Agent",
]

KEYWORD_ROUTES = [
    (("wifi", "wi-fi", "connection"), "wifi"),
    (("password", "reset", "lock"), "password"),
    (("laptop", "screen", "hardware", "display"), "laptop"),
    (("outlook", "email", "phone"), "outlook"),
    (("printer", "print"), "printer"),
    (("vpn",), "vpn"),
]

AGENT_META = {
    "Version 1: Exact Match Agent": {
        "label": "Exact Match",
        "description": "Best for exact FAQ wording. Fast, strict, and deterministic.",
        "logic": "String comparison verification (`==`). The user query must match a database question exactly (case-insensitive, trimmed) to fetch a result.",
        "accent": "#D97757",
        "icon": "🎯",
    },
    "Version 2: Pattern Matching Agent": {
        "label": "Pattern Match",
        "description": "Best for keyword-heavy questions and quick topic routing.",
        "logic": "Fuzzy token inclusion mapping (`in` operator). The pipeline sweeps user inputs for high-frequency technical key-terms to trigger responses.",
        "accent": "#D97757",
        "icon": "🧭",
    },
    "Version 3: Machine Learning Agent": {
        "label": "Machine Learning",
        "description": "Best for flexible wording and broader semantic matching.",
        "logic": "TF-IDF text vectorization with cosine similarity retrieval across all FAQ questions. Returns the closest-matching answer with a confidence score, or flags low-confidence queries below the similarity threshold.",
        "accent": "#D97757",
        "icon": "🧠",
    },
}


def render_confidence_meter(score: float, threshold: float) -> str:
    """Slim rounded progress bar showing match confidence (0-1)."""
    pct = max(0, min(100, round(score * 100)))
    if score < threshold:
        color = "#B5471E"
    elif score < 0.5:
        color = "#D97757"
    else:
        color = "#2E7D5B"
    return (
        f'<span class="confidence-track">'
        f'<span class="confidence-fill" style="width:{pct}%; --meter-color:{color}"></span>'
        f'</span>'
    )


def render_chat_message(message: dict) -> None:
    """Render one chat turn. User turns show as a soft rounded bubble;
    assistant turns are plain text (Claude-style, no bubble) with a small
    meta line for matched question / topic / confidence when available."""
    role = message["role"]
    with st.chat_message(role, avatar=("🧑" if role == "user" else "✳️")):
        if role == "user":
            st.markdown(
                f"<div class='user-bubble'>{html.escape(message['content'])}</div>",
                unsafe_allow_html=True,
            )
            return

        meta = message.get("meta") or {}

        comparison_results = meta.get("comparison_results")
        if comparison_results:
            st.markdown(message["content"])
            result_cols = st.columns(3)
            for column, result in zip(result_cols, comparison_results):
                with column:
                    st.markdown(f"**{result['label']}**")
                    st.caption(f"Latency: {result['latency_ms']:.2f} ms")
                    st.markdown(result["answer"])

                    result_meta = result.get("meta") or {}
                    local_caption_parts = []
                    if result_meta.get("matched_question"):
                        local_caption_parts.append(
                            f"matched \u201c{html.escape(str(result_meta['matched_question']))}\u201d"
                        )
                    if result_meta.get("topic"):
                        local_caption_parts.append(
                            f"<span class='topic-tag'>{html.escape(str(result_meta['topic']))}</span>"
                        )
                    if result_meta.get("score") is not None:
                        score = result_meta["score"]
                        threshold = result_meta.get("threshold", SIMILARITY_THRESHOLD)
                        local_caption_parts.append(f"{render_confidence_meter(score, threshold)} {score:.2f}")

                    if local_caption_parts:
                        st.markdown(
                            f"<div class='chat-caption'>{' &middot; '.join(local_caption_parts)}</div>",
                            unsafe_allow_html=True,
                        )
            return

        st.markdown(message["content"])
        caption_parts = []
        if meta.get("matched_question"):
            caption_parts.append(
                f"matched \u201c{html.escape(str(meta['matched_question']))}\u201d"
            )
        if meta.get("topic"):
            caption_parts.append(f"<span class='topic-tag'>{html.escape(str(meta['topic']))}</span>")
        if meta.get("score") is not None:
            score = meta["score"]
            threshold = meta.get("threshold", SIMILARITY_THRESHOLD)
            caption_parts.append(f"{render_confidence_meter(score, threshold)} {score:.2f}")

        if caption_parts:
            st.markdown(
                f"<div class='chat-caption'>{' &middot; '.join(caption_parts)}</div>",
                unsafe_allow_html=True,
            )

        feedback = message.get("feedback")
        st.markdown("<div class='feedback-label'>Was this response helpful?</div>", unsafe_allow_html=True)
        feedback_cols = st.columns([1, 1, 8])
        with feedback_cols[0]:
            if st.button(
                "👍",
                key=f"feedback_up_{message['id']}",
                type="primary" if feedback == "up" else "secondary",
                help="Mark this response as helpful",
            ):
                message["feedback"] = "up"
                st.rerun()
        with feedback_cols[1]:
            if st.button(
                "👎",
                key=f"feedback_down_{message['id']}",
                type="primary" if feedback == "down" else "secondary",
                help="Mark this response as not helpful",
            ):
                message["feedback"] = "down"
                st.rerun()

WELCOME_MESSAGE = "Hello. I am your automated IT support agent. Feel free to submit a query regarding wifi setups, password policies, or system issues."

if "active_agent" not in st.session_state:
    st.session_state.active_agent = AGENT_OPTIONS[2]

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = [
        {
            "id": str(uuid4()),
            "title": "New Chat",
            "messages": [
                {
                    "id": str(uuid4()),
                    "role": "assistant",
                    "content": WELCOME_MESSAGE,
                }
            ],
        }
    ]

if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = st.session_state.chat_sessions[0]["id"]

if "session_search_mode" not in st.session_state:
    st.session_state.session_search_mode = False

if "session_search_query" not in st.session_state:
    st.session_state.session_search_query = ""

def set_agent(agent_name: str) -> None:
    st.session_state.active_agent = agent_name

def get_active_session() -> dict:
    for session in st.session_state.chat_sessions:
        if session["id"] == st.session_state.active_session_id:
            return session
    st.session_state.active_session_id = st.session_state.chat_sessions[0]["id"]
    return st.session_state.chat_sessions[0]

def get_active_messages() -> list[dict]:
    return get_active_session()["messages"]

def create_new_chat() -> None:
    new_session = {
        "id": str(uuid4()),
        "title": "New Chat",
        "messages": [
            {
                "id": str(uuid4()),
                "role": "assistant",
                "content": WELCOME_MESSAGE,
            }
        ],
    }
    st.session_state.chat_sessions.insert(0, new_session)
    st.session_state.active_session_id = new_session["id"]

def clear_chat_history() -> None:
    active_session = get_active_session()
    active_session["messages"] = [
        {
            "id": str(uuid4()),
            "role": "assistant",
            "content": WELCOME_MESSAGE,
        }
    ]
    active_session["title"] = "New Chat"

def set_active_session(session_id: str) -> None:
    st.session_state.active_session_id = session_id

def label_chat_topic(user_text: str) -> str:
    normalized_text = user_text.lower()

    topic_rules = [
        (("wifi", "wi-fi", "wireless", "internet", "network", "connection"), "Wifi Connection"),
        (("password", "reset", "locked", "login", "sign in", "signin"), "Password Reset"),
        (("laptop", "screen", "display", "monitor", "hardware", "computer"), "Laptop Display"),
        (("outlook", "email", "mail", "phone", "mailbox"), "Email Issues"),
        (("printer", "print"), "Printer Issue"),
        (("vpn",), "VPN Connection"),
    ]

    for keywords, title in topic_rules:
        if any(keyword in normalized_text for keyword in keywords):
            return title

    words = re.findall(r"[A-Za-z0-9]+", user_text)
    if not words:
        return "New Chat"

    short_words = [word for word in words if len(word) > 2][:3]
    if not short_words:
        short_words = words[:3]

    title = " ".join(short_words)
    return title[:28].strip() or "New Chat"

def update_active_session_title(user_text: str) -> None:
    session = get_active_session()
    if session["title"] == "New Chat":
        session["title"] = label_chat_topic(user_text)

def get_session_preview(messages: list[dict]) -> str:
    for message in messages:
        if message["role"] == "user":
            return message["content"]
    return messages[0]["content"] if messages else ""

def format_session_title(session: dict) -> str:
    title = session.get("title", "New Chat").strip()
    return title if title else "New Chat"

def migrate_legacy_chat_titles() -> None:
    for session in st.session_state.chat_sessions:
        title = str(session.get("title", "")).strip()
        if not title:
            session["title"] = "New Chat"
            continue

        if title == "LOQ Laptop Monitor Not Working":
            first_user_message = next(
                (
                    message["content"].strip()
                    for message in session.get("messages", [])
                    if message.get("role") == "user" and message.get("content", "").strip()
                ),
                "",
            )
            session["title"] = label_chat_topic(first_user_message) if first_user_message else "New Chat"

migrate_legacy_chat_titles()

for session in st.session_state.chat_sessions:
    for message in session.get("messages", []):
        message.setdefault("id", str(uuid4()))

# ------------------------------------------------------------------
# BACKEND: DATA LOADING & ML INITIALIZATION
# ------------------------------------------------------------------
@st.cache_resource
def load_data():
    """
    Load and adapt the raw support-ticket export (it_support_dataset.csv) into the
    topic/question/answer shape the retrieval engine expects.

    The source file is a multilingual customer-support ticket dump with
    columns: subject, body, answer, type, queue, priority, language,
    business_type, tag_1..tag_9 — not a pre-built FAQ table. To turn it into
    an FAQ-style knowledge base we:
      1. Keep only English-language tickets (language == "en"), since the
         chat UI and similarity matching are English-only.
      2. Use `subject` as the FAQ "question". About 12% of English rows have
         a blank subject, so for those we fall back to the first ~80
         characters of `body` as a stand-in question.
      3. Use `answer` as-is for the FAQ "answer".
      4. Use `queue` as the FAQ "topic" (e.g. "Technical Support",
         "Billing and Payments") since it's the closest categorical field to
         a topic label.
      5. Drop duplicate question/answer pairs and any rows still missing a
         usable question or answer after the fallback.
    """
    try:
        df = pd.read_csv("it_support_dataset.csv", engine="python", on_bad_lines="skip")
    except FileNotFoundError:
        return None

    if {"question", "answer"}.issubset(df.columns):
        df = df.copy()
        df["question"] = df["question"].astype(str).str.strip()
        if "topic" in df.columns:
            df["topic"] = df["topic"].astype(str).str.strip()
        else:
            df["topic"] = ""
    else:
        if "language" in df.columns:
            df = df[df["language"].astype(str).str.lower().eq("en")].copy()

        def fallback_question(row) -> str:
            subject = str(row.get("subject", "")).strip()
            if subject and subject.lower() != "nan":
                return subject
            body = str(row.get("body", "")).strip()
            body = re.sub(r"\s+", " ", body)
            return body[:80].strip()

        df["question"] = df.apply(fallback_question, axis=1)
        topic_column = "queue" if "queue" in df.columns else "topic"
        if topic_column in df.columns:
            df["topic"] = df[topic_column].astype(str).str.strip()
        else:
            df["topic"] = ""

    df["answer"] = df["answer"].astype(str).str.strip()

    df = df[(df["question"] != "") & (df["answer"] != "") & (df["answer"].str.lower() != "nan")]
    df = df.drop_duplicates(subset=["question", "answer"])
    df = df.reset_index(drop=True)
    return df


@st.cache_resource
def build_retrieval_engine(_df: pd.DataFrame):
    """
    Fit a TF-IDF vectorizer over every FAQ question and use cosine similarity
    for retrieval. NOTE: every answer in this dataset is unique (787 unique
    answers for 787 rows), so treating this as a multi-class classification
    problem (one class per answer) is not viable — after any train/test split,
    the model would never have seen the correct class for a test-set question.
    Nearest-neighbour retrieval via cosine similarity is the correct approach
    and is cached once via @st.cache_resource so it only runs on first load.
    """
    vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
    question_matrix = vectorizer.fit_transform(_df["question"])
    return vectorizer, question_matrix


@st.cache_resource
def evaluate_retrieval(_df: pd.DataFrame, _vectorizer: TfidfVectorizer, threshold: float):
    """
    Evaluate retrieval quality on a held-out split. Since every question in
    the dataset is unique text, "correct" means the held-out question
    retrieves ITS OWN row (top-1 self-retrieval) when matched against the
    full question set. This tests whether TF-IDF similarity is discriminative
    enough to distinguish each question from its 786 neighbours.
    """
    train_idx, test_idx = train_test_split(
        np.arange(len(_df)), test_size=0.2, random_state=42
    )
    full_matrix = _vectorizer.transform(_df["question"])
    test_matrix = _vectorizer.transform(_df.iloc[test_idx]["question"])

    sims = cosine_similarity(test_matrix, full_matrix)
    top1_idx = sims.argmax(axis=1)
    top1_score = sims.max(axis=1)

    correct = (top1_idx == test_idx)
    accuracy_at_1 = float(correct.mean())
    above_threshold = float((top1_score >= threshold).mean())
    avg_similarity = float(top1_score.mean())
    median_similarity = float(np.median(top1_score))

    return {
        "n_test": len(test_idx),
        "accuracy_at_1": accuracy_at_1,
        "above_threshold_rate": above_threshold,
        "avg_similarity": avg_similarity,
        "median_similarity": median_similarity,
        "threshold": threshold,
    }


def retrieve_best_match(query: str, vectorizer: TfidfVectorizer, question_matrix, threshold: float):
    """Return (answer, matched_question, topic, score) or (None, None, None, score) if below threshold."""
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, question_matrix)[0]
    best_idx = int(sims.argmax())
    best_score = float(sims[best_idx])

    if best_score < threshold:
        return None, None, None, best_score

    row = df.iloc[best_idx]
    topic = row["topic"] if "topic" in df.columns else None
    return row["answer"], row["question"], topic, best_score


def run_exact_match(clean_query: str) -> tuple[str, dict]:
    matched_rows = df[df["question"].str.lower().str.strip() == clean_query]
    if not matched_rows.empty:
        row = matched_rows.iloc[0]
        return row["answer"], {"matched_question": row["question"], "topic": row.get("topic")}
    return (
        "**No exact match.** Exact Match requires the input to mimic a knowledge-base question exactly (case-insensitive, trimmed) — try Pattern Match or Machine Learning instead.",
        {},
    )


def run_pattern_match(clean_query: str) -> tuple[str, dict]:
    matched_route = next(
        (route_term for triggers, route_term in KEYWORD_ROUTES if any(t in clean_query for t in triggers)),
        None,
    )
    if matched_route:
        matches = df[df["question"].str.lower().str.contains(matched_route)]
        if not matches.empty:
            row = matches.iloc[0]
            return row["answer"], {"matched_question": row["question"], "topic": row.get("topic")}
        return f"Keyword **{matched_route}** was flagged, but no matching knowledge-base entries were found.", {}
    return (
        "**No key-term detected.** Pattern Match only fires on predefined terms (wifi, password, laptop, outlook, printer, vpn) — try Machine Learning for flexible wording.",
        {},
    )


def run_machine_learning(query: str) -> tuple[str, dict]:
    answer, matched_question, topic, score = retrieve_best_match(
        query, vectorizer, question_matrix, SIMILARITY_THRESHOLD
    )
    if answer is not None:
        return answer, {
            "matched_question": matched_question,
            "topic": topic,
            "score": score,
            "threshold": SIMILARITY_THRESHOLD,
        }
    return (
        "**No confident match.** Try rephrasing with more specific keywords or system names.",
        {"score": score, "threshold": SIMILARITY_THRESHOLD},
    )


df = load_data()

if df is None:
    st.error("⚠️ 'it_support_dataset.csv' not found! Please verify it is saved in your project root directory.")
    st.stop()

vectorizer, question_matrix = build_retrieval_engine(df)
eval_report = evaluate_retrieval(df, vectorizer, SIMILARITY_THRESHOLD)

# ------------------------------------------------------------------
# SIDEBAR: CHAT SESSION LIST
# ------------------------------------------------------------------
with st.sidebar.expander("Retrieval quality report", expanded=False):
    st.caption(
        "Held-out split — does TF-IDF similarity retrieve each question's "
        f"own answer over its {len(df) - 1:,} neighbours?"
    )
    m1, m2 = st.columns(2)
    m1.metric("Top-1 Accuracy", f"{eval_report['accuracy_at_1']*100:.1f}%")
    m2.metric("Avg. Confidence", f"{eval_report['avg_similarity']:.2f}")
    m3, m4 = st.columns(2)
    m3.metric("Median Confidence", f"{eval_report['median_similarity']:.2f}")
    m4.metric("Above Threshold", f"{eval_report['above_threshold_rate']*100:.1f}%")
    st.caption(
        f"n = {eval_report['n_test']} held-out questions · "
        f"threshold = {eval_report['threshold']}"
    )

st.sidebar.markdown("### Chat sessions")
st.sidebar.caption("New chat, search chat, and recent conversations.")

top_actions = st.sidebar.columns(2)
with top_actions[0]:
    if st.button("New Chat", use_container_width=True):
        create_new_chat()
        st.rerun()
with top_actions[1]:
    if st.button("Search Chat", use_container_width=True):
        st.session_state.session_search_mode = not st.session_state.session_search_mode
        st.rerun()

if st.sidebar.button("Clear Chat History", use_container_width=True):
    clear_chat_history()
    st.rerun()

chat_export_lines = [
    "IT Chat History",
    f"Exported: {datetime.now(timezone.utc).isoformat()}",
    f"Active agent: {st.session_state.active_agent}",
    "",
]
for session_number, session in enumerate(st.session_state.chat_sessions, start=1):
    chat_export_lines.extend([
        f"SESSION {session_number}: {format_session_title(session)}",
        f"Session ID: {session['id']}",
        "-" * 72,
    ])
    for message in session.get("messages", []):
        role = "User" if message.get("role") == "user" else "Assistant"
        chat_export_lines.extend([
            f"{role}:",
            str(message.get("content", "")),
        ])
        if message.get("feedback"):
            chat_export_lines.append(f"Feedback: {message['feedback']}")
        chat_export_lines.append("")
    chat_export_lines.append("")

st.sidebar.download_button(
    "Download Chat History",
    data="\n".join(chat_export_lines),
    file_name="it-chat-history.txt",
    mime="text/plain",
    use_container_width=True,
)

st.sidebar.markdown("### Recent")
st.sidebar.markdown("<div style='border-top:1px solid var(--border); margin: -0.4rem 0 0.6rem 0;'></div>", unsafe_allow_html=True)

if st.session_state.session_search_mode:
    if st_keyup is not None:
        with st.sidebar:
            st.session_state.session_search_query = st_keyup(
                "Search sessions",
                value=st.session_state.session_search_query,
                key="session_search_query_input",
                placeholder="Search by topic or keyword...",
            ) or ""
    else:
        st.session_state.session_search_query = st.sidebar.text_input(
            "Search sessions",
            value=st.session_state.session_search_query,
            key="session_search_query",
            placeholder="Search by topic or keyword...",
            label_visibility="collapsed",
        )
        st.sidebar.caption("Install st-keyup for instant key-by-key search updates.")
else:
    st.session_state.session_search_query = ""

search_text = st.session_state.session_search_query.lower().strip()
recent_sessions = st.session_state.chat_sessions
if search_text:
    recent_sessions = [
        session
        for session in st.session_state.chat_sessions
        if search_text in format_session_title(session).lower()
    ]

if recent_sessions:
    for session in recent_sessions[:20]:
        is_active = session["id"] == st.session_state.active_session_id
        preview = get_session_preview(session["messages"])
        session_title = format_session_title(session)
        button_label = f"{'● ' if is_active else '○ '}{session_title}"
        if st.sidebar.button(button_label, key=f"session_{session['id']}", use_container_width=True):
            set_active_session(session["id"])
            st.rerun()
        st.sidebar.caption(preview[:72] + ("..." if len(preview) > 72 else ""))
else:
    st.sidebar.caption("No recent chats found.")

# ------------------------------------------------------------------
# FRONTEND UI: CHAT COMPONENT
# ------------------------------------------------------------------
active_meta = AGENT_META[st.session_state.active_agent]
is_fresh_chat = len(get_active_messages()) <= 1
greeting_class = "greeting-wrap is-fresh" if is_fresh_chat else "greeting-wrap"

st.markdown(
    f"""
    <div class="{greeting_class}">
        <div class="greeting-title">How can I help with IT today<span class="accent-dot">?</span></div>
        <p class="greeting-subtitle">Ask a question and compare how exact match, keyword routing, and semantic search each respond — using the same {len(df):,}-entry knowledge base.</p>
        <div class="status-row">
            <span class="status-chip"><span class="swatch"></span>Using&nbsp;<b>{active_meta['label']}</b></span>
            <span class="status-chip">{len(df):,}&nbsp;entries indexed</span>
            <span class="status-chip">Sessions saved</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

agent_cols = st.columns(3, gap="medium")
for column, agent_name in zip(agent_cols, AGENT_OPTIONS):
    meta = AGENT_META[agent_name]
    with column:
        is_active = st.session_state.active_agent == agent_name
        card_style = "primary" if is_active else "secondary"
        if st.button(
            f"{meta['icon']} {meta['label']}",
            key=f"agent_btn_{agent_name}",
            use_container_width=True,
            type=card_style,
        ):
            set_agent(agent_name)
            st.rerun()

st.markdown(
    f"<p class='agent-desc-text'>{active_meta['description']}</p>",
    unsafe_allow_html=True,
)

comparison_mode = st.sidebar.toggle(
    "Algorithm Comparison Mode",
    value=False,
    help="Run the same query through Exact Match, Pattern Match, and Machine Learning side-by-side.",
)

chat_container = st.container()
with chat_container:
    active_messages = get_active_messages()
    for message in active_messages:
        render_chat_message(message)

# Capture user interactions
if user_query := st.chat_input("Ask an IT question..."):
    active_session = get_active_session()
    user_message = {"id": str(uuid4()), "role": "user", "content": user_query}
    active_session["messages"].append(user_message)
    update_active_session_title(user_query)
    render_chat_message(user_message)

    clean_query = user_query.lower().strip()
    active_agent = st.session_state.active_agent
    answer_text = ""
    meta: dict = {}

    if comparison_mode:
        comparison_specs = [
            ("Exact Match", lambda: run_exact_match(clean_query)),
            ("Pattern Match", lambda: run_pattern_match(clean_query)),
            ("Machine Learning", lambda: run_machine_learning(user_query)),
        ]
        comparison_results = []

        for label, resolver in comparison_specs:
            start = perf_counter()
            result_answer, result_meta = resolver()
            latency_ms = (perf_counter() - start) * 1000
            comparison_results.append(
                {
                    "label": label,
                    "answer": result_answer,
                    "meta": result_meta,
                    "latency_ms": latency_ms,
                }
            )

        answer_text = "Comparison results for your query:"
        meta = {"comparison_results": comparison_results}

    # ------------------------------------------------------------------
    # CORE ROUTING ROUTINES BY ALGORITHM
    # ------------------------------------------------------------------

    # 【Single-agent routing when comparison mode is off】
    if not comparison_mode:
        # 【Version 1: Exact String Matching】
        if active_agent == "Version 1: Exact Match Agent":
            answer_text, meta = run_exact_match(clean_query)

        # 【Version 2: Pattern & Keyword Matching with Safety Checks】
        elif active_agent == "Version 2: Pattern Matching Agent":
            answer_text, meta = run_pattern_match(clean_query)

        # 【Version 3: TF-IDF + Cosine Similarity Retrieval】
        elif active_agent == "Version 3: Machine Learning Agent":
            answer_text, meta = run_machine_learning(user_query)

    assistant_message = {
        "id": str(uuid4()),
        "role": "assistant",
        "content": answer_text,
        "agent": active_agent,
        "meta": meta,
    }
    active_session["messages"].append(assistant_message)
    render_chat_message(assistant_message)