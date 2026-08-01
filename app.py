import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import re
from uuid import uuid4
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

st.set_page_config(page_title="Multi-Algorithm IT Chatbot", page_icon="🤖", layout="wide")

CUSTOM_CSS = """
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(78, 205, 196, 0.14), transparent 28%),
            radial-gradient(circle at top right, rgba(58, 123, 213, 0.10), transparent 24%),
            linear-gradient(180deg, #f7fbfc 0%, #eef4f7 100%);
        color: #12303a;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1240px;
    }

    .hero-card, .surface-card {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(17, 48, 58, 0.08);
        border-radius: 22px;
        box-shadow: 0 16px 40px rgba(16, 32, 41, 0.08);
        backdrop-filter: blur(12px);
    }

    .hero-card {
        padding: 1.4rem 1.5rem;
        margin-bottom: 1rem;
    }

    .eyebrow {
        font-size: 0.8rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #2d6a73;
        font-weight: 700;
    }

    .hero-title {
        font-size: 2.15rem;
        line-height: 1.1;
        margin: 0.25rem 0 0.5rem 0;
        color: #0f2730;
        font-weight: 800;
    }

    .hero-subtitle {
        color: #45606a;
        font-size: 1rem;
        margin: 0;
    }

    .metric-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.55rem 0.8rem;
        border-radius: 999px;
        background: rgba(18, 48, 58, 0.05);
        color: #163943;
        font-weight: 600;
        margin-right: 0.45rem;
        margin-top: 0.35rem;
    }

    .agent-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
        margin: 0.5rem 0 1rem 0;
    }

    .agent-card {
        border: 1px solid rgba(17, 48, 58, 0.10);
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.9);
        padding: 0.95rem;
        text-align: left;
        width: 100%;
    }

    .agent-card strong {
        display: block;
        margin-bottom: 0.25rem;
        color: #102b33;
    }

    .agent-card small {
        color: #54717a;
        line-height: 1.35;
    }

    div[data-testid="stButton"] > button {
        border-radius: 14px;
        border: 1px solid rgba(17, 48, 58, 0.10);
        padding: 0.65rem 1rem;
        font-weight: 600;
    }

    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, #12303a 0%, #1e5f72 100%);
        color: white;
        border: none;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 18px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.95), rgba(240, 246, 248, 0.95));
        border-right: 1px solid rgba(17, 48, 58, 0.08);
        color: #12303a;
    }

    section[data-testid="stSidebar"] * {
        color: #12303a;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

AGENT_OPTIONS = [
    "Version 1: Exact Match Agent",
    "Version 2: Pattern Matching Agent",
    "Version 3: Machine Learning Agent",
]

AGENT_META = {
    "Version 1: Exact Match Agent": {
        "label": "Exact Match",
        "description": "Best for exact FAQ wording. Fast, strict, and deterministic.",
        "logic": "String comparison verification (`==`). The user query must match a database question exactly (case-insensitive, trimmed) to fetch a result.",
        "accent": "#2d6a73",
        "icon": "🎯",
    },
    "Version 2: Pattern Matching Agent": {
        "label": "Pattern Match",
        "description": "Best for keyword-heavy questions and quick topic routing.",
        "logic": "Fuzzy token inclusion mapping (`in` operator). The pipeline sweeps user inputs for high-frequency technical key-terms to trigger responses.",
        "accent": "#d98c2b",
        "icon": "🧭",
    },
    "Version 3: Machine Learning Agent": {
        "label": "Machine Learning",
        "description": "Best for flexible wording and broader semantic matching.",
        "logic": "TF-IDF text vectorization matrix combined with a Linear Support Vector Classifier (`LinearSVC`). Capable of semantic generalization and predictive inference.",
        "accent": "#2f7dd1",
        "icon": "🧠",
    },
}

if "active_agent" not in st.session_state:
    st.session_state.active_agent = AGENT_OPTIONS[2]

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = [
        {
            "id": str(uuid4()),
            "title": "LOQ Laptop Monitor Not Working",
            "messages": [
                {
                    "role": "assistant",
                    "content": "Hello. I am your automated IT support agent. Feel free to submit a query regarding wifi setups, password policies, or system issues.",
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
                "role": "assistant",
                "content": "Hello. I am your automated IT support agent. Feel free to submit a query regarding wifi setups, password policies, or system issues.",
            }
        ],
    }
    st.session_state.chat_sessions.insert(0, new_session)
    st.session_state.active_session_id = new_session["id"]

def clear_chat_history() -> None:
    get_active_session()["messages"] = [
        {
            "role": "assistant",
            "content": "Hello. I am your automated IT support agent. Feel free to submit a query regarding wifi setups, password policies, or system issues.",
        }
    ]

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

# ------------------------------------------------------------------
# BACKEND: DATA LOADING & ML INITIALIZATION
# ------------------------------------------------------------------
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv("it_support_dataset.csv", engine="python", on_bad_lines="skip")
    except FileNotFoundError:
        return None
    df = df.dropna(subset=["question", "answer"])
    df["question"] = df["question"].astype(str).str.strip()
    df["answer"] = df["answer"].astype(str).str.strip()
    df = df[(df["question"] != "") & (df["answer"] != "")]
    return df

df = load_data()

if df is None:
    st.error("⚠️ 'it_support_dataset.csv' not found! Please verify it is saved in your project root directory.")
    st.stop()

# Split and train historical reference data for Version 3 (Machine Learning)
X = df['question']
y = df['answer']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

ml_model = LinearSVC()
ml_model.fit(X_train_tfidf, y_train)

# Calculate Evaluation Report for Version 3
predictions = ml_model.predict(X_test_tfidf)
report_dict = classification_report(y_test, predictions, output_dict=True, zero_division=0)

# ------------------------------------------------------------------
# SIDEBAR: CHAT SESSION LIST
# ------------------------------------------------------------------
st.sidebar.header("Chat Sessions")
st.sidebar.caption("New Chat, Search Chat, and recent conversations.")

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

st.sidebar.markdown("### Recent")

if st.session_state.session_search_mode:
    st.session_state.session_search_query = st.sidebar.text_input(
        "Search sessions",
        value=st.session_state.session_search_query,
        placeholder="Search by topic or keyword...",
        label_visibility="collapsed",
    )
else:
    st.session_state.session_search_query = ""

search_text = st.session_state.session_search_query.lower().strip()
recent_sessions = st.session_state.chat_sessions
if search_text:
    recent_sessions = [
        session
        for session in st.session_state.chat_sessions
        if search_text in session["title"].lower()
        or search_text in get_session_preview(session["messages"]).lower()
    ]

if recent_sessions:
    for session in recent_sessions[:20]:
        is_active = session["id"] == st.session_state.active_session_id
        preview = get_session_preview(session["messages"])
        session_title = format_session_title(session)
        button_label = f"{'▶ ' if is_active else ''}{session_title}"
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

st.markdown(
    f"""
    <div class="hero-card">
        <div class="eyebrow">IT Support Chatbot</div>
        <div class="hero-title">Clean, focused support across three agent styles</div>
        <p class="hero-subtitle">Use the switcher to move between exact match, keyword routing, and machine learning without losing the conversation.</p>
        <div>
            <span class="metric-chip">{active_meta['icon']} Active: {active_meta['label']}</span>
            <span class="metric-chip">📚 {len(df):,} entries</span>
            <span class="metric-chip">💬 Persistent chat history</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Choose an agent")
agent_cols = st.columns(3, gap="medium")
for column, agent_name in zip(agent_cols, AGENT_OPTIONS):
    meta = AGENT_META[agent_name]
    with column:
        card_style = "primary" if st.session_state.active_agent == agent_name else "secondary"
        if st.button(
            f"{meta['icon']} {meta['label']}",
            key=f"agent_btn_{agent_name}",
            use_container_width=True,
            type=card_style,
        ):
            set_agent(agent_name)
            st.rerun()
        st.markdown(
            f"<div class='surface-card' style='padding: 0.9rem 1rem; margin-top: -0.25rem;'><strong>{agent_name}</strong><small>{meta['description']}</small></div>",
            unsafe_allow_html=True,
        )

st.caption(f"Active processing pipeline: {st.session_state.active_agent}")

chat_container = st.container()
with chat_container:
    active_messages = get_active_messages()
    for message in active_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Capture user interactions
if user_query := st.chat_input("Ask an IT question..."):
    active_session = get_active_session()
    active_session["messages"].append({"role": "user", "content": user_query})
    update_active_session_title(user_query)
    with st.chat_message("user"):
        st.write(user_query)
        
    clean_query = user_query.lower().strip()
    system_response = ""
    
    # ------------------------------------------------------------------
    # CORE ROUTING ROUTINES BY ALGORITHM
    # ------------------------------------------------------------------
    
    # 【Version 1: Exact String Matching】
    if st.session_state.active_agent == "Version 1: Exact Match Agent":
        matched_rows = df[df['question'].str.lower().str.strip() == clean_query]
        if not matched_rows.empty:
            system_response = matched_rows.iloc[0]['answer']
        else:
            system_response = "❌ **[Version 1 Error]** Intent mapping failed. Exact Match rules state that input characters must mimic a database entry exactly, with no variance."

    # 【Version 2: Pattern & Keyword Matching with Safety Checks】
    elif st.session_state.active_agent == "Version 2: Pattern Matching Agent":
        if "wifi" in clean_query or "wi-fi" in clean_query or "connection" in clean_query:
            matches = df[df['question'].str.lower().str.contains("wifi")]
            system_response = matches.iloc[0]['answer'] if not matches.empty else "⚠️ Keyword flagged, but no corresponding knowledge entries matched inside the dataset."
            
        elif "password" in clean_query or "reset" in clean_query or "lock" in clean_query:
            matches = df[df['question'].str.lower().str.contains("password")]
            system_response = matches.iloc[0]['answer'] if not matches.empty else "⚠️ Keyword flagged, but no corresponding knowledge entries matched inside the dataset."
            
        elif "laptop" in clean_query or "screen" in clean_query or "hardware" in clean_query or "display" in clean_query:
            matches = df[df['question'].str.lower().str.contains("laptop")]
            system_response = matches.iloc[0]['answer'] if not matches.empty else "⚠️ Keyword flagged, but no corresponding knowledge entries matched inside the dataset."
            
        elif "outlook" in clean_query or "email" in clean_query or "phone" in clean_query:
            matches = df[df['question'].str.lower().str.contains("outlook")]
            system_response = matches.iloc[0]['answer'] if not matches.empty else "⚠️ Keyword flagged, but no corresponding knowledge entries matched inside the dataset."
            
        else:
            system_response = "⚠️ **[Version 2 Warning]** Pattern recognition failed. The input did not contain any predefined IT key-terms (e.g., wifi, password, laptop, outlook)."

    # 【Version 3: Natural Language Machine Learning】
    elif st.session_state.active_agent == "Version 3: Machine Learning Agent":
        query_tfidf = vectorizer.transform([user_query])
        system_response = ml_model.predict(query_tfidf)[0]

    # Post processing output response stream
    active_session["messages"].append({"role": "assistant", "content": system_response})
    with st.chat_message("assistant"):
        st.write(system_response)