from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="IT HelpDesk AI", page_icon="💻", layout="wide")

# ------------------------------------------------------------------
# BACKEND: LOADING & INDEXING SUPPORT DATASET
# ------------------------------------------------------------------
@st.cache_resource
def load_support_knowledge_base():
    dataset_path = Path(__file__).resolve().parent / "it_support_dataset.csv"

    try:
        df = pd.read_csv(dataset_path, engine="python", on_bad_lines="skip")
    except FileNotFoundError:
        return None, None, None

    required_columns = {"question", "answer"}
    if not required_columns.issubset(df.columns):
        return None, None, None

    df = df.dropna(subset=["question", "answer"]).copy()
    df["question"] = df["question"].astype(str).str.strip()
    df["answer"] = df["answer"].astype(str).str.strip()
    df = df[df["question"] != ""]
    df = df[df["answer"] != ""]

    if df.empty:
        return None, None, None

    vectorizer = TfidfVectorizer(stop_words="english")
    question_matrix = vectorizer.fit_transform(df["question"])

    return df, vectorizer, question_matrix

knowledge_base_df, vectorizer, question_matrix = load_support_knowledge_base()

# ------------------------------------------------------------------
# FRONTEND: STREAMLIT UI DESIGN
# ------------------------------------------------------------------
st.title("💻 Enterprise IT Service Desk Agent")
st.caption("AI Retrieval Pipeline • Connected to it_support_dataset.csv")

if knowledge_base_df is None:
    st.error("⚠️ 'it_support_dataset.csv' was not found or does not contain question/answer columns.")
    st.stop()

# Sidebar metrics panel
st.sidebar.header("📊 Knowledge Base")
st.sidebar.metric("FAQ rows", len(knowledge_base_df))
st.sidebar.metric("Topics", knowledge_base_df["topic"].nunique() if "topic" in knowledge_base_df.columns else 0)
if "topic" in knowledge_base_df.columns:
    st.sidebar.write("Top topics")
    st.sidebar.dataframe(knowledge_base_df["topic"].value_counts().head(10).to_frame("count"))

# Chat System History Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "assistant", "content": "Hello. Describe your corporate IT or system issue, and I will route it to the correct queue."}]

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ACCEPT USER INPUT
if employee_query := st.chat_input("Type your IT issue..."):
    st.session_state.chat_history.append({"role": "user", "content": employee_query})
    with st.chat_message("user"):
        st.write(employee_query)
        
    # FIX 1: Intercept standard greetings before they confuse the ML model
    greetings = ["hi", "hello", "good morning", "hey", "good afternoon", "yo"]
    clean_query = employee_query.lower().strip().replace("!", "").replace(".", "")
    
    if clean_query in greetings:
        system_response = "👋 Hello! I am your automated IT triage agent. Please describe a specific technical, system, or product issue you are experiencing so I can route it to the correct department."
        
    else:
        # Retrieve the closest FAQ entry from the support knowledge base.
        query_vector = vectorizer.transform([employee_query])
        similarity_scores = cosine_similarity(query_vector, question_matrix).flatten()
        best_match_index = similarity_scores.argmax()
        best_match_score = similarity_scores[best_match_index]
        best_match_row = knowledge_base_df.iloc[best_match_index]

        if best_match_score < 0.15:
            system_response = (
                "I could not find a confident match in the support dataset. "
                "Please rephrase your issue with a few more keywords, or mention the exact system, app, or error message."
            )
        else:
            topic_label = f" [{best_match_row['topic']}]" if 'topic' in knowledge_base_df.columns else ""
            system_response = (
                f"**Closest match{topic_label}:** {best_match_row['question']}\n\n"
                f"**Answer:** {best_match_row['answer']}"
            )

    # Output bot response
    st.session_state.chat_history.append({"role": "assistant", "content": system_response})
    with st.chat_message("assistant"):
        st.write(system_response)