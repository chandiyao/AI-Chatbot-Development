import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

st.set_page_config(page_title="Multi-Algorithm IT Chatbot", page_icon="🤖", layout="wide")

# ------------------------------------------------------------------
# BACKEND: DATA LOADING & ML INITIALIZATION
# ------------------------------------------------------------------
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv("it_support_dataset.csv")
    except FileNotFoundError:
        return None
    df = df.dropna(subset=['question', 'answer'])
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
# SIDEBAR: AI AGENT SELECTOR CONTROL PANEL
# ------------------------------------------------------------------
st.sidebar.header("🤖 AI Agent Control Panel")
agent_choice = st.sidebar.radio(
    "Select the active algorithm framework:",
    [
        "Version 1: Exact Match Agent", 
        "Version 2: Pattern Matching Agent", 
        "Version 3: Machine Learning Agent"
    ]
)

st.sidebar.markdown("---")

# Provide algorithmic breakdowns for grading metrics
if agent_choice == "Version 1: Exact Match Agent":
    st.sidebar.info("💡 **Operational Logic:** String comparison verification (`==`). The user query must match a database question exactly (case-insensitive, trimmed) to fetch a result.")
elif agent_choice == "Version 2: Pattern Matching Agent":
    st.sidebar.info("💡 **Operational Logic:** Fuzzy token inclusion mapping (`in` operator). The pipeline sweeps user inputs for high-frequency technical key-terms to trigger responses.")
elif agent_choice == "Version 3: Machine Learning Agent":
    st.sidebar.success("💡 **Operational Logic:** TF-IDF text vectorization matrix combined with a Linear Support Vector Classifier (`LinearSVC`). Capable of semantic generalization and predictive inference.")
    
    st.sidebar.markdown("### 📊 V3 Machine Learning Metrics")
    metrics_df = pd.DataFrame(report_dict).transpose().iloc[:-3, :3]
    metrics_df.index = [idx[:15] + "..." if len(idx) > 15 else idx for idx in metrics_df.index]
    st.sidebar.dataframe(metrics_df.style.format("{:.2f}"))

# ------------------------------------------------------------------
# FRONTEND UI: CHAT COMPONENT
# ------------------------------------------------------------------
st.title("💻 Multi-Algorithm IT Triage Center")
st.caption(f"Active Processing Pipeline Engine: 🚀 {agent_choice}")

# Keep persistent chat logs across model switches
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "assistant", "content": "Hello. I am your automated IT support agent. Feel free to submit a query regarding wifi setups, password policies, or system issues."}]

# Render active interface lines
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Capture user interactions
if user_query := st.chat_input("Ask an IT question..."):
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)
        
    clean_query = user_query.lower().strip()
    system_response = ""
    
    # ------------------------------------------------------------------
    # CORE ROUTING ROUTINES BY ALGORITHM
    # ------------------------------------------------------------------
    
    # 【Version 1: Exact String Matching】
    if agent_choice == "Version 1: Exact Match Agent":
        matched_rows = df[df['question'].str.lower().str.strip() == clean_query]
        if not matched_rows.empty:
            system_response = matched_rows.iloc[0]['answer']
        else:
            system_response = "❌ **[Version 1 Error]** Intent mapping failed. Exact Match rules state that input characters must mimic a database entry exactly, with no variance."

    # 【Version 2: Pattern & Keyword Matching with Safety Checks】
    elif agent_choice == "Version 2: Pattern Matching Agent":
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
    elif agent_choice == "Version 3: Machine Learning Agent":
        query_tfidf = vectorizer.transform([user_query])
        system_response = ml_model.predict(query_tfidf)[0]

    # Post processing output response stream
    st.session_state.chat_history.append({"role": "assistant", "content": system_response})
    with st.chat_message("assistant"):
        st.write(system_response)