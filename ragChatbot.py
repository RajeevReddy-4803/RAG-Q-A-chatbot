import streamlit as st
import pandas as pd
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

st.set_page_config(page_title="RAG Q&A Chatbot", layout="centered")
st.title("🧠 RAG Q&A Chatbot using 🤗 Transformers (No OpenAI)")

# -------------------------
# CONFIGS
# -------------------------
DATA_PATH = "Training Dataset.csv"
INDEX_FILE = "faiss_index.pkl"
EMBEDDINGS_FILE = "embeddings.npy"
CONTEXT_FILE = "contexts.pkl"

# -------------------------
# Load Dataset
# -------------------------
@st.cache_data
def load_csv():
    df = pd.read_csv(DATA_PATH)
    df.fillna("Unknown", inplace=True)
    return df

df = load_csv()

@st.cache_data
def build_contexts(df):
    return df.astype(str).apply(lambda row: " | ".join(row), axis=1).tolist()

contexts = build_contexts(df)

# -------------------------
# Sentence Embeddings + FAISS Index (Cached with .pkl)
# -------------------------
@st.cache_resource
def embed_and_index(contexts):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    if os.path.exists(INDEX_FILE) and os.path.exists(EMBEDDINGS_FILE) and os.path.exists(CONTEXT_FILE):
        with open(INDEX_FILE, "rb") as f:
            index = pickle.load(f)
        embeddings = np.load(EMBEDDINGS_FILE)
        with open(CONTEXT_FILE, "rb") as f:
            stored_contexts = pickle.load(f)

        # Ensure embeddings match context
        if len(stored_contexts) != len(contexts):
            raise ValueError("Stored contexts do not match current dataset. Delete .pkl/.npy to refresh.")

    else:
        embeddings = model.encode(contexts, convert_to_numpy=True, show_progress_bar=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        with open(INDEX_FILE, "wb") as f:
            pickle.dump(index, f)
        np.save(EMBEDDINGS_FILE, embeddings)
        with open(CONTEXT_FILE, "wb") as f:
            pickle.dump(contexts, f)

    return model, index, embeddings

embedder, index, embedding_vectors = embed_and_index(contexts)

# -------------------------
# Load HuggingFace QA Model
# -------------------------
@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

qa_model = load_qa_pipeline()

# -------------------------
# Sidebar Controls
# -------------------------
st.sidebar.header("🔧 Settings")
top_k = st.sidebar.slider("Top K Documents", min_value=1, max_value=10, value=3)

# -------------------------
# User Question Input
# -------------------------
user_question = st.text_input("Ask a question about the dataset:")

def get_top_contexts(question, k=3):
    query_vector = embedder.encode([question], convert_to_numpy=True)
    _, indices = index.search(query_vector, k)
    return [contexts[i] for i in indices[0]]

if user_question:
    with st.spinner("🔍 Searching for answers..."):
        try:
            top_contexts = get_top_contexts(user_question, k=top_k)
            combined_context = "\n".join(top_contexts)
            result = qa_model(question=user_question, context=combined_context)
            st.success("✅ Answer:")
            st.write(result['answer'])
            st.write(f"Confidence: {result['score']:.2f}")

            with st.expander("🔎 Top Contexts Used"):
                for i, ctx in enumerate(top_contexts, 1):
                    st.markdown(f"**Context {i}:** {ctx}")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
