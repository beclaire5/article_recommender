import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import zipfile, os, gdown

# --- Load Data & Models ---
@st.cache_resource
def load_resources():
    # 📁 Check for model folder and CSV
    model_folder = "bertopic_model_simple"
    zip_file = f"{model_folder}.zip"
    model_drive_id = "1oTBuwKboDFazlrAejk7911RtzTdemCNz"
    csv_drive_id = "1izjvaemSRnmEfDdG4aGF8yC8XEI_NVkB"

    # 🔽 Download and unzip model if missing
    if not os.path.exists(model_folder):
        with st.spinner("Downloading BERTopic model from Google Drive..."):
            url = f"https://drive.google.com/uc?id={model_drive_id}"
            gdown.download(url, zip_file, quiet=False)
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(".")

    # 🔽 Download CSV if missing
    if not os.path.exists("train_topic_output.csv"):
        with st.spinner("Downloading topic output CSV from Google Drive..."):
            csv_url = f"https://drive.google.com/uc?id={csv_drive_id}"
            gdown.download(csv_url, "train_topic_output.csv", quiet=False)

    # 📦 Load model and CSV
    df = pd.read_csv("train_topic_output.csv")
    topic_model = BERTopic.load(model_folder)
    embedding_model = SentenceTransformer("paraphrase-MiniLM-L12-v2")
    topic_model.embedding_model = embedding_model

    # 🧠 Load embeddings
    texts = df['cleaned_text'].astype(str).tolist()
    doc_embeddings = np.load("doc_embeddings.npy").astype("float32")
    faiss.normalize_L2(doc_embeddings)

    # 🔍 Build FAISS index
    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings)

    return df, topic_model, embedding_model, index, texts

# --- Initialize
df, topic_model, embedding_model, index, texts = load_resources()

# --- Streamlit UI ---
st.title("🔍 Semantic Article Recommender")
# 🇮🇹 LUISS University Logo
st.markdown(
    "<div style='text-align: center;'>"
    "<img src='https://upload.wikimedia.org/wikipedia/commons/2/27/Luiss_logo.png' width='250'>"
    "</div>",
    unsafe_allow_html=True
)

st.markdown("### 👩‍💻 Group Members")
st.markdown("- Chiara Barontini  \n- Daniele Biggi  \n- Michele Baldo")

st.markdown("### 📘 Project Summary")
st.markdown("""
This app helps you find semantically similar academic articles using **BERTopic** and **FAISS**.  
It was developed for our Data Science in Action course at **LUISS University of Rome**.
""")

query = st.text_input("Enter your search query", placeholder="e.g. green supply chain resilience")
top_k = st.slider("Number of results", min_value=3, max_value=20, value=5)

if query:
    with st.spinner("Searching..."):
        query_vec = embedding_model.encode([query])[0].astype("float32").reshape(1, -1)
        faiss.normalize_L2(query_vec)
        scores, indices = index.search(query_vec, top_k)

        st.markdown(f"### 🔎 Top {top_k} Matches for: *{query}*")
        for idx, score in zip(indices[0], scores[0]):
            row = df.iloc[idx]
            st.markdown(f"**{row['display_name']}**")
            st.markdown(f"**Topic:** {row['topic']} | **Score:** {score:.3f}")
            st.write(texts[idx][:300] + "...")
            st.markdown("---")
