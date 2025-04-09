import streamlit as st
import pandas as pd
import numpy as np
import faiss
from bertopic import BERTopic
import zipfile, os, gdown
from sentence_transformers import SentenceTransformer

# --- Load Data & Models ---
@st.cache_resource
def load_resources():
    # 📁 Define paths and Google Drive IDs
    model_folder = "bertopic_model_simple"
    embedding_folder = "embedding_model"
    zip_model = f"{model_folder}.zip"
    zip_embed = f"{embedding_folder}.zip"
    model_drive_id = "1oTBuwKboDFazlrAejk7911RtzTdemCNz"
    csv_drive_id = "1izjvaemSRnmEfDdG4aGF8yC8XEI_NVkB"
    embed_drive_id = "1BhjEi2SSn1hEIEEqrzPuNmbH93C0lewm"

    # 🔽 Download and unzip BERTopic model
    if not os.path.exists(model_folder):
        with st.spinner("Downloading BERTopic model..."):
            url = f"https://drive.google.com/uc?id={model_drive_id}"
            gdown.download(url, zip_model, quiet=False)
            with zipfile.ZipFile(zip_model, "r") as zip_ref:
                zip_ref.extractall(".")

    # 🔽 Download and unzip embedding model
    if not os.path.exists(embedding_folder):
        with st.spinner("Downloading embedding model..."):
            url = f"https://drive.google.com/uc?id={embed_drive_id}"
            gdown.download(url, zip_embed, quiet=False)
            with zipfile.ZipFile(zip_embed, "r") as zip_ref:
                zip_ref.extractall(".")

    # 🔽 Download CSV if missing
    if not os.path.exists("train_topic_output.csv"):
        with st.spinner("Downloading topic output CSV..."):
            url = f"https://drive.google.com/uc?id={csv_drive_id}"
            gdown.download(url, "train_topic_output.csv", quiet=False)

    # ✅ Load resources
    df = pd.read_csv("train_topic_output.csv")
    topic_model = BERTopic.load(model_folder)
    embedding_model = topic_model.embedding_model

    # ✅ Load embeddings
    texts = df['cleaned_text'].astype(str).tolist()
    doc_embeddings = np.load("doc_embeddings.npy").astype("float32")
    faiss.normalize_L2(doc_embeddings)

    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings)

    return df, topic_model, embedding_model, index, texts

# --- Initialize
df, topic_model, embedding_model, index, texts = load_resources()

# --- UI ---
st.title("🔍 Semantic Article Recommender")
st.image("Logo_LUISS.png", width=250)

st.markdown("""
This app helps you find semantically similar academic articles using **BERTopic** and **FAISS**.  
Developed for the *Data Science in Action* course at **LUISS University of Rome**.  
👩‍💻 *By Chiara Barontini, Daniele Biggi, Michele Baldo*
""")

query = st.text_input("Enter your search query", placeholder="e.g. green supply chain resilience")
top_k = st.slider("Number of results", min_value=3, max_value=20, value=5)

if query:
    with st.spinner("Searching..."):
        query_vec = topic_model._embedding_model.embed([query])[0].astype("float32").reshape(1, -1)
        faiss.normalize_L2(query_vec)
        scores, indices = index.search(query_vec, top_k)

        st.markdown(f"### 🔎 Top {top_k} Matches for: *{query}*")
        for idx, score in zip(indices[0], scores[0]):
            row = df.iloc[idx]
            st.markdown(f"**{row['display_name']}**")
            st.markdown(f"**Topic:** {row['topic']} | **Score:** {score:.3f}")
            st.write(texts[idx][:300] + "...")
            st.markdown("---")
