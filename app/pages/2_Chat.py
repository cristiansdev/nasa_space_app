import streamlit as st
import pandas as pd
from google import genai
from sentence_transformers import SentenceTransformer
import faiss
import boto3
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Chat con RAG", page_icon="💬", layout="wide")
st.title("💬 Chat con RAG - NASA Papers")

RAG_BUCKET = os.getenv("RAG_BUCKET", "nasa-rag-rag")
RAG_PREFIX = os.getenv("RAG_PREFIX", "index/latest")
MODEL_NAME = "all-MiniLM-L6-v2"

s3 = boto3.client("s3")

@st.cache_resource
def load_index_and_meta(bucket, prefix):
    idx_path = f"/tmp/{prefix.replace('/', '_')}_index.faiss"
    meta_path = f"/tmp/{prefix.replace('/', '_')}_meta.parquet"

    s3.download_file(bucket, f"{prefix}/index.faiss", idx_path)
    s3.download_file(bucket, f"{prefix}/meta.parquet", meta_path)

    index = faiss.read_index(idx_path)
    df = pd.read_parquet(meta_path)
    return index, df

index, df = load_index_and_meta(RAG_BUCKET, RAG_PREFIX)
model = SentenceTransformer(MODEL_NAME)
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def retrieve(query, k=3):
    q_emb = model.encode([query])
    D, I = index.search(q_emb, k)
    results = []
    for idx in I[0]:
        row = df.iloc[int(idx)]
        results.append(f"**{row['Title']}**\n\n{row['Abstract']}")
    return "\n\n---\n\n".join(results)

st.markdown("Pregunta sobre biología espacial, microgravedad, o fisiología en el espacio 🚀")

query = st.text_input("🧠 Escribe tu pregunta:")
if query:
    context = retrieve(query)
    prompt = f"Contexto:\n{context}\n\nPregunta:\n{query}\n\nResponde en español claro y cita los títulos relevantes."
    with st.spinner("Consultando Gemini..."):
        try:
            response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            st.markdown(response.text)
        except Exception as e:
            st.error(f"Error con Gemini: {e}")
