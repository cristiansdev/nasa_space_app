# app/Home.py
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
import streamlit as st
from dotenv import load_dotenv
from raglib.s3_utils import list_versions, ensure_bucket

load_dotenv()

st.set_page_config(page_title="NASA Space Biology RAG", page_icon="🚀", layout="wide")
st.title("🚀 NASA Space Biology RAG System")

st.write(
    "Este sistema permite consultar papers con RAG (embeddings en S3) y Gemini como modelo. "
    "Ve al Treemap para explorar categorías y chatear sobre una temática."
)

# Diagnóstico mínimo
bucket = os.getenv("RAG_BUCKET", "nasa-rag-rag")
prefix = os.getenv("RAG_PREFIX", "index/v5_categories")

left, right = st.columns(2)

with left:
    try:
        ensure_bucket(bucket)
        versions = list_versions(bucket, base_folder="index/")
        if versions:
            st.success(f"Conexión a S3 OK. Versiones detectadas: {', '.join(versions[-5:])}")
        else:
            st.warning("Conexión a S3 OK, pero no se detectaron versiones en 'index/'.")
    except Exception as e:
        st.error(f"Error S3: {e}")

with right:
    if os.getenv("GOOGLE_API_KEY"):
        st.success("GOOGLE_API_KEY detectada.")
    else:
        st.warning("No se detectó GOOGLE_API_KEY en tu entorno.")

st.markdown("### Navegación")
st.markdown("- 📊 **Treemap**: Dashboard de categorías y chat por temática.")
st.markdown("- 💬 **Chat**: (página secundaria, si la usas).")
st.caption("Desarrollado para NASA Space Apps • Demo")
