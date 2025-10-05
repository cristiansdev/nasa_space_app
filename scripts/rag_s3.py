#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chat RAG desde consola:
  - Carga S3 (última versión si RAG_PREFIX=index/latest)
  - Recupera con FAISS y responde con Gemini
"""

import os, re, io, pandas as pd, faiss, boto3
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai

load_dotenv()
RAG_BUCKET = os.getenv("RAG_BUCKET", "nasa-rag-rag")
RAG_PREFIX = os.getenv("RAG_PREFIX", "index/latest")
CACHE_DIR  = os.getenv("RAG_CACHE_DIR", "/tmp/rag")
s3 = boto3.client("s3")
_V_RE = re.compile(r"^v(\d+)$")

def _latest_prefix(bucket: str, base_folder: str = "index/") -> str:
    pages = s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=base_folder, Delimiter="/")
    vers = []
    for p in pages:
        for cp in p.get("CommonPrefixes", []):
            sub = cp["Prefix"].strip("/").split("/")[-1]
            if _V_RE.match(sub): vers.append(sub)
    if not vers: raise RuntimeError("No hay versiones")
    vers.sort(key=lambda v:int(_V_RE.match(v).group(1)))
    return f"{base_folder}{vers[-1]}"

def _resolve_prefix(bucket: str, prefix_env: str | None) -> str:
    if not prefix_env or prefix_env.endswith("/latest") or prefix_env == "index":
        base_folder = (prefix_env.split("/")[0] if prefix_env else "index") + "/"
        resolved = _latest_prefix(bucket, base_folder)
        print(f"🧭 Usando última versión: s3://{bucket}/{resolved}")
        return resolved
    return prefix_env

def _download(bucket: str, key: str) -> bytes:
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()

# Resolver prefijo
RAG_PREFIX = _resolve_prefix(RAG_BUCKET, RAG_PREFIX)

print(f"☁️ Cargando artefactos desde s3://{RAG_BUCKET}/{RAG_PREFIX} ...")
idx_bytes  = _download(RAG_BUCKET, f"{RAG_PREFIX}/index.faiss")
meta_bytes = _download(RAG_BUCKET, f"{RAG_PREFIX}/meta.parquet")

index = faiss.read_index(io.BytesIO(idx_bytes))
df    = pd.read_parquet(io.BytesIO(meta_bytes))

try:
    model_name = _download(RAG_BUCKET, f"{RAG_PREFIX}/model.txt").decode().strip() or "all-MiniLM-L6-v2"
except Exception:
    model_name = "all-MiniLM-L6-v2"

print(f"🧠 Cargando modelo de embeddings: {model_name}")
model = SentenceTransformer(model_name)

def retrieve(query: str, k: int = 5):
    if not query.strip(): return []
    k = max(1, min(k, len(df)))
    q = model.encode([query])
    D, I = index.search(q, k)
    out = []
    for idx in I[0]:
        row = df.iloc[int(idx)]
        out.append({"Title": row.get("Title",""), "Link": row.get("Link",""), "Abstract": row.get("Abstract","")})
    return out

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("Falta GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

def ask_gemini(query: str, ctx_docs: list[dict]) -> str:
    if not ctx_docs:
        return "No encontré contexto relevante."
    ctx = "\n\n".join([f"Título: {d['Title']}\nAbstract: {d['Abstract']}" for d in ctx_docs])
    prompt = (
        "Eres un asistente científico. Responde en español, conciso y claro, citando títulos cuando apliquen. "
        f"Contexto:\n{ctx}\n\nPregunta: {query}"
    )
    try:
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return (resp.text or "").strip()
    except Exception as e:
        return f"Error con Gemini: {e}"

if __name__ == "__main__":
    print("💬 Chat listo. Escribe tu pregunta (Ctrl+C para salir).")
    while True:
        try:
            q = input("\n> ")
            docs = retrieve(q, k=5)
            ans = ask_gemini(q, docs)
            print("\n" + ans + "\n")
        except KeyboardInterrupt:
            print("\n👋 Bye!")
            break
