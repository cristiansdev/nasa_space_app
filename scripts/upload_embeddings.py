#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construye embeddings + índice FAISS desde abstracts_full.csv y publica en S3.
Archivos generados:
  index.faiss, meta.parquet, embeddings.npz, manifest.json, model.txt
"""

import os, io, json, time, numpy as np, pandas as pd
from dotenv import load_dotenv
import boto3, faiss
from sentence_transformers import SentenceTransformer

load_dotenv()
BUCKET = os.getenv("RAG_BUCKET", "nasa-rag-rag")
PREFIX = os.getenv("RAG_PREFIX", "index/v1")  # e.g., index/v1
MODEL_NAME = os.getenv("RAG_EMBED_MODEL", "all-MiniLM-L6-v2")

s3 = boto3.client("s3")

def ensure_bucket(bucket: str):
    region = boto3.session.Session().region_name
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"🪣 Bucket '{bucket}' OK.")
    except Exception:
        print(f"⚠️ Creando bucket {bucket} en {region}...")
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(Bucket=bucket,
                             CreateBucketConfiguration={"LocationConstraint": region})
        print("✅ Creado.")

def upload_path(bucket: str, key: str, path: str):
    s3.upload_file(path, bucket, key)
    print(f"↑ s3://{bucket}/{key}")

def main():
    print("📘 Cargando datos...")
    df = pd.read_csv("abstracts_full.csv")
    df = df[df["Error"].isna()]
    df = df.dropna(subset=["Abstract"])
    meta = df[["Title", "Link", "Abstract"]].reset_index(drop=True)

    print("🧠 Creando embeddings...")
    model = SentenceTransformer(MODEL_NAME)
    X = model.encode(meta["Abstract"].astype(str).tolist(), show_progress_bar=True).astype(np.float32)

    print("📦 Construyendo índice FAISS...")
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)

    print("💾 Guardando artefactos...")
    os.makedirs("artifacts", exist_ok=True)
    faiss.write_index(index, "artifacts/index.faiss")
    meta.to_parquet("artifacts/meta.parquet", index=False)
    np.savez_compressed("artifacts/embeddings.npz", x=X)
    with open("artifacts/model.txt", "w") as f:
        f.write(MODEL_NAME)
    manifest = {
        "version": PREFIX.split("/")[-1],
        "model": MODEL_NAME,
        "count": int(len(meta)),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    with open("artifacts/manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"☁️ Publicando en s3://{BUCKET}/{PREFIX}")
    ensure_bucket(BUCKET)
    upload_path(BUCKET, f"{PREFIX}/index.faiss", "artifacts/index.faiss")
    upload_path(BUCKET, f"{PREFIX}/meta.parquet", "artifacts/meta.parquet")
    upload_path(BUCKET, f"{PREFIX}/embeddings.npz", "artifacts/embeddings.npz")
    upload_path(BUCKET, f"{PREFIX}/manifest.json", "artifacts/manifest.json")
    upload_path(BUCKET, f"{PREFIX}/model.txt", "artifacts/model.txt")
    print("✅ Listo.")

if __name__ == "__main__":
    main()
