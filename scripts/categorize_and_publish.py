#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Categoriza los abstracts existentes (KMeans) y publica una nueva versión en S3 con:
- meta.parquet (agrega columna cluster_id)
- categories.json (label ≤2 palabras, summary, keywords por cluster)
- manifest.json

Uso:
  python3 -m scripts.categorize_and_publish \
    --bucket nasa-rag-rag \
    --base-prefix index/latest \
    --n-clusters 8 \
    --new-prefix index/v5_categories
"""

import os
import io
import re
import json
import time
import argparse
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

# asegurar imports de paquete local
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from raglib.s3_utils import (
    ensure_bucket,
    _s3_download, _s3_upload,
    download_bytes, upload_bytes,
    latest_version_prefix, next_version_prefix,
    _base_folder_from_prefix, resolve_base_prefix
)

load_dotenv()

MODEL_FALLBACK = "all-MiniLM-L6-v2"
CACHE_DIR = "/tmp/rag"


def parse_args():
    ap = argparse.ArgumentParser(description="Categoriza abstracts y publica categorías en S3.")
    ap.add_argument("--bucket", required=True, help="Bucket S3 (ej. nasa-rag-rag).")
    ap.add_argument("--base-prefix", required=True, help="Prefijo base (ej. index/latest o index/vN).")
    ap.add_argument("--n-clusters", type=int, default=8, help="Número de clusters KMeans (default=8).")
    ap.add_argument("--new-prefix", default=None, help="Prefijo NUEVO (ej. index/vX_categories). Si no se pasa, se calcula automáticamente.")
    return ap.parse_args()


# --------------------- Helpers Gemini / JSON robusto ---------------------
JSON_FENCE = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)
BRACES = re.compile(r"\{.*\}", re.S)
GENERIC = set("""
study studies research results framework dataset task approach method methods
analysis model models modeling experimental simulation review overview paper
system systems technique techniques application applications evaluation
space nasa biology biological science sciences journal issue volume data
""".split())

def _safe_parse_json(text: str) -> dict:
    text = (text or "").strip()
    m = JSON_FENCE.search(text)
    if m:
        return json.loads(m.group(1))
    m = BRACES.search(text)
    if m:
        return json.loads(m.group(0))
    return json.loads(text.replace("```", "").strip())

def _normalize_label(label: str, max_words: int = 2) -> str:
    # deja solo letras y espacios, colapsa espacios y limita a 1–2 palabras
    label = re.sub(r"[^A-Za-zÁÉÍÓÚÜÑáéíóúüñ\s\-]", " ", label or "")
    label = re.sub(r"\s+", " ", label).strip()
    words = [w for w in label.split() if w]
    if not words:
        return "Tema"
    words = words[:max_words]
    return " ".join(w.capitalize() for w in words)

def _candidate_label_ngrams(texts, topk=8):
    # candidatos locales por TF-IDF (1–2 gram), filtra genéricos
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            min_df=1,
            max_features=5000
        )
        X = vec.fit_transform(texts)
        scores = np.asarray(X.sum(axis=0)).ravel()
        vocab = np.array(vec.get_feature_names_out())
        order = scores.argsort()[::-1]
        cands = []
        for idx in order:
            term = vocab[idx]
            if len(term) < 4:
                continue
            parts = [p for p in term.split() if p not in GENERIC and len(p) > 2]
            if not parts:
                continue
            parts = parts[:2]
            cand = " ".join(parts).title()
            if cand.lower() in GENERIC:
                continue
            if cand not in cands:
                cands.append(cand)
            if len(cands) >= topk:
                break
        if not cands:
            cands = ["Microgravity", "Bone Loss", "Stem Cells", "Space Immunology"]
        return cands
    except Exception:
        # fallback si no está scikit-learn feature_extraction
        return ["Microgravity", "Bone Loss", "Stem Cells", "Space Immunology"]


def summarize_cluster_gemini(client, subset_df: pd.DataFrame, cid: int) -> dict:
    """
    Pide a Gemini: label (≤2 palabras), summary (2–3 frases), keywords (3–6 items).
    Devuelve dict robusto con saneo y fallbacks.
    """
    # candidatos locales (título+abstract) para anclar especificidad
    texts_for_cands = (subset_df["Title"].astype(str) + " " + subset_df["Abstract"].astype(str)).tolist()
    candidates = _candidate_label_ngrams(texts_for_cands, topk=8)

    # contexto acotado
    ctx = "\n\n".join([
        f"Título: {r.Title}\nAbstract: {str(r.Abstract)[:700]}"
        for _, r in subset_df.iterrows()
    ])

    prompt = (
        "Eres un bibliotecario científico. Dadas estas publicaciones y la lista de candidatos, "
        "elige el mejor **label** (1–2 palabras, nada genérico) y devuelve SOLO un bloque JSON "
        "con claves exactas `label`, `summary`, `keywords`.\n\n"
        f"Candidatos (elige o combina máximo 2 palabras): {candidates}\n\n"
        "Requisitos:\n"
        "- `label`: máximo 2 palabras, específicas (ej: 'Bone Loss', 'Stem Cells'). Evita genéricos como 'Research'.\n"
        "- `summary`: 2–3 frases en español, claras para no expertos.\n"
        "- `keywords`: lista de 3–6 términos.\n\n"
        f"Contexto:\n{ctx}\n\n"
        "Responde SOLO en un bloque:\n```json\n{ \"label\": \"...\", \"summary\": \"...\", \"keywords\": [\"...\"] }\n```"
    )

    label = _normalize_label(candidates[0], 2)
    summary = "Resumen no disponible."
    keywords = candidates[:3]

    try:
        from google import genai
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        raw = (resp.text or "").strip()
        data = _safe_parse_json(raw)

        if "label" in data and data["label"]:
            label = _normalize_label(str(data["label"]), 2)
        if "summary" in data and data["summary"]:
            summary = str(data["summary"]).strip()
        kw = data.get("keywords") or keywords
        if isinstance(kw, str):
            kw = [x.strip() for x in kw.split(",") if x.strip()]
        keywords = [k for k in kw if k and k.lower() not in GENERIC][:6]
    except Exception as e:
        print(f"⚠️ Cluster {cid}: fallo al resumir con Gemini → {e} (usando fallback)")

    return {"label": label, "summary": summary, "keywords": keywords}


def main():
    args = parse_args()

    # asegurar bucket y resolver base_prefix (latest -> vN)
    ensure_bucket(args.bucket)
    base_folder = _base_folder_from_prefix(args.base_prefix)
    args.base_prefix = resolve_base_prefix(args.bucket, args.base_prefix)
    print(f"🌤 Base resuelta: s3://{args.bucket}/{args.base_prefix}")

    # si no se pasó new_prefix → siguiente versión bajo misma carpeta base
    if not args.new_prefix:
        args.new_prefix = next_version_prefix(args.bucket, base_folder)
    print(f"🎯 Nueva versión: s3://{args.bucket}/{args.new_prefix}")

    os.makedirs(CACHE_DIR, exist_ok=True)

    # descargar meta.parquet
    meta_local = _s3_download(args.bucket, f"{args.base_prefix}/meta.parquet", os.path.join(CACHE_DIR, "meta.parquet"))
    meta = pd.read_parquet(meta_local)
    print(f"📚 {len(meta)} documentos cargados.")

    # modelo: intentar leer model.txt, si no, fallback
    model_name = MODEL_FALLBACK
    try:
        model_txt = download_bytes(args.bucket, f"{args.base_prefix}/model.txt").decode().strip()
        if model_txt:
            model_name = model_txt
    except Exception:
        pass
    print(f"🧠 Modelo de embeddings: {model_name}")

    # embeddings: usar embeddings.npz si existe; si no, re-embebemos
    try:
        emb_bytes = download_bytes(args.bucket, f"{args.base_prefix}/embeddings.npz")
        X = np.load(io.BytesIO(emb_bytes))["x"].astype(np.float32)
        print("✅ Embeddings cargados desde embeddings.npz")
    except Exception:
        print("ℹ️ No hay embeddings.npz → calculando embeddings desde Abstract...")
        model = SentenceTransformer(model_name)
        X = model.encode(meta["Abstract"].astype(str).tolist(), show_progress_bar=True).astype(np.float32)

    # KMeans
    k = int(args.n_clusters)
    print(f"🧩 KMeans K={k}")
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X)
    meta = meta.reset_index(drop=True)
    meta["cluster_id"] = labels

    # Gemini client
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Falta GOOGLE_API_KEY en el entorno/.env")
    from google import genai
    client = genai.Client(api_key=api_key)

    # Resumen por cluster
    categories = []
    for cid in sorted(meta["cluster_id"].unique()):
        subset = meta[meta["cluster_id"] == cid].head(12)  # limitar contexto
        desc = summarize_cluster_gemini(client, subset, cid)
        categories.append({
            "cluster_id": int(cid),
            "label": desc["label"],                  # ≤ 2 palabras
            "summary": desc["summary"],
            "keywords": desc["keywords"],
            "count": int((meta["cluster_id"] == cid).sum()),
            "example_titles": subset["Title"].astype(str).head(3).tolist()
        })
        print(f"✅ Cluster {cid}: {desc['label']} ({(meta['cluster_id']==cid).sum()} docs)")

    # Guardar artefactos locales
    out_dir = "artifacts/categories"
    os.makedirs(out_dir, exist_ok=True)
    meta_out = os.path.join(out_dir, "meta.parquet")
    cats_out = os.path.join(out_dir, "categories.json")
    mani_out = os.path.join(out_dir, "manifest.json")

    meta.to_parquet(meta_out, index=False)
    with open(cats_out, "w", encoding="utf-8") as f:
        json.dump(categories, f, indent=2, ensure_ascii=False)
    manifest = {
        "version": args.new_prefix.split("/")[-1],
        "model": model_name,
        "count": int(len(meta)),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base": args.base_prefix,
        "clusters": k,
    }
    with open(mani_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Publicar a S3
    print(f"📤 Subiendo a s3://{args.bucket}/{args.new_prefix}")
    _s3_upload(args.bucket, f"{args.new_prefix}/meta.parquet", meta_out)
    _s3_upload(args.bucket, f"{args.new_prefix}/categories.json", cats_out)
    _s3_upload(args.bucket, f"{args.new_prefix}/manifest.json", mani_out)

    # copiar index.faiss y embeddings.npz/model.txt de la base (si existen)
    try:
        idx_bytes = download_bytes(args.bucket, f"{args.base_prefix}/index.faiss")
        upload_bytes(args.bucket, f"{args.new_prefix}/index.faiss", idx_bytes)
    except Exception as e:
        print(f"⚠️ No se copió index.faiss: {e}")

    try:
        emb_bytes = download_bytes(args.bucket, f"{args.base_prefix}/embeddings.npz")
        upload_bytes(args.bucket, f"{args.new_prefix}/embeddings.npz", emb_bytes)
    except Exception:
        pass

    try:
        upload_bytes(args.bucket, f"{args.new_prefix}/model.txt", model_name.encode("utf-8"))
    except Exception:
        pass

    print("✅ Categorización completada y publicada.")


if __name__ == "__main__":
    main()
