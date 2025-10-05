from __future__ import annotations
import io
import json
import time
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import faiss

from .embeddings import load_model, encode_texts
from .faiss_utils import build_flat_index, write_index_bytes
from .s3_utils import upload_bytes

# ---------- Clustering ----------
def kmeans_labels(X: np.ndarray, k: int, random_state: int = 42) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X.astype(np.float32))
    return labels

# ---------- Resumen de un cluster con Gemini ----------
def summarize_cluster_gemini(subset_df: pd.DataFrame, api_key: str) -> Dict:
    """
    subset_df debe tener columnas Title, Abstract.
    Retorna: {"label": str, "summary": str, "keywords": List[str]}
    """
    from google import genai
    client = genai.Client(api_key=api_key)

    ctx = "\n\n".join([
        f"Título: {r.Title}\nAbstract: {r.Abstract}"
        for _, r in subset_df.iterrows()
    ])

    prompt = (
        "Eres un bibliotecario científico. Dadas estas publicaciones, responde en JSON con claves exactas "
        "`label`, `summary` y `keywords`:\n"
        "- `label`: nombre de la categoría (≤5 palabras)\n"
        "- `summary`: 2–3 frases en español, claras para no expertos\n"
        "- `keywords`: 3–6 palabras clave como lista\n\n"
        f"Contexto:\n{ctx}\n\n"
        "Responde SOLO el JSON."
    )

    raw = ""
    try:
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        raw = (resp.text or "").strip()
        try:
            data = json.loads(raw)
        except Exception:
            start = raw.find("{"); end = raw.rfind("}")
            if start >= 0 and end > start:
                data = json.loads(raw[start:end+1])
            else:
                raise ValueError("No se pudo parsear JSON del modelo.")
    except Exception:
        data = {"label": "Tema", "summary": "Resumen no disponible.", "keywords": []}

    # saneo básico
    label = (data.get("label") or "Tema").strip()
    summary = (data.get("summary") or "Resumen no disponible.").strip()
    keywords = data.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [x.strip() for x in keywords.split(",") if x.strip()]
    return {"label": label, "summary": summary, "keywords": keywords}

# ---------- Construcción de categories.json ----------
def build_categories_json(meta: pd.DataFrame, api_key: str, top_n_per_cluster: int = 10) -> List[Dict]:
    """
    Calcula un resumen por cluster. meta debe incluir 'cluster_id', 'Title', 'Abstract'.
    """
    categories: List[Dict] = []
    for cid in sorted(meta["cluster_id"].unique()):
        subset = meta[meta["cluster_id"] == cid].head(top_n_per_cluster)
        desc = summarize_cluster_gemini(subset, api_key=api_key)
        categories.append({
            "cluster_id": int(cid),
            "label": desc["label"],
            "summary": desc["summary"],
            "keywords": desc["keywords"],
            "count": int((meta["cluster_id"] == cid).sum()),
            "example_titles": subset["Title"].astype(str).head(3).tolist()
        })
    return categories

# ---------- Subíndices por categoría ----------
def build_and_upload_subindices(
    bucket: str,
    new_prefix: str,
    meta_with_labels: pd.DataFrame,
    X_vectors: np.ndarray,
) -> None:
    """
    Para cada cluster_id crea: by_cluster/<cid>/{index.faiss, meta.parquet}
    X_vectors y meta_with_labels deben estar alineados (mismo orden).
    """
    dim = X_vectors.shape[1]
    for cid in sorted(meta_with_labels["cluster_id"].unique()):
        mask = (meta_with_labels["cluster_id"] == cid).values
        Xc = X_vectors[mask].astype(np.float32)
        metac = meta_with_labels.loc[mask, ["Title", "Link", "Abstract", "cluster_id"]]

        # índice flat simple
        idx = build_flat_index(dim)
        idx.add(Xc)

        # subir index
        idx_bytes = write_index_bytes(idx)
        upload_bytes(bucket, f"{new_prefix}/by_cluster/{cid}/index.faiss", idx_bytes)

        # subir meta
        buf_meta = io.BytesIO()
        metac.to_parquet(buf_meta, index=False)
        upload_bytes(bucket, f"{new_prefix}/by_cluster/{cid}/meta.parquet", buf_meta.getvalue())
