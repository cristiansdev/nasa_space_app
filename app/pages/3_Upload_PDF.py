# app/pages/3_Upload_PDF.py
# ---------------------------------------------------------------
# Subir PDF → (Categorizar) → Embeddings → Indexar → Guardar en S3
# - Evita import circular de raglib.categorize y pdf_utils.
# - Usa S3() / resolve_base_prefix() de raglib.s3_utils.
# - Si raglib.embeddings falla, usa fallback Gemini (si GOOGLE_API_KEY existe).
# - Botón "Enviar y procesar".
# ---------------------------------------------------------------

from __future__ import annotations

import io
import os
import pathlib
import sys
import uuid
from importlib import import_module
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import numpy as np
import pandas as pd
from pypdf import PdfReader

# ----------------------------- PATH --------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ----------------------------- raglib -------------------------------
# Importamos SOLO lo que no dispara el circular
try:
    import raglib.s3_utils as s3u
except Exception as e:
    st.error(f"❌ No pude importar raglib.s3_utils: {e}")
    st.stop()

# faiss/index (si no existe, seguimos sin indexar)
try:
    import raglib.faiss_utils as fu
except Exception as e:
    fu = None
    _faiss_err = e

# embeddings: puede fallar; tratamos aparte
_emb_mod = None
_emb_import_err = None
try:
    import raglib.embeddings as _emb_mod
except Exception as e:
    _emb_import_err = e

# categorize: lo importamos dinámicamente dentro de una función
def _try_import_categorize():
    try:
        return import_module("raglib.categorize")
    except Exception:
        return None

# ----------------------- Fallback Embeddings ------------------------
def _fallback_get_client():
    try:
        import google.generativeai as genai  # requiere estar instalado y GOOGLE_API_KEY set
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None
        return genai.Client(api_key=api_key)
    except Exception:
        return None

def _fallback_chunk_text(text: str, size: int = 2000, overlap: int = 200) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks, n, start = [], len(text), 0
    while start < n:
        end = min(start + size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def _fallback_embed_texts(client, chunks: List[str]) -> np.ndarray:
    if client is None or not chunks:
        return np.empty((0, 0))
    try:
        resp = client.models.embed_content(
            model="text-embedding-004",
            contents=[{"role": "user", "parts": [t]} for t in chunks],
        )
        vecs = []
        if hasattr(resp, "embeddings"):
            for e in resp.embeddings:
                vecs.append(np.array(e.values, dtype=np.float32))
        elif hasattr(resp, "data"):
            for item in resp.data:
                vec = item.get("embedding", {}).get("values", [])
                vecs.append(np.array(vec, dtype=np.float32))
        if not vecs:
            return np.empty((0, 0))
        return np.vstack(vecs)
    except Exception:
        return np.empty((0, 0))

# --------------------------- S3 helpers -----------------------------
def build_s3() -> Tuple[Optional[object], Optional[str], Optional[str]]:
    """
    Instancia S3() si es posible y lee bucket/prefix desde el objeto.
    Si no hay prefix, usa resolve_base_prefix() si existe.
    """
    s3_obj = None
    bucket = None
    prefix = None

    if hasattr(s3u, "S3"):
        try:
            s3_obj = s3u.S3()
        except Exception:
            s3_obj = None

    if s3_obj is not None:
        bucket = getattr(s3_obj, "bucket", None)
        prefix = getattr(s3_obj, "base_prefix", None) or getattr(s3_obj, "prefix", None)

    if (prefix is None or prefix == "") and hasattr(s3u, "resolve_base_prefix"):
        try:
            prefix = s3u.resolve_base_prefix()
        except Exception:
            prefix = None

    return s3_obj, bucket, prefix

def s3_uri(bucket: str, key: str) -> str:
    if hasattr(s3u, "s3_uri"):
        try:
            return s3u.s3_uri(bucket, key)
        except Exception:
            pass
    return f"s3://{bucket}/{key}"

def upload_bytes(s3_obj: Optional[object], bucket: str, key: str, data: bytes, content_type: str) -> bool:
    # Primero método del objeto S3
    if s3_obj is not None:
        for m in ("upload_bytes", "upload_file", "put_bytes"):
            fn = getattr(s3_obj, m, None)
            if callable(fn):
                try:
                    fn(key, data, content_type=content_type)
                    return True
                except TypeError:
                    try:
                        fn(key, data)
                        return True
                    except Exception:
                        pass
                except Exception:
                    pass
    # Luego funciones del módulo
    for m in ("upload_bytes", "upload_file", "_s3_upload"):
        fn = getattr(s3u, m, None)
        if callable(fn):
            try:
                fn(bucket, key, data, content_type=content_type)
                return True
            except TypeError:
                try:
                    fn(bucket, key, data)
                    return True
                except Exception:
                    pass
            except Exception:
                pass
    return False

# --------------------------- PDF helper -----------------------------
def extract_text_from_pdf(data: bytes) -> Dict[str, Any]:
    # Evitamos raglib.pdf_utils para no disparar el circular; usamos PyPDF local.
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return {"n_pages": len(pages), "pages": pages, "text": "\n\n".join(pages)}

# ------------------------- Categorize helper -----------------------
def get_categories_and_classifier():
    """
    Importa raglib.categorize de forma perezosa.
    Devuelve (lista_categorias, funcion_clasificar | None).
    """
    mod = _try_import_categorize()
    if mod is None:
        return [], None

    cats = list(getattr(mod, "KNOWN_CATEGORIES", getattr(mod, "CATEGORIES", [])))
    clf = None
    for name in ("classify_text", "categorize_text", "classify", "predict_category"):
        fn = getattr(mod, name, None)
        if callable(fn):
            clf = fn
            break
    return cats, clf

# ---------------------------- Embeddings ---------------------------
def get_client_and_ops():
    """
    Devuelve (client, chunk_text_fn, embed_texts_fn, source_str).
    Usa raglib.embeddings si cargó; si no, fallback Gemini.
    """
    if _emb_mod is not None:
        # Cliente
        client = None
        for name in ("get_client", "build_client", "client"):
            fn = getattr(_emb_mod, name, None)
            if callable(fn):
                try:
                    client = fn()
                    break
                except Exception:
                    pass
            if name == "client" and fn is not None:
                client = fn
                break
        # Funciones
        chunk_fn = getattr(_emb_mod, "chunk_text", None)
        embed_fn = getattr(_emb_mod, "embed_texts", None) or getattr(_emb_mod, "embed", None)
        if client is not None and callable(chunk_fn) and callable(embed_fn):
            return client, chunk_fn, embed_fn, "raglib.embeddings"

    # Fallback
    client = _fallback_get_client()
    if client is None:
        return None, None, None, f"fallback: no se pudo importar raglib.embeddings ({_emb_import_err}) y no hay GOOGLE_API_KEY"
    return client, _fallback_chunk_text, _fallback_embed_texts, "fallback: google-generativeai"

# ------------------------------ UI --------------------------------
st.set_page_config(page_title="Upload PDF", page_icon="📄", layout="wide")
st.title("📄 Subir PDF → Categorizar, Embeddings, Indexar y Guardar")
st.caption("Evita el import circular. Usa S3() / resolve_base_prefix(). Fallback a Gemini si embeddings falla.")

uploaded = st.file_uploader("Sube uno o varios PDFs", type=["pdf"], accept_multiple_files=True)
enviar = st.button("🚀 Enviar y procesar", type="primary", use_container_width=True)

# --------------------------- Acción -------------------------------
if uploaded and enviar:
    s3_obj, bucket, prefix = build_s3()
    if not bucket:
        st.error(
            "No pude determinar el bucket (S3().bucket / resolve_base_prefix()). "
            f"Contenido de s3_utils: {', '.join(dir(s3u))}"
        )
        st.stop()

    client, chunk_text_fn, embed_texts_fn, source = get_client_and_ops()
    if client is None or chunk_text_fn is None or embed_texts_fn is None:
        st.error("No pude inicializar cliente/funciones de embeddings.")
        if _emb_import_err:
            st.info(f"Detalle al importar raglib.embeddings: {_emb_import_err}")
        st.stop()

    cats, clf = get_categories_and_classifier()
    if cats:
        st.write("**Categorías detectadas:**", ", ".join(cats))
    else:
        st.info("No pude importar raglib.categorize; la categoría se marcará como 'unknown'.")

    st.success(f"Usaré bucket **{bucket}**, prefijo **{prefix or '(sin prefix)'}**, embeddings desde **{source}**")

    for i, uf in enumerate(uploaded, start=1):
        st.divider()
        st.subheader(f"PDF #{i}: {uf.name}")

        data = uf.read()
        if not data:
            st.warning("Archivo vacío o no legible.")
            continue

        # Key base: usa prefix si existe
        doc_id = str(uuid.uuid4())
        base_key = f"{prefix.rstrip('/')}/{doc_id}" if prefix else doc_id
        pdf_key = f"{base_key}/{uf.name}"

        # 1) Subir PDF
        if not upload_bytes(s3_obj, bucket, pdf_key, data, content_type="application/pdf"):
            st.error("Error al subir PDF a S3 (revisa s3_utils).")
            continue
        pdf_link = s3_uri(bucket, pdf_key)
        st.write(f"📤 PDF en S3: `{pdf_link}`")

        # 2) Extraer texto
        extracted = extract_text_from_pdf(data)
        text = extracted.get("text", "").strip()
        if not text:
            st.warning("No se extrajo texto del PDF.")
            continue

        # 3) Categoría (opcional si clf existe)
        if callable(clf):
            try:
                category = str(clf(text, allowed=cats)) if cats else str(clf(text))
            except TypeError:
                category = str(clf(text))
        else:
            category = "unknown"
        st.write(f"🏷️ Categoría: **{category}**")

        # 4) Chunks + Embeddings
        chunks = list(chunk_text_fn(text))
        st.write(f"🧩 Chunks generados: {len(chunks)}")

        embeds = embed_texts_fn(client, chunks)
        arr = np.array(embeds)
        if arr.ndim == 1:
            arr = np.expand_dims(arr, 0)
        if arr.size == 0 or arr.shape[0] == 0:
            st.error("No se generaron embeddings.")
            continue
        st.write(f"🧬 Embeddings shape: {arr.shape}")

        # 5) Guardar artefactos en S3
        chunk_csv_key = f"{base_key}/chunks.csv"
        emb_npy_key  = f"{base_key}/embeddings.npy"

        df_chunks = pd.DataFrame({"doc_id": doc_id, "chunk_id": range(len(chunks)), "text": chunks})
        csv_bytes = df_chunks.to_csv(index=False).encode("utf-8")
        npy_buf = io.BytesIO(); np.save(npy_buf, arr)

        upload_bytes(s3_obj, bucket, chunk_csv_key, csv_bytes, content_type="text/csv")
        upload_bytes(s3_obj, bucket, emb_npy_key,  npy_buf.getvalue(), content_type="application/octet-stream")

        st.write("📦 Artefactos guardados en S3:")
        st.json({
            "chunks_csv": s3_uri(bucket, chunk_csv_key),
            "embeddings_npy": s3_uri(bucket, emb_npy_key),
        })

        # 6) Indexar (si faiss_utils disponible)
        if fu is not None:
            meta = [
                {"doc_id": doc_id, "chunk_id": j, "title": uf.name, "category": category, "pdf_uri": pdf_link}
                for j in range(len(chunks))
            ]
            ok = False
            for name in ("upsert_chunks", "upsert", "index_chunks", "add_vectors"):
                fn = getattr(fu, name, None)
                if callable(fn):
                    try:
                        fn(doc_id=doc_id, chunks=chunks, embeddings=arr, metadata=meta)
                        ok = True
                        break
                    except TypeError:
                        try:
                            fn(chunks, arr, meta)
                            ok = True
                            break
                        except Exception:
                            pass
                    except Exception:
                        pass
            if ok:
                st.success("🗂️ Índice vectorial actualizado.")
            else:
                st.warning("No pude indexar (revisa raglib.faiss_utils).")
        else:
            st.info(f"No se importó faiss_utils: {_faiss_err}")

        # 7) Vista previa
        with st.expander("Vista previa del texto extraído (primeros 1200 chars)"):
            st.text_area("Texto", value=(text[:1200] + ("…" if len(text) > 1200 else "")), height=220)
