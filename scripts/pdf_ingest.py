#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ingesta incremental de un PDF:
  - Extrae título y abstract (pypdf + fallback PyMuPDF)
  - Carga base desde S3 (index/vN o index/latest)
  - Embeber y agregar al índice + metadata
  - Publicar nueva versión automática (vN+1) si no se pasa --new-prefix
"""

import os, re, json, time, argparse, numpy as np, pandas as pd
from dotenv import load_dotenv
import boto3, faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

load_dotenv()
DEFAULT_BUCKET = os.getenv("RAG_BUCKET", "nasa-rag-rag")
DEFAULT_BASE_PREFIX = os.getenv("RAG_PREFIX", "index/latest")
MODEL_NAME = "all-MiniLM-L6-v2"
S3 = boto3.client("s3")
_V_RE = re.compile(r"^v(\d+)$")

def _s3_download(bucket: str, key: str, local_path: str) -> str:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    S3.download_file(bucket, key, local_path)
    return local_path

def _s3_upload(bucket: str, key: str, local_path: str):
    S3.upload_file(local_path, bucket, key)
    print(f"↑ s3://{bucket}/{key}")

def _ensure_bucket(bucket: str):
    region = boto3.session.Session().region_name
    try:
        S3.head_bucket(Bucket=bucket)
        print(f"🪣 Bucket '{bucket}' OK.")
    except Exception:
        print(f"⚠️ Bucket '{bucket}' no existe. Creando en {region}...")
        if region == "us-east-1":
            S3.create_bucket(Bucket=bucket)
        else:
            S3.create_bucket(Bucket=bucket,
                             CreateBucketConfiguration={'LocationConstraint': region})
        print("✅ Bucket creado.")

def _base_folder_from_prefix(prefix: str) -> str:
    parts = prefix.strip("/").split("/")
    return (parts[0] if parts else "index").rstrip("/") + "/"

def list_versions(bucket: str, base_folder: str = "index/"):
    paginator = S3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=base_folder, Delimiter="/")
    vers = []
    for page in pages:
        for cp in page.get("CommonPrefixes", []):
            sub = cp["Prefix"].strip("/").split("/")[-1]
            if _V_RE.match(sub):
                vers.append(sub)
    vers.sort(key=lambda v: int(_V_RE.match(v).group(1)))
    return vers

def latest_version_prefix(bucket: str, base_folder: str = "index/"):
    vers = list_versions(bucket, base_folder)
    if not vers:
        raise RuntimeError(f"No hay versiones en s3://{bucket}/{base_folder}")
    return f"{base_folder}{vers[-1]}"

def next_version_prefix(bucket: str, base_folder: str = "index/"):
    vers = list_versions(bucket, base_folder)
    if not vers:
        return f"{base_folder}v1"
    n = int(_V_RE.match(vers[-1]).group(1)) + 1
    return f"{base_folder}v{n}"

def extract_text_first_pages(pdf_path: str, max_pages: int = 3) -> str:
    if not os.path.exists(pdf_path): raise FileNotFoundError(pdf_path)
    if os.path.getsize(pdf_path) == 0: raise ValueError("PDF vacío")
    try:
        reader = PdfReader(pdf_path)
        pages = min(len(reader.pages), max_pages)
        text = []
        for i in range(pages):
            try:
                text.append(reader.pages[i].extract_text() or "")
            except Exception:
                continue
        joined = "\n".join(text).strip()
        if len(joined) > 20: return joined
    except Exception:
        pass
    try:
        import fitz
        doc = fitz.open(pdf_path)
        pages = min(len(doc), max_pages)
        parts = [doc.load_page(i).get_text("text") or "" for i in range(pages)]
        joined = "\n".join(parts).strip()
        if len(joined) > 20: return joined
    except Exception as e:
        raise RuntimeError(f"No pude extraer texto: {e}")
    raise RuntimeError("Texto insuficiente.")

def guess_title(pdf_path: str, first_page_text: str) -> str:
    try:
        reader = PdfReader(pdf_path)
        t = (reader.metadata.title or "").strip()
        if t: return t
    except Exception:
        pass
    lines = [l.strip() for l in first_page_text.splitlines() if l.strip()]
    for l in lines[:10]:
        if len(l) > 6 and not re.search(r"(abstract|resumen|keywords|introduction|doi)", l, re.I):
            return l[:220]
    return "Untitled paper"

_ABSTRACT_START = re.compile(r"\b(Abstract|Resumen)\b\s*[:.]?\s*", re.I)
_ABSTRACT_END   = re.compile(r"\b(Keywords?|Palabras\s+clave|Introducción|Introduction|1\.|I\.)\b", re.I)

def extract_abstract_from_pdf(pdf_path: str):
    text = extract_text_first_pages(pdf_path, max_pages=3)
    title = guess_title(pdf_path, text)
    norm = re.sub(r"[ \t]+", " ", text)
    norm = re.sub(r"\n{2,}", "\n\n", norm)
    m = _ABSTRACT_START.search(norm)
    if m:
        start = m.end()
        tail  = norm[start:]
        endm  = _ABSTRACT_END.search(tail)
        abstract = tail[:endm.start()].strip() if endm else tail.strip()
    else:
        abstract = norm.strip()
        abstract = abstract[:1200].rsplit("\n", 1)[0].strip() or abstract[:1200].strip()
    abstract = re.sub(r"\s+\n", "\n", abstract)
    abstract = re.sub(r"\n{3,}", "\n\n", abstract)
    return title, abstract

def load_base_index_and_meta(bucket: str, base_prefix: str, cache_dir: str = "/tmp/rag"):
    idx_local  = _s3_download(bucket, f"{base_prefix}/index.faiss", os.path.join(cache_dir, "base_index.faiss"))
    meta_local = _s3_download(bucket, f"{base_prefix}/meta.parquet", os.path.join(cache_dir, "base_meta.parquet"))
    index = faiss.read_index(idx_local)
    df    = pd.read_parquet(meta_local)
    return index, df

def append_one_paper_and_upload(pdf_path: str, bucket: str, base_prefix: str, new_prefix: str | None):
    _ensure_bucket(bucket)
    base_folder = _base_folder_from_prefix(base_prefix)
    if not new_prefix:
        new_prefix = next_version_prefix(bucket, base_folder)
        print(f"🔁 new-prefix no provisto. Usaré: {new_prefix}")

    if base_prefix.endswith("/latest") or base_prefix == "index/latest":
        base_prefix = latest_version_prefix(bucket, base_folder)
        print(f"ℹ️ 'latest' → {base_prefix}")

    print(f"📄 Extrayendo Abstract de: {pdf_path}")
    title, abstract = extract_abstract_from_pdf(pdf_path)
    if not abstract or len(abstract) < 50:
        raise RuntimeError("Abstract insuficiente.")

    print(f"☁️ Cargando base desde s3://{bucket}/{base_prefix}")
    index, df_meta = load_base_index_and_meta(bucket, base_prefix)

    print(f"🧠 Generando embedding ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)
    vec = model.encode([abstract], show_progress_bar=False).astype(np.float32)
    index.add(vec)

    df_new = pd.DataFrame([{"Title": title, "Link": "", "Abstract": abstract}])
    df_all = pd.concat([df_meta[["Title","Link","Abstract"]], df_new], ignore_index=True)

    print(f"💾 Guardando artefactos locales...")
    os.makedirs("artifacts_pdf", exist_ok=True)
    faiss.write_index(index, "artifacts_pdf/index.faiss")
    df_all.to_parquet("artifacts_pdf/meta.parquet", index=False)
    np.savez_compressed("artifacts_pdf/embeddings_added.npz", x=vec)
    manifest = {
        "version": new_prefix.split("/")[-1],
        "model": MODEL_NAME,
        "count": int(len(df_all)),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": os.path.basename(pdf_path),
        "base": base_prefix
    }
    with open("artifacts_pdf/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"📤 Publicando en s3://{bucket}/{new_prefix}")
    _s3_upload(bucket, f"{new_prefix}/index.faiss", "artifacts_pdf/index.faiss")
    _s3_upload(bucket, f"{new_prefix}/meta.parquet", "artifacts_pdf/meta.parquet")
    _s3_upload(bucket, f"{new_prefix}/embeddings_added.npz", "artifacts_pdf/embeddings_added.npz")
    _s3_upload(bucket, f"{new_prefix}/manifest.json", "artifacts_pdf/manifest.json")
    print("✅ Listo.")

def parse_args():
    ap = argparse.ArgumentParser(description="Ingesta PDF → Abstract → Embedding → S3 (versionado automático por defecto)")
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--base-prefix", default=DEFAULT_BASE_PREFIX)
    ap.add_argument("--new-prefix", default=None)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    append_one_paper_and_upload(args.pdf, args.bucket, args.base_prefix, args.new_prefix)
