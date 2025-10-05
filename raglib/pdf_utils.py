import io
import os
import re
from typing import Tuple

from pypdf import PdfReader

# Regex para localizar el abstract
_ABSTRACT_START = re.compile(r"\b(Abstract|Resumen)\b\s*[:.]?\s*", re.I)
_ABSTRACT_END = re.compile(
    r"\b(Keywords?|Index\s+Terms?|Palabras\s+clave|Introducción|Introduction|1\.|I\.)\b",
    re.I
)

# ---------- Extracción de texto ----------

def extract_text_first_pages(pdf_path: str, max_pages: int = 3) -> str:
    """Extrae texto de las primeras páginas de un PDF en disco."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)
    if os.path.getsize(pdf_path) == 0:
        raise ValueError(f"PDF vacío: {pdf_path}")

    # pypdf
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
        if len(joined) > 20:
            return joined
    except Exception:
        pass

    # fallback: PyMuPDF
    try:
        import fitz
        doc = fitz.open(pdf_path)
        pages = min(len(doc), max_pages)
        parts = [doc.load_page(i).get_text("text") or "" for i in range(pages)]
        joined = "\n".join(parts).strip()
        if len(joined) > 20:
            return joined
    except Exception as e:
        raise RuntimeError(f"No pude extraer texto útil (pypdf/pymupdf): {e}")

    raise RuntimeError("Contenido insuficiente para extraer texto.")

def extract_text_first_pages_from_bytes(pdf_bytes: bytes, max_pages: int = 3) -> str:
    """Versión para archivos en memoria (por ejemplo, subida en Streamlit)."""
    # pypdf
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = min(len(reader.pages), max_pages)
        text = []
        for i in range(pages):
            try:
                text.append(reader.pages[i].extract_text() or "")
            except Exception:
                continue
        joined = "\n".join(text).strip()
        if len(joined) > 20:
            return joined
    except Exception:
        pass

    # fallback: PyMuPDF
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = min(len(doc), max_pages)
        parts = [doc.load_page(i).get_text("text") or "" for i in range(pages)]
        joined = "\n".join(parts).strip()
        if len(joined) > 20:
            return joined
    except Exception as e:
        raise RuntimeError(f"No pude extraer texto útil (pypdf/pymupdf): {e}")

    raise RuntimeError("Contenido insuficiente para extraer texto.")


# ---------- Heurísticas de título y abstract ----------

def guess_title(pdf_path: str, first_page_text: str) -> str:
    # metadata
    try:
        reader = PdfReader(pdf_path)
        t = (reader.metadata.title or "").strip()
        if t:
            return t
    except Exception:
        pass
    # primeras líneas “limpias”
    for l in [l.strip() for l in first_page_text.splitlines() if l.strip()][:10]:
        if len(l) > 6 and not re.search(r"(abstract|resumen|keywords|introduction|doi)", l, re.I):
            return l[:220]
    return "Untitled paper"

def _extract_abstract_from_text(norm_text: str) -> str:
    m = _ABSTRACT_START.search(norm_text)
    if m:
        start = m.end()
        tail = norm_text[start:]
        endm = _ABSTRACT_END.search(tail)
        abstract = tail[:endm.start()].strip() if endm else tail.strip()
    else:
        abstract = norm_text.strip()
        abstract = abstract[:1200].rsplit("\n", 1)[0].strip() or abstract[:1200].strip()
    abstract = re.sub(r"\s+\n", "\n", abstract)
    abstract = re.sub(r"\n{3,}", "\n\n", abstract)
    return abstract

def extract_title_and_abstract(pdf_path: str) -> Tuple[str, str]:
    text = extract_text_first_pages(pdf_path, 3)
    title = guess_title(pdf_path, text)
    norm = re.sub(r"[ \t]+", " ", text)
    norm = re.sub(r"\n{2,}", "\n\n", norm)
    abstract = _extract_abstract_from_text(norm)
    return title, abstract

def extract_title_and_abstract_from_bytes(pdf_bytes: bytes, fallback_title: str) -> Tuple[str, str]:
    # titulo de metadata (si existe)
    title_guess = fallback_title
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        t = (reader.metadata.title or "").strip()
        if t:
            title_guess = t
    except Exception:
        pass

    text = extract_text_first_pages_from_bytes(pdf_bytes, 3)
    norm = re.sub(r"[ \t]+", " ", text)
    norm = re.sub(r"\n{2,}", "\n\n", norm)
    abstract = _extract_abstract_from_text(norm)
    title_guess = title_guess if title_guess else (text.split("\n", 1)[0][:220] if text else "Untitled paper")
    return title_guess, abstract
