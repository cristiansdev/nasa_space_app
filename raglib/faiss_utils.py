from __future__ import annotations
import os
import io
from typing import Tuple
import numpy as np
import faiss

# ---------- Construcción ----------
def build_flat_index(dim: int) -> faiss.IndexFlatL2:
    """Índice L2 plano (exacto)."""
    return faiss.IndexFlatL2(dim)

def add_vectors(index: faiss.Index, X: np.ndarray) -> None:
    """Añade vector(es) (N, d) o (1, d) al índice."""
    if X.ndim == 1:
        X = X.reshape(1, -1)
    index.add(X.astype(np.float32))

def search_vectors(index: faiss.Index, Q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Busca los k más cercanos. Q puede ser (1, d) o (N, d)."""
    if Q.ndim == 1:
        Q = Q.reshape(1, -1)
    D, I = index.search(Q.astype(np.float32), k)
    return D, I

# ---------- IO: path ----------
def read_index_path(path: str) -> faiss.Index:
    return faiss.read_index(path)

def write_index_path(index: faiss.Index, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    faiss.write_index(index, path)

# ---------- IO: bytes (FAISS no soporta BytesIO directamente en Python) ----------
def read_index_bytes(data: bytes) -> faiss.Index:
    tmp = "/tmp/_faiss_read.idx"
    with open(tmp, "wb") as f:
        f.write(data)
    return faiss.read_index(tmp)

def write_index_bytes(index: faiss.Index) -> bytes:
    tmp = "/tmp/_faiss_write.idx"
    faiss.write_index(index, tmp)
    with open(tmp, "rb") as f:
        return f.read()
