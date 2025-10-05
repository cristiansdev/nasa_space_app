# raglib/embeddings.py
from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable, List, Optional

import numpy as np

# Evita que sentence-transformers trate de usar GPU si no quieres
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise RuntimeError(
        "Necesitas instalar sentence-transformers:\n"
        "  pip install sentence-transformers"
    ) from e


DEFAULT_MODEL = os.getenv("RAG_MODEL", "all-MiniLM-L6-v2")


@lru_cache(maxsize=4)
def get_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """
    Carga y cachea un modelo de sentence-transformers.
    Usa cache LRU para que no se recargue en cada llamada.
    """
    return SentenceTransformer(model_name)


def embed_texts(
    texts: Iterable[str],
    model_name: Optional[str] = None,
    show_progress_bar: bool = False,
    normalize: bool = False,
    as_numpy: bool = True,
) -> np.ndarray:
    """
    Genera embeddings para una colección de textos.

    Args:
        texts: Iterable de strings.
        model_name: Nombre del modelo (opcional).
        show_progress_bar: Muestra barra de progreso en el encode.
        normalize: Aplica normalización L2 (útil para cosine similarity).
        as_numpy: Devuelve np.ndarray float32.

    Returns:
        np.ndarray de shape (n_texts, dim).
    """
    if not isinstance(texts, (list, tuple)):
        texts = list(texts)

    model = get_model(model_name or DEFAULT_MODEL)
    emb = model.encode(
        list(texts),
        show_progress_bar=show_progress_bar,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    # asegurar dtype consistente
    emb = emb.astype(np.float32, copy=False)
    return emb
