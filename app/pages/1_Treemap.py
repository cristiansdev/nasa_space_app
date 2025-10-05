import os
import io
import sys
import json
import pathlib
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# Cargar variables de entorno desde .env
# ---------------------------------------------------------------------
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# ---------------------------------------------------------------------
# AÑADIR RAÍZ DEL PROYECTO AL PYTHONPATH
# ---------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raglib.s3_utils import download_bytes
from raglib.embeddings import embed_texts  # helper interno

# ---------------------------------------------------------------------
# Si está disponible Gemini, activarlo
# ---------------------------------------------------------------------
USE_GEMINI = False
if GEMINI_KEY:
    try:
        import google.genai as genai
        client = genai.Client(api_key=GEMINI_KEY)
        USE_GEMINI = True
    except Exception as e:
        st.warning(f"⚠️ Gemini no se pudo inicializar: {e}")
else:
    st.warning("⚠️ No se detectó GEMINI_API_KEY ni GOOGLE_API_KEY en tu .env")

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
BUCKET = os.getenv("RAG_BUCKET", "nasa-rag-rag")
PREFIX = os.getenv("RAG_PREFIX", "index/v5_categories")
CATEGORIES_KEY = f"{PREFIX}/categories.json"
META_KEY = f"{PREFIX}/meta.parquet"

st.set_page_config(page_title="Treemap", page_icon="🧠", layout="wide")

if "selected_category_id" not in st.session_state:
    st.session_state.selected_category_id = None
if "selected_category_label" not in st.session_state:
    st.session_state.selected_category_label = None

# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------
def _normalize_categories(raw: list[dict]) -> list[dict]:
    norm = []
    for i, c in enumerate(raw):
        label = c.get("label", f"Cluster {i}")
        count = int(c.get("count", 0))
        summary = c.get("summary", "")
        cid = c.get("cluster_id", c.get("id", i))
        norm.append({
            "label": label,
            "count": count,
            "summary": summary,
            "id": i,
            "cluster_id": int(cid),
        })
    return norm


@st.cache_data(show_spinner=False)
def load_categories_and_meta() -> tuple[list[dict], pd.DataFrame]:
    cat_bytes = download_bytes(BUCKET, CATEGORIES_KEY)
    meta_bytes = download_bytes(BUCKET, META_KEY)
    cats_raw = json.loads(cat_bytes.decode("utf-8"))
    cats = _normalize_categories(cats_raw)
    df_meta = pd.read_parquet(io.BytesIO(meta_bytes))
    if "cluster_id" not in df_meta.columns:
        df_meta["cluster_id"] = 0
    return cats, df_meta


def _short(s: str, n: int = 200) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    cut = s[:n].rsplit(" ", 1)[0]
    return cut + "…"


def _wrap_for_hover(s: str, width: int = 58, max_lines: int = 6) -> str:
    if not s:
        return ""
    words = s.split()
    lines, cur, cur_len = [], [], 0
    for w in words:
        if cur_len + len(w) + (1 if cur else 0) > width:
            lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
            if len(lines) >= max_lines:
                lines[-1] += "…"
                break
        else:
            cur.append(w)
            cur_len += len(w) + (1 if cur_len else 0)
    if cur and len(lines) < max_lines:
        lines.append(" ".join(cur))
    return "<br>".join(lines)


def build_treemap_fig(categories: list[dict]) -> tuple[Any, pd.DataFrame]:
    labels = [c["label"] for c in categories]
    values = [c["count"] for c in categories]
    summaries = [c.get("summary", "") for c in categories]
    hover_summaries = [_wrap_for_hover(_short(s, 200), width=60, max_lines=6) for s in summaries]
    ids = [c["id"] for c in categories]

    df = pd.DataFrame({
        "labels": labels,
        "values": values,
        "summary": summaries,
        "hover_summary": hover_summaries,
        "id": ids,
    })

    pastel = ["#F8E8A6", "#BFD7EA", "#F4A6A6", "#B8E2C8",
              "#FFD09B", "#C7C2E8", "#EAC1E9", "#CFE2A3"]
    colors = (pastel * ((len(df) // len(pastel)) + 1))[: len(df)]

    fig = px.treemap(df, path=["labels"], values="values", color_discrete_sequence=colors)
    fig.update_traces(
        texttemplate="%{label}<br>%{value} papers",
        textfont=dict(size=16, color="black"),
        customdata=np.array(df[["hover_summary"]].values.tolist(), dtype=object),
        hovertemplate="<b>%{label}</b><br>%{value} papers<br><br>%{customdata[0]}<extra></extra>",
        marker=dict(line=dict(color="#2b2b2b", width=1.5)),
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
        uniformtext_minsize=10,
        uniformtext_mode="hide",
        hoverlabel=dict(
            bgcolor="rgba(36,36,36,0.92)",
            bordercolor="#444",
            font_size=14,
            font_color="#FAFAFA",
            align="left",
        ),
    )
    return fig, df


def get_category_by_label(categories: list[dict], label: str) -> Optional[dict]:
    for c in categories:
        if c["label"] == label:
            return c
    return None


def rag_answer_for_category(query: str, df_meta: pd.DataFrame, cluster_id: int, top_k: int = 5):
    """Usa RAG con Gemini si está habilitado, o heurístico si no."""
    df_c = df_meta[df_meta["cluster_id"] == cluster_id].copy()
    if df_c.empty:
        return "No hay documentos en esta categoría.", []

    abstracts = df_c["Abstract"].fillna("").astype(str).tolist()
    titles = df_c["Title"].fillna("").astype(str).tolist()
    links = df_c["Link"].fillna("").astype(str).tolist()

    corpus_emb = embed_texts(abstracts).astype(np.float32)
    q_emb = embed_texts([query]).astype(np.float32)[0]
    denom = np.linalg.norm(corpus_emb, axis=1) * (np.linalg.norm(q_emb) + 1e-8)
    sims = (corpus_emb @ q_emb) / (denom + 1e-8)
    idxs = np.argsort(-sims)[: top_k]

    retrieved = [{"Title": titles[i], "Link": links[i], "Abstract": abstracts[i], "score": float(sims[i])}
                 for i in idxs]

    # Contexto concatenado
    context = "\n\n".join([f"{r['Title']}:\n{r['Abstract']}" for r in retrieved])

    if USE_GEMINI:
        prompt = (
            f"Responde con base en los siguientes abstracts científicos.\n\n"
            f"Consulta: {query}\n\nContexto:\n{context}"
        )
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        answer = response.text or "No se obtuvo respuesta de Gemini."
    else:
        snippet = " • ".join([r["Title"] for r in retrieved[:3]])
        answer = (
            f"Con base en los documentos más cercanos en esta categoría, "
            f"los temas relevantes incluyen: {snippet}.\n\n"
            f"(Gemini no está configurado; usando resumen heurístico.)"
        )

    return answer, retrieved


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.title("🧭 Dashboard de Categorías de Papers")
st.caption(f"Usando: `s3://{BUCKET}/{PREFIX}`")

try:
    categories, df_meta = load_categories_and_meta()
except Exception as e:
    st.error(f"❌ No se pudo cargar categorías/meta desde S3: {e}")
    st.stop()

left, right = st.columns([7, 5], gap="large")

with left:
    labels = [c["label"] for c in categories]
    default_index = 0
    if st.session_state.selected_category_label in labels:
        default_index = labels.index(st.session_state.selected_category_label)
    selected_label = st.selectbox("Selecciona una categoría:", labels, index=default_index)

    selected_cat = get_category_by_label(categories, selected_label)
    if selected_cat:
        st.session_state.selected_category_label = selected_cat["label"]
        st.session_state.selected_category_id = selected_cat["cluster_id"]

        with st.container(border=True):
            st.markdown(
                f"<div style='text-align:center;line-height:1.4;'>"
                f"<div style='font-weight:700;font-size:1.1rem;'>{selected_cat['label']}</div>"
                f"<div style='opacity:0.85;'>Papers: {selected_cat['count']}</div></div>",
                unsafe_allow_html=True,
            )

    st.subheader("Categorías")
    fig, _ = build_treemap_fig(categories)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    if selected_cat and selected_cat.get("summary"):
        with st.container(border=True):
            st.markdown(
                f"<div style='text-align:center;line-height:1.55;'>"
                f"<div style='margin-bottom:6px;font-weight:600;'>Resumen</div>"
                f"<div style='max-width:92%;margin:0 auto;'>{selected_cat['summary']}</div></div>",
                unsafe_allow_html=True,
            )

with right:
    st.subheader("💬 Chat (RAG de esta categoría)")
    if st.session_state.selected_category_id is None:
        st.info("Primero selecciona una categoría en el panel izquierdo.")
    else:
        cid = st.session_state.selected_category_id
        label = st.session_state.selected_category_label or "Categoría"

        with st.container(border=True):
            st.markdown(f"**Categoría activa:** {label}")
            user_q = st.text_input("Tu pregunta",
                                   placeholder="Escribe tu pregunta sobre esta categoría…",
                                   label_visibility="collapsed",
                                   key="cat_query")
            ask = st.button("Preguntar", type="primary", use_container_width=True)

            if ask and user_q.strip():
                with st.spinner("Generando respuesta con Gemini..."):
                    answer, used_docs = rag_answer_for_category(user_q, df_meta, cid, top_k=5)
                st.markdown("**Respuesta:**")
                st.write(answer)
                with st.expander("Ver documentos usados en la respuesta"):
                    for r in used_docs:
                        st.markdown(f"- [{r['Title']}]({r['Link']})  \n  _(score: {r['score']:.3f})_")

        st.divider()
        st.subheader("📚 Papers de la categoría")
        df_cat = df_meta[df_meta["cluster_id"] == cid].copy()
        with st.expander(f"Ver listado de {len(df_cat)} papers", expanded=False):
            if df_cat.empty:
                st.write("No hay documentos en esta categoría.")
            else:
                for _, row in df_cat.iterrows():
                    st.markdown(f"- [{row['Title']}]({row['Link']})")
