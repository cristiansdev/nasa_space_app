#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extrae **únicamente el Abstract** de artículos PMC (PubMed Central)
a partir de CSV/Parquet con columnas Title y Link.

Uso:
  python scripts/extract_pmc_abstracts.py --input SB_publication_PMC.csv --output abstracts_full.csv
  python scripts/extract_pmc_abstracts.py --input SB_publication_PMC.csv --output abstracts_50.csv --limit 50
  python scripts/extract_pmc_abstracts.py --input SB_publication_PMC.csv --output abstracts_full.csv --checkpoint ckpt.csv

Requisitos:
  pip install pandas requests beautifulsoup4 html5lib tenacity tqdm pyarrow
"""

from __future__ import annotations
import argparse, pathlib, re, time
from typing import Dict, List, Optional, Tuple
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

HEADERS: Dict[str, str] = {
    "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
}

ABSTRACT_SELECTORS: List[str] = [
    'section.abstract[id^="abstract"]',
    'section#abstract',
    'section[id^="abstract"]',
    'div.abstract',
    'div.abstr',
]

def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def read_table(path: str) -> pd.DataFrame:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {path}")
    if p.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    if "Link" not in df.columns:
        raise ValueError("La entrada debe contener la columna 'Link'.")
    if "Title" not in df.columns:
        df["Title"] = ""
    return df[["Title", "Link"]].copy()

@retry(stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=2, max=15),
       retry=retry_if_exception_type((requests.RequestException,)))
def fetch_html(url: str, timeout: int = 25) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.text

def extract_abstract_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html5lib")
    for sel in ABSTRACT_SELECTORS:
        sec = soup.select_one(sel)
        if sec:
            parts = []
            for el in sec.find_all(["p", "li"]):
                txt = normalize_space(el.get_text(" "))
                if txt: parts.append(txt)
            candidate = normalize_space(" ".join(parts))
            if len(candidate) > 40:
                return candidate
    for sec in soup.find_all("section"):
        if (sec.get("aria-label") or "").lower() == "abstract":
            parts = []
            for el in sec.find_all(["p", "li"]):
                txt = normalize_space(el.get_text(" "))
                if txt: parts.append(txt)
            candidate = normalize_space(" ".join(parts))
            if len(candidate) > 40:
                return candidate
    return ""

def process_dataframe(
    df: pd.DataFrame,
    limit: Optional[int] = None,
    sleep: float = 0.5,
    checkpoint_path: Optional[str] = None,
) -> pd.DataFrame:
    data = df.copy()
    if limit is not None:
        data = data.head(limit)

    done: Dict[str, Tuple[str, str]] = {}
    if checkpoint_path and pathlib.Path(checkpoint_path).exists():
        ck = pd.read_csv(checkpoint_path)
        for _, r in ck.iterrows():
            done[str(r.get("Link", "")).strip()] = (
                str(r.get("Abstract", "") or ""),
                str(r.get("Error", "") or ""),
            )

    rows: List[Dict[str, str]] = []
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Extrayendo abstracts PMC"):
        title = str(row.get("Title", "") or "")
        url = str(row.get("Link", "") or "").strip()

        if not url:
            rows.append({"Title": title, "Link": url, "Abstract": "", "Error": "URL vacía"})
            continue

        if url in done:
            abs_text, err = done[url]
            rows.append({"Title": title, "Link": url, "Abstract": abs_text, "Error": err})
            continue

        abstract_text = ""
        error_msg = ""
        try:
            html = fetch_html(url)
            abstract_text = extract_abstract_from_html(html)
            if not abstract_text:
                error_msg = "Abstract no encontrado"
        except Exception as e:
            error_msg = repr(e)

        rows.append({"Title": title, "Link": url, "Abstract": abstract_text, "Error": error_msg})

        if checkpoint_path:
            pd.DataFrame(rows).to_csv(checkpoint_path, index=False, encoding="utf-8")

        time.sleep(max(0.0, sleep))

    return pd.DataFrame(rows, columns=["Title", "Link", "Abstract", "Error"])

def main() -> None:
    ap = argparse.ArgumentParser(description="Extrae Abstracts de páginas PMC desde un listado de enlaces.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    df = read_table(args.input)
    out_df = process_dataframe(df, limit=args.limit, sleep=args.sleep, checkpoint_path=args.checkpoint)
    out_df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"✅ Listo. Guardado: {args.output}")
    if args.checkpoint:
        print(f"💾 Checkpoint: {args.checkpoint}")

if __name__ == "__main__":
    main()
