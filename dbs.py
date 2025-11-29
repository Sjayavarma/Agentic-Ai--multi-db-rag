# dbs.py

import os
import faiss
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from models import embed_texts
from config import FAISS_INDEX_PATH, TOP_K


# ---------- Load CSVs ----------

DATA_DIR = "data"

VECTOR_DOCS_PATH = os.path.join(DATA_DIR, "vector_docs.csv")
SQL_DATA_PATH = os.path.join(DATA_DIR, "sql_data.csv")
GRAPH_DATA_PATH = os.path.join(DATA_DIR, "graph_nodes.csv")

vector_df = pd.read_csv(VECTOR_DOCS_PATH)   # id, text, category
sql_df = pd.read_csv(SQL_DATA_PATH)         # car, trips, avg_km, fuel_type, maintenance_score
graph_df = pd.read_csv(GRAPH_DATA_PATH)     # car, feature, relation


# ---------- FAISS Vector Index ----------

_faiss_index = None
_vector_id_map: List[int] = []  # map FAISS index row -> DataFrame index


def build_or_load_faiss_index():
    """
    Build FAISS index from vector_docs.csv using MiniLM embeddings.
    For simplicity, we rebuild every time; you can add saving/loading if needed.
    """
    global _faiss_index, _vector_id_map

    texts = vector_df["text"].tolist()
    embeddings = embed_texts(texts).astype("float32")
    dim = embeddings.shape[1]

    # Create index (L2)
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    _faiss_index = index
    _vector_id_map = list(vector_df.index)


def vector_search(query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Perform semantic search over vector_docs using FAISS.
    Returns list of row dicts from vector_df.
    """
    global _faiss_index, _vector_id_map

    if _faiss_index is None:
        build_or_load_faiss_index()

    q_emb = embed_texts([query]).astype("float32")
    distances, indices = _faiss_index.search(q_emb, k)  # shape: (1, k)

    results = []
    for idx in indices[0]:
        if idx == -1:
            continue
        row_idx = _vector_id_map[idx]
        row = vector_df.iloc[row_idx].to_dict()
        results.append(row)

    return results


# ---------- SQL-like DB (via pandas) ----------

def sql_query_by_car_keyword(keyword: str) -> List[Dict[str, Any]]:
    """
    Simple 'SQL' filter: SELECT * FROM sql_df WHERE car LIKE '%keyword%'.
    """
    keyword = keyword.lower()
    mask = sql_df["car"].str.lower().str.contains(keyword)
    return sql_df[mask].to_dict(orient="records")


# ---------- Graph DB (simple KG from CSV) ----------

def graph_query_by_car(car_name: str) -> List[Dict[str, Any]]:
    """
    Return all relations for a given car.
    """
    mask = graph_df["car"].str.lower() == car_name.lower()
    return graph_df[mask].to_dict(orient="records")
