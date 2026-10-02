import os

import numpy as np
from sentence_transformers import CrossEncoder

from backend import PDF_PATH, get_conn, embedder

USE_RERANK = True   # False = plain vector search (to compare)
FETCH_K = 15        # candidates pulled from the vector search
TOP_K = 3           # chunks kept for the LLM (after reranking)
MIN_SCORE = 0.15    # drop clearly unrelated chunks before reranking (0 to 1)

# small cross-encoder, runs on CPU; downloads once on first use
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_text(row):
    """Text the reranker reads for a row: summary for tables/images, raw text otherwise."""
    kind, page, original, summary, score = row
    return (summary if kind in ("table", "image") else original) or original or ""


def retrieve(question, file_path, top_k=TOP_K):
    """Vector search for candidates, then rerank them and keep the best few.

    Returns rows of: (chunk_type, page_number, original_content, summary, score)
    """
    source = os.path.basename(file_path)
    qvec = np.array(embedder.embed_query(question), dtype=np.float32)
    fetch = FETCH_K if USE_RERANK else top_k
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT chunk_type, page_number, original_content, summary,
                      1 - (embedding <=> %s) AS score
               FROM chunks WHERE source_file = %s
               ORDER BY embedding <=> %s LIMIT %s""",
            (qvec, source, qvec, fetch),
        ).fetchall()
    rows = [r for r in rows if r[4] >= MIN_SCORE]
    if not rows or not USE_RERANK:
        return rows[:top_k]

    # rerank: the cross-encoder reads question + chunk together, so it judges meaning better
    raw = reranker.predict([(question, rerank_text(r)) for r in rows])
    scores = 1 / (1 + np.exp(-raw))  # squash to 0-1
    ranked = sorted(zip(rows, scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [(r[0], r[1], r[2], r[3], float(s)) for r, s in ranked]


def build_context(rows):
    """Turn the retrieved rows into text the LLM can read."""
    parts = []
    for i, (kind, page, original, summary, score) in enumerate(rows, 1):
        if kind == "table":
            body = f"{summary}\nTable data:\n{(original or '')[:3000]}"
        elif kind == "image":
            body = f"Image description: {summary}"
        else:
            body = original
        parts.append(f"[Source {i} | {kind} | page {page}]\n{body}")
    return "\n\n".join(parts)


# Quick test without the LLM:  python ret.py
if __name__ == "__main__":
    q = input("Question: ")
    for kind, page, original, summary, score in retrieve(q, PDF_PATH):
        print(f"{score:.3f} | {kind} | page {page} | {(summary or original)[:100]}")