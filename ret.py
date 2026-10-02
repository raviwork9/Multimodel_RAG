import os

import numpy as np
from sentence_transformers import CrossEncoder

from backend import PDF_PATH, get_conn, embedder

USE_RERANK = True
FETCH_K = 15
TOP_K = 3
MIN_SCORE = 0.15

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank_text(row):
    kind, page, original, summary, score = row

    return (
        summary
        if kind in ("table", "image")
        else original
    ) or original or ""


def retrieve(question, file_path, top_k=TOP_K):
    source = os.path.basename(file_path)

    qvec = np.array(
        embedder.embed_query(question),
        dtype=np.float32
    )

    fetch = FETCH_K if USE_RERANK else top_k

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT chunk_type, page_number, original_content, summary,
                   1 - (embedding <=> %s) AS score
            FROM chunks
            WHERE source_file = %s
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (qvec, source, qvec, fetch),
        ).fetchall()

    rows = [
        r for r in rows
        if r[4] >= MIN_SCORE
    ]

    if not rows or not USE_RERANK:
        return rows[:top_k]

    raw = reranker.predict([
        (question, rerank_text(r))
        for r in rows
    ])

    scores = 1 / (1 + np.exp(-raw))

    ranked = sorted(
        zip(rows, scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    return [
        (
            r[0],
            r[1],
            r[2],
            r[3],
            float(s)
        )
        for r, s in ranked
    ]


def build_context(rows):
    parts = []

    for i, (kind, page, original, summary, score) in enumerate(
        rows,
        1
    ):
        if kind == "table":
            body = (
                f"{summary}\n"
                f"Table data:\n"
                f"{(original or '')[:3000]}"
            )
        elif kind == "image":
            body = (
                f"Image description: {summary}"
            )
        else:
            body = original

        parts.append(
            f"[Source {i} | {kind} | page {page}]\n"
            f"{body}"
        )

    return "\n\n".join(parts)


if __name__ == "__main__":
    q = input("Question: ")

    for kind, page, original, summary, score in retrieve(
        q,
        PDF_PATH
    ):
        print(
            f"{score:.3f} | {kind} | page {page} | "
            f"{(summary or original)[:100]}"
        )