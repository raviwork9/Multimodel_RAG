import os

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

TABLE = "chunks_test"   # same table name as in check_llm.py

conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
register_vector(conn)

total = conn.execute(f"SELECT count(*) FROM {TABLE}").fetchone()[0]
print("Total rows:", total)
if total == 0:
    raise SystemExit("Table is empty. Run: python check_llm.py")

print("\nRows by type:")
for kind, n in conn.execute(
    f"SELECT chunk_type, count(*) FROM {TABLE} GROUP BY chunk_type ORDER BY 1"
):
    print(f"  {kind}: {n}")

print("\nSample rows:")
rows = conn.execute(
    f"""SELECT id, chunk_type, page_number,
               left(replace(coalesce(summary, original_content), chr(10), ' '), 100)
        FROM {TABLE} ORDER BY page_number LIMIT 5"""
).fetchall()
for r in rows:
    print(" ", r)

dim = conn.execute(f"SELECT vector_dims(embedding) FROM {TABLE} LIMIT 1").fetchone()[0]
vec = conn.execute(f"SELECT embedding FROM {TABLE} LIMIT 1").fetchone()[0]
print("\nEmbedding dimension:", dim, "(should be 384)")
print("First 5 numbers of one embedding:", vec.to_list()[:5])

query = input("\nType a test question: ")
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)
qvec = np.array(embedder.embed_query(query), dtype=np.float32)
hits = conn.execute(
    f"""SELECT chunk_type, page_number, 1 - (embedding <=> %s) AS score,
               left(replace(coalesce(summary, original_content), chr(10), ' '), 120)
        FROM {TABLE} ORDER BY embedding <=> %s LIMIT 3""",
    (qvec, qvec),
).fetchall()
print("\nTop 3 matches:")
for kind, page, score, text in hits:
    print(f"  {score:.3f} | {kind} | page {page} | {text}")

conn.close()