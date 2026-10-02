import os

import numpy as np
import psycopg
from psycopg.types.json import Jsonb
from pgvector.psycopg import register_vector
from dotenv import load_dotenv

from backend import (
    chunk_node, split_node, prepare_text_node,
    summarize_tables_node, summarize_images_node, embed_node,
)

load_dotenv()

PDF = "AI_in_Education_3_4_Page_Report.pdf"
LIMIT = 2                 # how many tables / images to test (None = all)
DATABASE_URL = os.environ["DATABASE_URL"]
TABLE = "chunks_test"     # test table, so your real "chunks" table stays clean


def show(text):
    return (text or "")[:450].replace("\n", " ")


state = {"file_path": PDF}

print("\n1. chunk_node")
state.update(chunk_node(state))
print("   docs:", len(state["docs"]))

print("\n2. split_node")
state.update(split_node(state))
if LIMIT:
    state["table_chunks"] = state["table_chunks"][:LIMIT]
    state["image_chunks"] = state["image_chunks"][:LIMIT]

print("\n3. prepare_text_node")
state.update(prepare_text_node(state))
print("   text records:", len(state["text_records"]))
for r in state["text_records"][:2]:
    print("   page", r["page_number"], "|", show(r["embed_text"]))

print("\n4. summarize_tables_node")
state.update(summarize_tables_node(state))
print("   table records:", len(state["table_records"]))
for r in state["table_records"]:
    print("   page", r["page_number"], "|", show(r["summary"]))

print("\n5. summarize_images_node")
state.update(summarize_images_node(state))
print("   image records:", len(state["image_records"]))
for r in state["image_records"]:
    print("   page", r["page_number"], "|", show(r["summary"]))

print("\n6. embed_node")
state.update(embed_node(state))
print("   embedding length:", len(state["embeddings"][0]), "(should be 384)")

print("\n7. store in postgres")
conn = psycopg.connect(DATABASE_URL, autocommit=True)
conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
register_vector(conn)
conn.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE} (
        id TEXT PRIMARY KEY,
        source_file TEXT,
        chunk_type TEXT,
        page_number INT,
        original_content TEXT,
        summary TEXT,
        image_path TEXT,
        metadata JSONB,
        embedding vector(384)
    )
""")

source = os.path.basename(PDF)
for r, emb in zip(state["records"], state["embeddings"]):
    conn.execute(
        f"""
        INSERT INTO {TABLE}
            (id, source_file, chunk_type, page_number, original_content,
             summary, image_path, metadata, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            original_content = EXCLUDED.original_content,
            summary = EXCLUDED.summary,
            embedding = EXCLUDED.embedding
        """,
        (r["id"], source, r["chunk_type"], r["page_number"],
         r["original_content"], r["summary"], r["image_path"],
         Jsonb(r["metadata"]), np.array(emb, dtype=np.float32)),
    )
print(f"   stored {len(state['records'])} rows in table '{TABLE}'")
conn.close()