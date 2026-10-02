import os
import base64
import hashlib
from typing import TypedDict, List, Optional

import numpy as np
import psycopg
from psycopg.types.json import Jsonb
from pgvector.psycopg import register_vector
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_core.documents import Document
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

load_dotenv()
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Tesseract-OCR"

PDF_PATH = os.getenv("PDF_PATH", "2018121815819.pdf")
DATABASE_URL = os.environ["DATABASE_URL"]
HF_TOKEN = os.environ["kk"]

EMBED_DIM = 384
VISION_MODEL = "Qwen/Qwen3.8-27B"
MAX_TABLE_CHARS = 6000

TABLE_PROMPT = (
    "You are summarizing a table extracted from a document. Describe what the "
    "table is about, its columns/rows, and the key figures and trends. Keep the "
    "important words, names and numbers so the summary is searchable.\n\n{table}"
)
IMAGE_PROMPT = (
    "Describe this image from a document in detail. Include any visible text, "
    "labels, axis names, numbers, and what the figure shows. Keep important "
    "words so the description is searchable."
)


class Record(TypedDict):
    id: str
    chunk_type: str
    page_number: Optional[int]
    original_content: str
    summary: Optional[str]
    embed_text: str
    image_path: Optional[str]
    metadata: dict


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=300,
    temperature=0.1,
    huggingfacehub_api_token=HF_TOKEN,
)
chat_model = ChatHuggingFace(llm=llm)
vision_client = InferenceClient(model=VISION_MODEL, token=HF_TOKEN)
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)


def get_conn() -> psycopg.Connection:
    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id               TEXT PRIMARY KEY,
            source_file      TEXT NOT NULL,
            chunk_type       TEXT NOT NULL CHECK (chunk_type IN ('text', 'table', 'image')),
            page_number      INT,
            original_content TEXT,
            summary          TEXT,
            image_path       TEXT,
            metadata         JSONB,
            embedding        vector({EMBED_DIM}) NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
        "ON chunks USING hnsw (embedding vector_cosine_ops)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS chunks_source_idx ON chunks (source_file)")


def make_id(source: str, chunk_type: str, page, basis: str) -> str:
    h = hashlib.sha256(f"{source}|{chunk_type}|{page}|{basis}".encode("utf-8")).hexdigest()[:16]
    return f"{chunk_type}_{h}"


def clean_metadata(meta: dict) -> dict:
    keep = ("filename", "page_number", "category", "languages")
    return {k: meta[k] for k in keep if k in meta}


def caption_image(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    mime = "image/jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "image/png"
    resp = vision_client.chat_completion(
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": IMAGE_PROMPT},
            ],
        }],
        max_tokens=300,
    )
    return resp.choices[0].message.content


def elements_to_docs(elements, max_chars=1000, combine_under=200):
    text_els = [e for e in elements if e.category not in ("Table", "Image")]
    tables = [e for e in elements if e.category == "Table"]
    images = [e for e in elements if e.category == "Image"]
    chunks = chunk_by_title(
        text_els, max_characters=max_chars, combine_text_under_n_chars=combine_under
    )
    docs = []
    for e in chunks + tables + images:
        meta = e.metadata.to_dict()
        meta.pop("orig_elements", None)
        meta["category"] = e.category
        docs.append(Document(page_content=e.text or "", metadata=meta))
    docs.sort(key=lambda d: d.metadata.get("page_number") or 0)
    return docs


def chunk_node(state: dict) -> dict:
    elements = partition_pdf(
        filename=state["file_path"],
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=False,
        extract_image_block_output_dir="extracted_images",
    )
    return {"docs": elements_to_docs(elements)}


def split_node(state: dict) -> dict:
    text_chunks, table_chunks, image_chunks = [], [], []
    for doc in state["docs"]:
        category = doc.metadata.get("category")
        if category == "Table":
            table_chunks.append(doc)
        elif category == "Image":
            image_chunks.append(doc)
        else:
            text_chunks.append(doc)

    print(f"Text chunks: {len(text_chunks)}")
    print(f"Table chunks: {len(table_chunks)}")
    print(f"Image chunks: {len(image_chunks)}")
    return {
        "text_chunks": text_chunks,
        "table_chunks": table_chunks,
        "image_chunks": image_chunks,
    }


def prepare_text_node(state: dict) -> dict:
    source = os.path.basename(state["file_path"])
    records: List[Record] = []
    for doc in state["text_chunks"]:
        text = doc.page_content.strip()
        if not text:
            continue
        page = doc.metadata.get("page_number")
        records.append({
            "id": make_id(source, "text", page, text),
            "chunk_type": "text",
            "page_number": page,
            "original_content": text,
            "summary": None,
            "embed_text": text,
            "image_path": None,
            "metadata": clean_metadata(doc.metadata),
        })
    return {"text_records": records}


def summarize_tables_node(state: dict) -> dict:
    source = os.path.basename(state["file_path"])
    docs, originals, prompts = [], [], []
    for doc in state["table_chunks"]:
        table = doc.metadata.get("text_as_html") or doc.page_content
        if not table or not table.strip():
            continue
        docs.append(doc)
        originals.append(table)
        prompts.append(TABLE_PROMPT.format(table=table[:MAX_TABLE_CHARS]))

    summaries = (
        [m.content for m in chat_model.batch(prompts, config={"max_concurrency": 4})]
        if prompts else []
    )

    records: List[Record] = []
    for doc, original, summary in zip(docs, originals, summaries):
        page = doc.metadata.get("page_number")
        records.append({
            "id": make_id(source, "table", page, original),
            "chunk_type": "table",
            "page_number": page,
            "original_content": original,
            "summary": summary,
            "embed_text": summary,
            "image_path": doc.metadata.get("image_path"),
            "metadata": clean_metadata(doc.metadata),
        })
    return {"table_records": records}


def summarize_images_node(state: dict) -> dict:
    source = os.path.basename(state["file_path"])
    records: List[Record] = []
    for doc in state["image_chunks"]:
        path = doc.metadata.get("image_path")
        ocr_text = (doc.page_content or "").strip()
        page = doc.metadata.get("page_number")

        summary = None
        if path and os.path.exists(path):
            try:
                summary = caption_image(path)
            except Exception as e:
                print(f"Vision captioning failed for {path}: {e}")
        if not summary:
            summary = ocr_text
        if not summary:
            continue

        records.append({
            "id": make_id(source, "image", page, path or ocr_text),
            "chunk_type": "image",
            "page_number": page,
            "original_content": ocr_text,
            "summary": summary,
            "embed_text": summary,
            "image_path": path,
            "metadata": clean_metadata(doc.metadata),
        })
    return {"image_records": records}


def embed_node(state: dict) -> dict:
    records = (
        state.get("text_records", [])
        + state.get("table_records", [])
        + state.get("image_records", [])
    )
    if not records:
        raise ValueError("No records to embed - check the chunking output.")
    embeddings = embedder.embed_documents([r["embed_text"] for r in records])
    print(f"Embedded {len(records)} records")
    return {"records": records, "embeddings": embeddings}


def store_postgres_node(state: dict) -> dict:
    source = os.path.basename(state["file_path"])
    rows = [
        (
            r["id"], source, r["chunk_type"], r["page_number"],
            r["original_content"], r["summary"], r["image_path"],
            Jsonb(r["metadata"]), np.array(emb, dtype=np.float32),
        )
        for r, emb in zip(state["records"], state["embeddings"])
    ]
    with get_conn() as conn:
        init_schema(conn)
        with conn.transaction():
            conn.cursor().executemany(
                """
                INSERT INTO chunks
                    (id, source_file, chunk_type, page_number, original_content,
                     summary, image_path, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    source_file      = EXCLUDED.source_file,
                    chunk_type       = EXCLUDED.chunk_type,
                    page_number      = EXCLUDED.page_number,
                    original_content = EXCLUDED.original_content,
                    summary          = EXCLUDED.summary,
                    image_path       = EXCLUDED.image_path,
                    metadata         = EXCLUDED.metadata,
                    embedding        = EXCLUDED.embedding
                """,
                rows,
            )
        total = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    print(f"Upserted {len(rows)} rows ({total} total in table)")
    return {}