import os
import sys
from typing import TypedDict, List

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

# the nodes and helpers already written in backend.py
from backend import (
    PDF_PATH, Record, get_conn, init_schema, chat_model,
    chunk_node, split_node, prepare_text_node,
    summarize_tables_node, summarize_images_node,
    embed_node, store_postgres_node,
)

# the retrieval step lives in ret.py
from ret import retrieve, build_context


# ---------------------------------------------------------------
# 1. State: everything that flows through the graphs
# ---------------------------------------------------------------
class MainState(TypedDict, total=False):
    # ingestion
    file_path: str
    docs: List[Document]
    text_chunks: List[Document]
    table_chunks: List[Document]
    image_chunks: List[Document]
    text_records: List[Record]
    table_records: List[Record]
    image_records: List[Record]
    records: List[Record]
    embeddings: List[List[float]]
    # question answering
    question: str
    rows: list
    context: str
    answer: str


# ---------------------------------------------------------------
# 2. New nodes (the rest come from backend.py)
# ---------------------------------------------------------------
def reset_node(state: MainState) -> dict:
    """Start from the beginning: remove old rows of this PDF before re-ingesting."""
    source = os.path.basename(state["file_path"])
    with get_conn() as conn:
        init_schema(conn)
        conn.execute("DELETE FROM chunks WHERE source_file = %s", (source,))
    print(f"Cleared old rows for {source}")
    return {}


def retrieve_node(state: MainState) -> dict:
    """Retrieval step: implemented in ret.py."""
    return {"rows": retrieve(state["question"], state["file_path"])}


def context_node(state: MainState) -> dict:
    """Turn the retrieved rows into text for the LLM: implemented in ret.py."""
    return {"context": build_context(state["rows"])}


def generate_node(state: MainState) -> dict:
    """Ask the LLM to answer using only the retrieved context."""
    if not state["rows"]:
        return {"answer": "I could not find anything relevant in the document."}

    system = (
        "You answer questions about a document using ONLY the context provided by the user. "
        "Read the context carefully and pay attention to words like 'same', 'not', 'only'. "
        "For true/false questions answer True or False. "
        "If the question asks you to explain something, the answer can be several sentences. "
        "If the context does not answer the question, say you could not find it in the document. "
        "Always reply in exactly this format:\n"
        "Quote: <the exact sentence from the context that answers the question>\n"
        "Answer: <your answer based only on the context> (page <number>)"
    )
    user = f"Context:\n{state['context']}\n\nQuestion: {state['question']}"

    reply = chat_model.invoke([SystemMessage(content=system), HumanMessage(content=user)]).content
    if "Answer:" not in reply:
        retry = user + "\n\nNow answer this question using the context above, in the Quote/Answer format."
        reply = chat_model.invoke([SystemMessage(content=system), HumanMessage(content=retry)]).content
    return {"answer": reply}


# ---------------------------------------------------------------
# 3. Graph 1: ingestion (runs once per run, from the beginning)
# ---------------------------------------------------------------
ingest = StateGraph(MainState)
ingest.add_node("reset", reset_node)
ingest.add_node("chunk", chunk_node)
ingest.add_node("split", split_node)
ingest.add_node("prepare_text", prepare_text_node)
ingest.add_node("summarize_tables", summarize_tables_node)
ingest.add_node("summarize_images", summarize_images_node)
ingest.add_node("embed", embed_node)
ingest.add_node("store_postgres", store_postgres_node)

ingest.add_edge(START, "reset")
ingest.add_edge("reset", "chunk")
ingest.add_edge("chunk", "split")
# these three run in parallel
ingest.add_edge("split", "prepare_text")
ingest.add_edge("split", "summarize_tables")
ingest.add_edge("split", "summarize_images")
# embed waits for all three
ingest.add_edge(["prepare_text", "summarize_tables", "summarize_images"], "embed")
ingest.add_edge("embed", "store_postgres")
ingest.add_edge("store_postgres", END)
ingest_graph = ingest.compile()


# ---------------------------------------------------------------
# 4. Graph 2: question answering (runs once per question)
# ---------------------------------------------------------------
qa = StateGraph(MainState)
qa.add_node("retrieve", retrieve_node)
qa.add_node("context", context_node)
qa.add_node("generate", generate_node)

qa.add_edge(START, "retrieve")
qa.add_edge("retrieve", "context")
qa.add_edge("context", "generate")
qa.add_edge("generate", END)
qa_graph = qa.compile()


# ---------------------------------------------------------------
# 5. Run
#    python main.py        -> ingest from the beginning, then ask questions
#    python main.py ask    -> skip ingestion, just ask questions
# ---------------------------------------------------------------
if __name__ == "__main__":
    if "ask" not in sys.argv:
        print("\n=== INGESTION ===")
        ingest_graph.invoke({"file_path": PDF_PATH})
        print("\nIngestion finished.")

    print("\n=== ASK QUESTIONS ===")
    while True:
        question = input("\nAsk a question (or press Enter to quit): ").strip()
        if not question:
            break
        result = qa_graph.invoke({"file_path": PDF_PATH, "question": question})
        print("\nANSWER:\n" + result["answer"])
        print("\nSOURCES USED:")
        for kind, page, original, summary, score in result["rows"]:
            print(f"  {score:.3f} | {kind} | page {page}")