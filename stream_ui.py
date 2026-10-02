import os

import streamlit as st

from backend import get_conn, init_schema
from main import ingest_graph, qa_graph

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


st.set_page_config(page_title="Document Chatbot", page_icon="📄")
st.title("📄 Document Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_doc" not in st.session_state:
    st.session_state.last_doc = None


def processed_docs():
    try:
        with get_conn() as conn:
            init_schema(conn)
            rows = conn.execute("SELECT DISTINCT source_file FROM chunks ORDER BY 1").fetchall()
        return [r[0] for r in rows]
    except Exception as e:
        st.sidebar.error(f"Database error: {e}")
        return []


def split_answer(text):
    if "Answer:" in text:
        quote, answer = text.split("Answer:", 1)
        return answer.strip(), quote.replace("Quote:", "").strip()
    return text.strip(), ""


def show_extras(quote, sources):
    if quote:
        with st.expander("Evidence quoted by the model"):
            st.write(quote)
    if sources:
        with st.expander(f"Sources used ({len(sources)})"):
            for s in sources:
                st.markdown(f"**{s['type']}** · page {s['page']} · score {s['score']:.2f}")
                st.caption(s["text"])


with st.sidebar:
    st.header("Document")
    uploaded = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded and st.button("Process document", type="primary"):
        path = os.path.join(UPLOAD_DIR, uploaded.name)
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())

        with st.status("Processing document. This can take a few minutes.", expanded=True) as status:
            try:
                for update in ingest_graph.stream({"file_path": path}, stream_mode="updates"):
                    for node in update:
                        st.write(f"Finished: {node}")
                status.update(label="Document ready", state="complete")
                st.session_state["doc_choice"] = uploaded.name
            except Exception as e:
                status.update(label="Processing failed", state="error")
                st.error(str(e))

    docs = processed_docs()
    doc = st.selectbox("Chat with document", docs, key="doc_choice") if docs else None

    if st.button("Clear chat"):
        st.session_state.messages = []

if doc != st.session_state.last_doc:
    st.session_state.messages = []
    st.session_state.last_doc = doc

if not doc:
    st.info("Upload a PDF in the sidebar and click **Process document** to start.")
    st.stop()

st.caption(f"Chatting with: {doc}")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            show_extras(m.get("quote"), m.get("sources"))

question = st.chat_input("Ask a question about the document")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching and thinking..."):
                result = qa_graph.invoke({"file_path": doc, "question": question})
            answer, quote = split_answer(result["answer"])
            sources = [
                {
                    "type": kind,
                    "page": page,
                    "score": float(score),
                    "text": ((summary if kind in ("table", "image") else original) or "")[:300],
                }
                for kind, page, original, summary, score in result["rows"]
            ]
            st.markdown(answer)
            show_extras(quote, sources)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "quote": quote, "sources": sources}
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")