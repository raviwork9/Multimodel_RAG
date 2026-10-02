"""
Inspect your document loader + chunking without running the whole RAG pipeline.

Usage:
    python inspect_chunks.py                      # uses DEFAULT_PDF
    python inspect_chunks.py path\\to\\file.pdf
    python inspect_chunks.py file.pdf --out my_report_folder

Outputs (in ./chunk_check/):
    chunks.json   every chunk with full text + metadata
    report.html   open in a browser to eyeball chunks, tables and images
"""
import os
import json
import time
import html
import argparse
import statistics
from collections import Counter, defaultdict

from dotenv import load_dotenv
from langchain_core.documents import Document
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

load_dotenv()
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Tesseract-OCR"

DEFAULT_PDF = "2018121815819.pdf"
IMG_DIR = "extracted_images"
MAX_CHARS = 1000        # keep in sync with your pipeline
COMBINE_UNDER = 200     # keep in sync with your pipeline
TINY_CHARS = 50         # chunks shorter than this are suspicious (page numbers, stray headers)

KEEP_META = ("category", "page_number", "filename", "image_path",
             "text_as_html", "languages", "element_id", "parent_id")


# ---------------------------------------------------------------
# Loading (same settings as the pipeline's chunk_node)
# ---------------------------------------------------------------
def load_chunks(pdf_path: str):
    # 1) Partition WITHOUT chunking so Image/Table elements keep their identity
    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=False,
        extract_image_block_output_dir=IMG_DIR,
    )
    raw_counts = Counter(e.category for e in elements)
    print("Raw elements before chunking:", dict(raw_counts))

    # 2) Chunk only the text; keep tables and images as standalone elements
    text_els = [e for e in elements if e.category not in ("Table", "Image")]
    tables = [e for e in elements if e.category == "Table"]
    images = [e for e in elements if e.category == "Image"]
    chunks = chunk_by_title(
        text_els,
        max_characters=MAX_CHARS,
        combine_text_under_n_chars=COMBINE_UNDER,
    )

    docs = []
    for e in chunks + tables + images:
        meta = e.metadata.to_dict()
        meta.pop("orig_elements", None)  # large base64 blob, not needed
        meta["category"] = e.category
        docs.append(Document(page_content=e.text or "", metadata=meta))
    docs.sort(key=lambda d: d.metadata.get("page_number") or 0)  # stable: page order
    return docs


def one_line(text: str, limit: int) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= limit else t[:limit] + "..."


# ---------------------------------------------------------------
# Checks
# ---------------------------------------------------------------
def run_checks(docs):
    warnings = []
    by_cat = defaultdict(list)
    for i, d in enumerate(docs):
        by_cat[d.metadata.get("category", "UNKNOWN")].append((i, d))

    for cat, items in by_cat.items():
        for i, d in items:
            text = (d.page_content or "").strip()
            page = d.metadata.get("page_number")
            tag = f"[#{i} {cat} p.{page}]"

            if cat in ("Table", "TableChunk"):
                if not d.metadata.get("text_as_html"):
                    warnings.append(f"{tag} table has no text_as_html (infer_table_structure may have failed)")
                if not text:
                    warnings.append(f"{tag} table has empty text")
            elif cat == "Image":
                path = d.metadata.get("image_path")
                if not path:
                    warnings.append(f"{tag} image has no image_path in metadata")
                elif not os.path.exists(path):
                    warnings.append(f"{tag} image_path does not exist on disk: {path}")
            else:
                if not text:
                    warnings.append(f"{tag} EMPTY chunk")
                elif len(text) < TINY_CHARS:
                    warnings.append(f"{tag} tiny chunk ({len(text)} chars): {one_line(text, 60)!r}")
                if len(text) > MAX_CHARS:
                    warnings.append(f"{tag} chunk exceeds max_characters ({len(text)} > {MAX_CHARS})")

    # pages that produced no chunks at all
    pages = {d.metadata.get("page_number") for d in docs if d.metadata.get("page_number")}
    if pages:
        missing = sorted(set(range(1, max(pages) + 1)) - pages)
        if missing:
            warnings.append(f"pages with NO chunks: {missing} (blank pages, or loader missed them)")
    return by_cat, warnings


# ---------------------------------------------------------------
# Console report
# ---------------------------------------------------------------
def print_report(docs, by_cat, warnings, elapsed):
    print("=" * 70)
    print(f"Loaded {len(docs)} chunks in {elapsed:.1f}s")
    print("=" * 70)

    print("\nChunks by category:")
    for cat, items in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        print(f"  {cat:<12} {len(items)}")

    page_counts = Counter(d.metadata.get("page_number") for d in docs)
    print(f"\nPages covered: {len([p for p in page_counts if p])}")

    text_lens = [len((d.page_content or "")) for cat, items in by_cat.items()
                 if cat not in ("Table", "TableChunk", "Image") for _, d in items]
    if text_lens:
        print("\nText chunk length (chars):")
        print(f"  min {min(text_lens)} | median {int(statistics.median(text_lens))} | "
              f"mean {int(statistics.mean(text_lens))} | max {max(text_lens)}")
        under = sum(1 for n in text_lens if n < COMBINE_UNDER)
        print(f"  under combine threshold ({COMBINE_UNDER}): {under}")

    print("\nExtracted image files:",
          len(os.listdir(IMG_DIR)) if os.path.isdir(IMG_DIR) else f"(folder '{IMG_DIR}' not found)")

    print("\nWarnings:" if warnings else "\nNo warnings.")
    for w in warnings[:40]:
        print("  -", w)
    if len(warnings) > 40:
        print(f"  ... and {len(warnings) - 40} more (see report.html / chunks.json)")


# ---------------------------------------------------------------
# Exports
# ---------------------------------------------------------------
def export_json(docs, out_dir):
    rows = []
    for i, d in enumerate(docs):
        meta = {k: d.metadata[k] for k in KEEP_META if k in d.metadata}
        rows.append({"index": i, "length": len(d.page_content or ""),
                     "text": d.page_content, "metadata": meta})
    path = os.path.join(out_dir, "chunks.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False, default=str)
    return path


def export_html(docs, out_dir, warnings):
    colors = {"Table": "#0F6E56", "TableChunk": "#0F6E56", "Image": "#993C1D"}
    parts = [
        "<html><head><meta charset='utf-8'><title>Chunk report</title><style>",
        "body{font-family:sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#222}",
        ".c{border:1px solid #ddd;border-radius:8px;padding:12px;margin:12px 0}",
        ".b{display:inline-block;color:#fff;border-radius:4px;padding:2px 8px;font-size:12px;margin-right:8px}",
        ".m{color:#666;font-size:12px}pre{white-space:pre-wrap;font-size:13px}",
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px;font-size:13px}",
        "img{max-width:100%;border:1px solid #eee}.w{background:#fff4e5;padding:8px;border-radius:6px}",
        "</style></head><body>",
        f"<h2>{len(docs)} chunks</h2>",
    ]
    if warnings:
        parts.append("<div class='w'><b>Warnings</b><ul>"
                     + "".join(f"<li>{html.escape(w)}</li>" for w in warnings) + "</ul></div>")

    for i, d in enumerate(docs):
        m = d.metadata
        cat = m.get("category", "UNKNOWN")
        color = colors.get(cat, "#534AB7")
        parts.append(f"<div class='c'><span class='b' style='background:{color}'>{html.escape(cat)}</span>"
                     f"<span class='m'>#{i} | page {m.get('page_number')} | {len(d.page_content or '')} chars</span>")
        if m.get("text_as_html"):
            parts.append(m["text_as_html"])  # rendered table
        else:
            parts.append(f"<pre>{html.escape(d.page_content or '')}</pre>")
        path = m.get("image_path")
        if path and os.path.exists(path):
            rel = os.path.relpath(path, out_dir).replace("\\", "/")
            parts.append(f"<img src='{html.escape(rel)}'>")
        parts.append("</div>")

    parts.append("</body></html>")
    path = os.path.join(out_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return path


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Inspect loader + chunking output")
    ap.add_argument("pdf", nargs="?", default=DEFAULT_PDF)
    ap.add_argument("--out", default="chunk_check")
    ap.add_argument("--no-html", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.pdf):
        raise SystemExit(f"PDF not found: {args.pdf}")
    os.makedirs(args.out, exist_ok=True)

    start = time.time()
    docs = load_chunks(args.pdf)
    elapsed = time.time() - start

    by_cat, warnings = run_checks(docs)
    print_report(docs, by_cat, warnings, elapsed)

    print("\nSaved:", export_json(docs, args.out))
    if not args.no_html:
        print("Saved:", export_html(docs, args.out, warnings), "(open in a browser)")


if __name__ == "__main__":
    main()