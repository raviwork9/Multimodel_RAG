# Multimodel RAG

A PDF question-answering app that extracts text, tables, and images, stores vector embeddings in PostgreSQL with pgvector, and answers questions using Hugging Face models.

## Requirements

- Python 3.11 or newer
- PostgreSQL with the pgvector extension available
- A Hugging Face access token with access to the models configured in `backend.py`
- Tesseract OCR installed and available on `PATH` (the Windows installer commonly adds `C:\Program Files\Tesseract-OCR`)

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `DATABASE_URL` and `kk` in `.env`. The database user must be able to enable PostgreSQL extensions. The app creates its `chunks` table and pgvector indexes when it runs.

By default, the command-line pipeline uses `2018121815819.pdf` in the project directory. Set `PDF_PATH` in `.env` to use another PDF. The Streamlit app accepts uploaded PDFs.

## Run

Start the web app:

```powershell
streamlit run stream_ui.py
```

Run ingestion followed by an interactive question session:

```powershell
python main.py
```

Skip ingestion and ask questions about a document already in the database:

```powershell
python main.py ask
```

The included PDFs are small sample documents. Uploaded files, extracted images, local environments, and credentials are excluded from Git.