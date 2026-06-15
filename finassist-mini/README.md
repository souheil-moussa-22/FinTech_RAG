# finassist-mini

`finassist-mini` is an educational Retrieval-Augmented Generation (RAG) project built with Python + FastAPI for financial PDF question answering.

## 1) Why this project exists

This project demonstrates a complete but compact RAG pipeline:

1. Upload PDF documents
2. Parse page text
3. Split text into overlapping chunks
4. Convert chunks to embeddings
5. Store embeddings in ChromaDB
6. Retrieve Top-K relevant chunks for a question
7. Ground an LLM prompt with retrieved evidence
8. Return answer + explicit sources (document/page)

The assistant is intentionally grounded to uploaded content so it can answer only from your own documents.

---

## 2) Project structure and responsibilities

```text
finassist-mini/
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── chat.py
│   │   └── documents.py
│   ├── services/
│   │   ├── parser.py
│   │   ├── chunker.py
│   │   ├── embedding.py
│   │   ├── vector_store.py
│   │   ├── rag.py
│   │   └── llm.py
│   ├── models/
│   │   └── document.py
│   ├── schemas/
│   │   ├── chat.py
│   │   └── document.py
│   └── config.py
├── data/
│   └── documents/
├── requirements.txt
├── .env.example
└── README.md
```

### File-by-file explanation

- `app/main.py`: FastAPI app bootstrap, startup/shutdown lifecycle, router registration, health endpoint.
- `app/config.py`: centralized settings from environment variables.
- `app/api/documents.py`: upload/list/delete endpoints; orchestrates parser/chunker/embedding/vector indexing flow for uploaded PDFs.
- `app/api/chat.py`: `/chat` endpoint that executes the RAG pipeline.
- `app/services/parser.py`: PDF text extraction with page numbers (PyMuPDF).
- `app/services/chunker.py`: 500-word style chunking with 50-word overlap.
- `app/services/embedding.py`: sentence-transformers model loader + vector generation.
- `app/services/vector_store.py`: persistent ChromaDB collection operations (insert/search/delete).
- `app/services/llm.py`: Ollama integration for answer generation using Mistral.
- `app/services/rag.py`: pipeline orchestrator (embed question → retrieve → prompt → generate answer).
- `app/models/document.py`: shared domain models for parsed pages/chunks/retrieval records.
- `app/schemas/*.py`: Pydantic request/response validation.

Design principle: routes stay thin, services own business logic, schemas validate contracts, models carry domain data.

---

## 3) Core RAG concepts (simple explanations)

### Embeddings
An embedding is a numeric vector that captures semantic meaning of text.

Example:
- "international transfer fee" and "cross-border transfer charge" become close vectors.

Why needed: vector similarity lets us retrieve by meaning, not exact keywords.

### Semantic search
Instead of matching exact words, we compare vector distance/similarity between question embedding and chunk embeddings.

### Chunking + overlap
Long pages are split into smaller chunks (~500 words) for precise retrieval.

Why overlap (~50 words) matters:
- It preserves context across chunk boundaries.
- If one sentence starts at the end of chunk A and ends at start of chunk B, overlap ensures at least one chunk keeps enough context.

### Cosine similarity and Top-K retrieval
Cosine similarity measures angle alignment between vectors (closer direction = more similar meaning).
Top-K retrieval means return the K most similar chunks for prompt grounding.

### Prompt grounding
The LLM prompt includes only retrieved chunks and explicit instruction to answer strictly from that context.

---

## 4) API endpoints

### Health
- `GET /health`

### Documents
- `POST /documents/upload` (PDF only)
- `GET /documents`
- `DELETE /documents/{id}`

### Chat
- `POST /chat`

Request:
```json
{
  "question": "What are the international transfer fees?"
}
```

Response:
```json
{
  "answer": "...",
  "sources": [
    {
      "document": "fees.pdf",
      "page": 2
    }
  ]
}
```

---

## 5) Execution flow (end-to-end)

```text
User uploads PDF
   |
   v
Parser extracts page text
   |
   v
Chunker splits into overlapping chunks
   |
   v
Embedding service creates vectors
   |
   v
Vector store persists vectors in Chroma

User asks question
   |
   v
Question embedding
   |
   v
Top-K semantic retrieval
   |
   v
Prompt built with retrieved chunks
   |
   v
Mistral via Ollama generates answer
   |
   v
API returns answer + sources (doc/page)
```

---

## 6) Setup and run

1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy env file:
   ```bash
   cp .env.example .env
   ```
4. Ensure Ollama is running and model is available:
   ```bash
   ollama pull mistral
   ollama serve
   ```
5. Start API:
   ```bash
   uvicorn app.main:app --reload
   ```

Open Swagger at: `http://127.0.0.1:8000/docs`

---

## 7) Function-level guidance

All classes/functions include docstrings describing:
- inputs
- outputs
- execution flow
- failure behavior

Use those docstrings as your code-reading guide when learning each step.
