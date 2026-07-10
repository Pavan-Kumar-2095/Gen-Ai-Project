# DocuQuery — LLM-Powered Document Q&A API

DocuQuery is a high-performance Retrieval-Augmented Generation (RAG) system built using **FastAPI, FAISS, Sentence Transformers, and Gemini 1.5 Flash**.

It enables users to:

- Upload documents via URL (PDF, DOCX, EML)
- Automatically index documents using vector embeddings
- Ask multiple questions in a single request
- Receive concise, context-aware AI-generated answers

---

## Demo Video

https://github.com/user-attachments/assets/74d997e7-212f-4491-b834-d5a0c5b399b1

---

# Architecture Overview

DocuQuery follows a Retrieval-Augmented Generation (RAG) pipeline:

```
Document URL
      │
      ▼
Text Extraction (PDF / DOCX / EML)
      │
      ▼
Chunking (LangChain Recursive Splitter)
      │
      ▼
Embeddings (all-MiniLM-L6-v2)
      │
      ▼
FAISS Vector Index
      │
      ▼
Question Embeddings
      │
      ▼
Semantic Search (Top-K Chunks)
      │
      ▼
Gemini 1.5 Flash
      │
      ▼
Structured Responses
```

---

# Features

- Supports PDF, DOCX, and EML documents via URL
- Fast semantic search using FAISS (Inner Product similarity)
- SentenceTransformers (`all-MiniLM-L6-v2`) for embedding generation
- Batch processing for multiple questions
- Bearer token authentication
- Structured AI-generated responses
- Asynchronous FastAPI backend
- Thread-safe FAISS indexing

---

# Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI, Uvicorn |
| Vector Database | FAISS |
| Embeddings | SentenceTransformers |
| LLM | Gemini 1.5 Flash |
| Text Processing | LangChain |
| Document Parsing | PyMuPDF, Mammoth, Python Email Parser |
| Networking | aiohttp, requests |
| Language | Python 3.10+ |

---

# Project Structure

```text
Server.py          Main FastAPI backend
Client.html        Frontend UI
requirements.txt   Project dependencies
```

---

# Setup & Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Pavan-Kumar-2095/Gen-Ai-Project.git
cd DocuQuery
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Configure Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key
BEARER_KEY=your_bearer_token
PORT=8000
```

## 4. Run the Backend Server

```bash
python Server.py
```

The server will be available at:

```
http://localhost:8000
```

## 5. Open the Frontend

Open the following file in your browser:

```
Client.html
```

---

# How It Works

## Step 1: Upload a Document

Provide a publicly accessible URL for one of the following document types:

- PDF
- DOCX
- EML

## Step 2: Automatic Processing

The system automatically:

1. Extracts document text
2. Splits the content into chunks
3. Generates vector embeddings
4. Stores embeddings in a FAISS index

## Step 3: Ask Questions

Example queries:

- Summarize this document.
- What are the key points?
- What does Section 2 explain?
- Who are the people mentioned?

## Step 4: Receive AI-Generated Answers

For every query:

- Relevant document chunks are retrieved through semantic search.
- Gemini 1.5 Flash generates grounded responses using the retrieved context.
- The API returns concise, structured answers.

---

# Use Cases

- Document Question Answering
- Enterprise Knowledge Base Search
- Retrieval-Augmented Generation (RAG) Applications
- AI Document Assistants
- Research Document Analysis
- Legal Document Search

---

# LinkedIn Post

https://www.linkedin.com/feed/update/urn:li:activity:7359036194195681280/