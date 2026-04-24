# 🚀 DocuQuery — LLM-Powered Document Q&A API

DocuQuery is a **high-performance Retrieval-Augmented Generation (RAG) system** built using **FastAPI, FAISS, Sentence Transformers, and Gemini 1.5 Flash**.

It enables users to:
- Upload documents via URL (PDF / DOCX / EML)
- Automatically index them using vector embeddings
- Ask multiple questions in a single request
- Receive concise, context-aware AI-generated answers

---
## 🎥 Demo Video

<video width="100%" controls>
  <source src="<video width="100%" controls>
  <source src="https://raw.githubusercontent.com/Pavan-Kumar-2095/Gen-Ai-Project/main/DemoVideo/DemoVideo.mp4" type="video/mp4">
</video>" type="video/mp4">
</video>

# 🧠 Architecture Overview

DocuQuery uses a **RAG pipeline**:

📥 Document URL  
→ 🧾 Text Extraction (PDF / DOCX / EML)  
→ ✂️ Chunking (LangChain Recursive Splitter)  
→ 🧠 Embeddings (all-MiniLM-L6-v2)  
→ 🔍 FAISS Vector Index  
→ ❓ Question Embeddings  
→ 🎯 Semantic Search (Top-K chunks)  
→ ✨ Gemini 1.5 Flash (Answer Generation)  
→ 📤 Structured Responses  

---

# ⚙️ Features

- 📄 Supports PDF, DOCX, EML documents via URL
- ⚡ Fast semantic search using FAISS (Inner Product similarity)
- 🧠 Embeddings using SentenceTransformers (`all-MiniLM-L6-v2`)
- 🔁 Batch processing for multiple questions
- 🔒 Bearer token authentication support
- 🧾 Structured answer formatting per question
- 🚀 Async FastAPI backend
- 🧵 Thread-safe FAISS indexing

---

# 🛠️ Tech Stack

- **Backend:** FastAPI, Uvicorn  
- **Vector DB:** FAISS  
- **Embeddings:** SentenceTransformers  
- **LLM:** Gemini 1.5 Flash API  
- **Text Processing:** LangChain  
- **Parsing:** PyMuPDF, Mammoth, Python email parser  
- **Async Networking:** aiohttp, requests  
- **Language:** Python 3.10+  

---

# 📁 Project Structure
```
Server.py          → Main FastAPI backend  
Client.html        → Frontend UI  
requirements.txt   → Dependencies  
```
---

# ⚙️ Setup & Installation

## 1️⃣ Clone Repository
git clone https://github.com/your-username/DocuQuery.git  
cd DocuQuery  



## 2️⃣  Install Dependencies
pip install -r requirements.txt  

---

## 3️⃣ Configure Environment Variables

Create a `.env` file:

GEMINI_API_KEY=your_gemini_api_key  
BEARER_KEY=your_bearer_token  
PORT=8000  

---

## 4️⃣ Run Backend Server
python Server.py  

Server runs at:
http://localhost:8000  

---

## 5️⃣ Open Frontend
Open:

Client.html  

in your browser.


---

## 🚀 How to Use

### Step 1: Upload Document

Provide a publicly accessible URL of:
- PDF
- DOCX
- EML

### Step 2: Processing Happens Automatically

System:
- Extracts text
- Splits into chunks
- Generates embeddings
- Stores vectors in FAISS

### Step 3: Ask Questions

Examples:
- Summarize this document
- What are the key points?
- What does section 2 explain?
- Who are the people mentioned?

### Step 4: Get AI Response

- Relevant chunks retrieved via semantic search
- Gemini 1.5 Flash processes context
- Returns grounded answer

---

## 🔗 Use Cases

- Document Q&A systems
- Enterprise knowledge base search
- RAG-based AI pipelines
- AI document assistants
- Research and legal document analysis

---

## 📈 LinkedIn Post

https://www.linkedin.com/feed/update/urn:li:activity:7359036194195681280/


---

