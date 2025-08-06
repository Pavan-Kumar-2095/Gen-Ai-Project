from fastapi import FastAPI, HTTPException, Body, Header, APIRouter
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import os
import tempfile
import time
import requests
import aiohttp
import math
import threading
import faiss
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import uvicorn

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BEARER = os.getenv("BEARER_KEY")
PORT = int(os.getenv("PORT", 8000))

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Global state
faiss_index = None
faiss_chunks = []
faiss_document_url = None
faiss_lock = threading.Lock()

class DocumentRequest(BaseModel):
    documents: Optional[str] = None
    questions: Optional[List[str]] = None

def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1500,
        chunk_overlap=400
    )
    return splitter.split_text(text)

def join_all_lines(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def build_prompt(questions: list[str], contexts: list[str]) -> str:
    answers_header = "\n".join([f"Answer {i + 1}: <your answer to Q{i + 1}>" for i in range(len(questions))])
    documents_questions = ""
    for i, (context, question) in enumerate(zip(contexts, questions)):
        documents_questions += f"\nDocument {i + 1}:\n{context}\n\nQ{i + 1}: {question}\n\n---"
    prompt = (
        f"Answer the following questions using only their respective Document. "
        f"Keep each answer to one concise sentence. Respond in **this exact format**:\n\n"
        f"{answers_header}\n\n{documents_questions.strip()}"
    )
    return prompt

def parse_answers(response: str, num_questions: int) -> list[str]:
    pattern = r"Answer\s+(\d+):\s*(.*?)(?=\s*Answer\s+\d+:|$)"
    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
    answers = ["The document "] * num_questions
    for match in matches:
        idx, ans = int(match[0]) - 1, match[1].strip()
        if 0 <= idx < num_questions and ans:
            answers[idx] = ans
    return answers

def choose_batch_size(num_questions: int) -> int:
    if num_questions <= 40:
        return 50
    elif num_questions <= 60:
        percentage = 0.60
    elif num_questions <= 80:
        percentage = 0.65
    elif num_questions <= 100:
        percentage = 0.70
    elif num_questions <= 150:
        percentage = 0.75
    else:
        percentage = 0.80
    batch_size = math.ceil(num_questions * percentage)
    return min(batch_size, 1000)

async def query_gemini(prompt):
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(endpoint, json=payload)
    if response.ok:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"Gemini API failed: {response.status_code} - {response.text}"

# Define router BEFORE use
api_router = APIRouter(prefix="/api/v1")

@api_router.post("/hackrx/run")
async def process_doc(
    body: DocumentRequest = Body(...),
    authorization: Optional[str] = Header(None),
):
    global faiss_index, faiss_chunks, faiss_document_url

    start = time.perf_counter()

    if not authorization or authorization != BEARER:
        raise HTTPException(status_code=401, detail="Unauthorized")

    url = body.documents
    questions = body.questions or []

    if not url:
        raise HTTPException(status_code=400, detail="Document URL is required.")

    try:
        with faiss_lock:
            if faiss_document_url != url:
                faiss_index = None
                faiss_chunks = []
                faiss_document_url = None

        if faiss_index is None:
            filename = url.split("?")[0].split("/")[-1]
            ext = filename.split(".")[-1].lower()

            # Download file
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        raise HTTPException(status_code=400, detail="Failed to fetch document from URL.")
                    contents = await resp.read()

            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                tmp.write(contents)
                tmp_path = tmp.name

            # Extract text
            if ext == "pdf":
                import fitz
                doc = fitz.open(tmp_path)
                text = "\n".join([page.get_text() for page in doc])
            elif ext == "docx":
                import mammoth
                result = mammoth.extract_raw_text({'path': tmp_path})
                text = result.value
            elif ext == "eml":
                import email
                from email import policy
                from email.parser import BytesParser
                with open(tmp_path, 'rb') as fp:
                    msg = BytesParser(policy=policy.default).parse(fp)
                    parts = [f"Subject: {msg['subject']}", f"From: {msg['from']}", f"To: {msg['to']}"]
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == 'text/plain':
                                parts.append(part.get_content())
                                break
                    else:
                        parts.append(msg.get_content())
                    text = "\n".join(parts)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")

            if not text.strip():
                raise HTTPException(status_code=400, detail="Empty or unextractable document")

            chunks = chunk_text(text)
            embeddings = embedding_model.encode(chunks, convert_to_numpy=True).astype('float32')
            faiss.normalize_L2(embeddings)

            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)

            with faiss_lock:
                faiss_index = index
                faiss_chunks = chunks
                faiss_document_url = url

        query_embeddings = embedding_model.encode(questions, convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_embeddings)

        answers = []
        batch_size = choose_batch_size(len(questions))

        with faiss_lock:
            index = faiss_index
            chunks = faiss_chunks

        for i in range(0, len(questions), batch_size):
            batch_qs = questions[i:i + batch_size]
            batch_embeds = query_embeddings[i:i + batch_size]
            contexts = []

            for embed in batch_embeds:
                D, I = index.search(np.array([embed]), 3)
                top_chunks = [chunks[idx] for idx in I[0] if idx < len(chunks)]
                context = join_all_lines("\n".join(top_chunks))
                contexts.append(context)

            prompt = build_prompt(batch_qs, contexts)
            response = await query_gemini(prompt)
            batch_answers = parse_answers(response, len(batch_qs))
            answers.extend(batch_answers)

        return {
            "answers": answers,
            "last_response_time": time.perf_counter() - start
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ✅ Mount the router after defining it and its routes
app.include_router(api_router)

@app.get("/")
def root():
    return {"message": "FAISS server running"}

if __name__ == "__main__":
    uvicorn.run("Server:app", host="0.0.0.0", port=PORT)
