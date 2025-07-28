from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, Header , APIRouter
from fastapi.responses import JSONResponse
from typing import Optional, List
import os
import tempfile
import time
import uvicorn
import requests
from sentence_transformers import SentenceTransformer
import uuid
from dotenv import load_dotenv
import aiohttp
from pydantic import BaseModel
import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import faiss
import numpy as np
import threading
import math

load_dotenv()

app = FastAPI()

# 🔐 API key for Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BEARER = os.getenv("BEARER_KEY")

# Initialize model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Globals for FAISS
faiss_index = None
faiss_chunks = []
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

async def get_embeddings(text_list):
    return embedding_model.encode(text_list, convert_to_numpy=True).tolist()

async def upload_to_faiss(embeddings, chunks):
    global faiss_index, faiss_chunks
    with faiss_lock:
        dim = len(embeddings[0])
        faiss_index = faiss.IndexFlatIP(dim)  # cosine sim with inner product
        embedding_array = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embedding_array)
        faiss_index.add(embedding_array)
        faiss_chunks = chunks.copy()
        print(f"[✓] FAISS index built with {len(faiss_chunks)} chunks")

async def search_faiss(vector, top_k=5):
    global faiss_index, faiss_chunks
    with faiss_lock:
        if faiss_index is None or len(faiss_chunks) == 0:
            return []
        query_vec = np.array([vector]).astype('float32')
        faiss.normalize_L2(query_vec)
        distances, indices = faiss_index.search(query_vec, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(faiss_chunks):
                results.append({"chunk": faiss_chunks[idx], "score": float(dist)})
        return results

async def query_gemini(prompt):
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(endpoint, json=payload)
    if response.ok:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"Gemini API failed: {response.status_code} - {response.text}"

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

def join_all_lines(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


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
        percentage = 0.80  # increased for >150

    batch_size = math.ceil(num_questions * percentage)
    return min(batch_size, 1000)  # optional max limit for safety

# Create a router with prefix /api/v1
api_router = APIRouter(prefix="/api/v1")


@app.get("/")
def root():
    return {"message": "***** FAISS-only FastAPI server running *****"}



@api_router.post("/hackrx/run")
async def process_doc(
    body: DocumentRequest = Body(...),
    authorization: Optional[str] = Header(None),
):
    start = time.perf_counter()

    if not authorization or authorization != BEARER:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        url = body.documents
        filename = url.split("?")[0].split("/")[-1]
        ext = filename.split(".")[-1].lower()

        # Download document content
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
            raise HTTPException(status_code=400, detail="Only .pdf, .docx or .eml files are supported.")

        if not text.strip():
            raise HTTPException(status_code=400, detail="Document contains no extractable text.")

        # Chunk & embed
        chunks = chunk_text(text)
        questions = body.questions or []

        start_embedd = time.perf_counter()

        embeddings, query_embeddings = await asyncio.gather(
            get_embeddings(chunks),
            get_embeddings(questions)
        )
        end_embedd = time.perf_counter()
        # Only upload to FAISS
        await upload_to_faiss(embeddings, chunks)

        # Process questions
        answers = []
        batch_size_questions = choose_batch_size(len(questions))
        


        for i in range(0, len(questions), batch_size_questions):
            batch_questions = questions[i:i + batch_size_questions]
            batch_embeddings = query_embeddings[i:i + batch_size_questions]
            contexts = []
            for embedding in batch_embeddings:
                top_chunks = await search_faiss(embedding, top_k=3)
                context = join_all_lines("\n".join([c['chunk'] for c in top_chunks]))
                contexts.append(context)

            prompt = build_prompt(batch_questions, contexts)
            response = await query_gemini(prompt)
            batch_answers = parse_answers(response, len(batch_questions))
            answers.extend(batch_answers)

        end = time.perf_counter()
        
        print("embedding time" , end_embedd - start_embedd)
        print(f"total query Processing time: {end - end_embedd:.2f} seconds")
        print(f"total Processing time: {end - start:.2f} seconds")
        return {"answers": answers}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
app.include_router(api_router)



if __name__ == "__main__":
    uvicorn.run("Server:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))




