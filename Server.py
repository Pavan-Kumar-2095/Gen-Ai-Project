from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, Header, APIRouter
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
import numpy as np
import math
import qdrant_client
from qdrant_client.http import models as qdrant_models

load_dotenv()

app = FastAPI()

# 🔐 API key for Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BEARER = os.getenv("BEARER_KEY")
port = int(os.getenv("PORT", 8000))

# Qdrant setup
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # If needed
QDRANT_COLLECTION = "documents"

qdrant = qdrant_client.QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=False
)

# Create collection if not exists
if QDRANT_COLLECTION not in [col.name for col in qdrant.get_collections().collections]:
    qdrant.recreate_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=qdrant_models.VectorParams(
            size=384,  # embedding dim of all-MiniLM-L6-v2
            distance=qdrant_models.Distance.COSINE
        )
    )

# Initialize model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

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

async def upload_to_qdrant(embeddings, chunks, batch_size=200):
    points = []
    for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
        point = qdrant_models.PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"text": chunk}
        )
        points.append(point)

        if len(points) >= batch_size:
            qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
            points.clear()

    # Upload remaining points
    if points:
        qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
    print(f"[✓] Uploaded {len(chunks)} chunks to Qdrant in batches of {batch_size}")

async def search_qdrant(vector, top_k=5):
    search_result = qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=vector,
        limit=top_k,
        with_payload=True
    )
    results = []
    for hit in search_result:
        chunk_text = hit.payload.get("text", "")
        results.append({"chunk": chunk_text, "score": hit.score})
    return results

# Helper functions (query_gemini, build_prompt, parse_answers, join_all_lines, choose_batch_size)

async def query_gemini(prompt: str) -> str:
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
        percentage = 0.80
    batch_size = math.ceil(num_questions * percentage)
    return min(batch_size, 1000)

# Root endpoint
@app.get("/")
def root():
    return {"message": "***** Qdrant-based FastAPI server running *****"}

# Create a router with prefix /api/v1
api_router = APIRouter(prefix="/api/v1")

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

        # Upload chunks + embeddings to Qdrant in batches of 200
        await upload_to_qdrant(embeddings, chunks, batch_size=200)

        # Process questions
        answers = []
        batch_size_questions = choose_batch_size(len(questions))

        for i in range(0, len(questions), batch_size_questions):
            batch_questions = questions[i:i + batch_size_questions]
            batch_embeddings = query_embeddings[i:i + batch_size_questions]
            contexts = []
            for embedding in batch_embeddings:
                top_chunks = await search_qdrant(embedding, top_k=3)
                context = join_all_lines("\n".join([c['chunk'] for c in top_chunks]))
                contexts.append(context)

            prompt = build_prompt(batch_questions, contexts)
            response = await query_gemini(prompt)
            batch_answers = parse_answers(response, len(batch_questions))
            answers.extend(batch_answers)

        end = time.perf_counter()

        print("embedding time", end_embedd - start_embedd)
        print(f"total query Processing time: {end - end_embedd:.2f} seconds")
        print(f"total Processing time: {end - start:.2f} seconds")
        print(GEMINI_API_KEY[-2:])
        return {"answers": answers}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run("test:app", host="0.0.0.0", port=port)


