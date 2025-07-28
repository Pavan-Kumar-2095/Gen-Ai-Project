from fastapi import FastAPI, HTTPException, Body, Header, APIRouter
from fastapi.responses import JSONResponse
from typing import Optional, List
import os
import tempfile
import time
import uvicorn
import requests
import uuid
from dotenv import load_dotenv
import aiohttp
from pydantic import BaseModel
import re
import math
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance

# Load env variables
load_dotenv()

# Setup FastAPI
app = FastAPI()
api_router = APIRouter(prefix="/api/v1")

# Auth & API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BEARER = os.getenv("BEARER_KEY")
PORT = int(os.getenv("PORT", 8000))

# Qdrant setup
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # Optional for cloud
QDRANT_COLLECTION = "documents"

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=False
)

# --------------------- Models ---------------------
class DocumentRequest(BaseModel):
    documents: Optional[str] = None
    questions: Optional[List[str]] = None

# ------------------ Utilities ---------------------

def chunk_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 400) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def upload_chunks_with_qdrant(collection_name: str, texts: list[str]):
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
    points = [
        PointStruct(id=str(uuid.uuid4()), payload={"text": text}, vector=None)
        for text in texts
    ]
    qdrant.upload_points(
        collection_name=collection_name,
        points=points,
        batch_size=64,
        embedding_model="text2vec-openai"
    )
    print(f"[✓] Uploaded {len(texts)} chunks using Qdrant's built-in embeddings")

def search_qdrant(collection_name: str, query: str, top_k: int = 5):
    results = qdrant.search(
        collection_name=collection_name,
        query_text=query,
        top=top_k,
        embedding_model="text2vec-openai"
    )
    return [hit.payload["text"] for hit in results if "text" in hit.payload]

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
    return min(math.ceil(num_questions * percentage), 1000)

# ------------------- FastAPI Routes --------------------

@app.get("/")
def root():
    return {"message": "***** Qdrant-based FastAPI server running *****"}

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

        # Download document
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

        # Chunk and upload
        chunks = chunk_text(text)
        questions = body.questions or []

        upload_chunks_with_qdrant(QDRANT_COLLECTION, chunks)

        # Question answering
        answers = []
        batch_size_questions = choose_batch_size(len(questions))
        for i in range(0, len(questions), batch_size_questions):
            batch_questions = questions[i:i + batch_size_questions]
            contexts = []
            for question in batch_questions:
                top_chunks = search_qdrant(QDRANT_COLLECTION, question, top_k=3)
                context = join_all_lines("\n".join(top_chunks))
                contexts.append(context)

            prompt = build_prompt(batch_questions, contexts)
            response = await query_gemini(prompt)
            batch_answers = parse_answers(response, len(batch_questions))
            answers.extend(batch_answers)

        end = time.perf_counter()
        print(f"[✓] Total Processing time: {end - start:.2f} seconds")

        return {"answers": answers}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

app.include_router(api_router)

# ------------------ Uvicorn Entry ----------------------

if __name__ == "__main__":
    uvicorn.run("Server:app", host="0.0.0.0", port=PORT)
