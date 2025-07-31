from fastapi import FastAPI, HTTPException, Body, Header, APIRouter
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing import Optional, List
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import aiohttp
import time
import uuid
import re
import math

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from qdrant_client.models import Document

# -------------------- Load Environment --------------------
load_dotenv()

PDFCO_API_KEY = os.getenv("PDFCO_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BEARER = os.getenv("BEARER_KEY")
PORT = int(os.getenv("PORT", 8000))
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = "documents"

print("[DEBUG] QDRANT_URL:", QDRANT_URL)

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

app = FastAPI()
api_router = APIRouter(prefix="/api/v1")

# -------------------- Models --------------------
class DocumentRequest(BaseModel):
    documents: Optional[str] = None
    questions: Optional[List[str]] = None

# -------------------- Utils --------------------
def chunk_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 400) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - chunk_overlap
    print(f"[DEBUG] Total chunks created: {len(chunks)}")
    return chunks

def upload_chunks_with_qdrant(collection_name: str, texts: list[str]):
    model_name = "BAAI/bge-small-en"
    vector_size = 384  # Set manually to avoid get_embedding_size

    if qdrant.collection_exists(collection_name):
        qdrant.delete_collection(collection_name)

    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

    documents = []
    payloads = []
    ids = []

    for chunk in texts:
        if not chunk or not isinstance(chunk, str):
            continue  # Skip invalid chunks
        documents.append(Document(text=chunk, model=model_name))
        payloads.append({"text": chunk})
        ids.append(str(uuid.uuid4()))

    print(f"[DEBUG] Uploading {len(documents)} documents to Qdrant")

    try:
        qdrant.upload_collection(
            collection_name=collection_name,
            vectors=documents,
            payload=payloads,
            ids=ids
        )
        print("[DEBUG] Upload completed.")
    except Exception as e:
        print("[ERROR] Failed to upload to Qdrant:", e)
        raise HTTPException(status_code=500, detail=f"Qdrant upload error: {str(e)}")


def search_qdrant(collection_name: str, query: str, top_k: int = 7) -> list[str]:
    model_name = "BAAI/bge-small-en"
    result = qdrant.query_points(
        collection_name=collection_name,
        query=Document(text=query, model=model_name),
        limit=top_k
    )
    return [point.payload.get("text", "") for point in result.points]

async def query_gemini(prompt: str) -> str:
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return data['candidates'][0]['content']['parts'][0]['text']
            else:
                raise HTTPException(status_code=500, detail="Gemini API failed")

def build_prompt(questions: List[str], contexts: List[str]) -> str:
    answers_header = "\n".join([f"Answer {i + 1}: <your answer to Q{i + 1}>" for i in range(len(questions))])
    documents_questions = ""
    for i, (context, question) in enumerate(zip(contexts, questions)):
        documents_questions += f"\nDocument {i + 1}:\n{context}\n\nQ{i + 1}: {question}\n\n---"
    prompt = (
        f"Answer the following questions using only their respective Document.\n\n"
        f"Keep each answer to one concise sentence.\n"
        f"Respond **only** in the exact format below, without adding anything else:\n\n"
        f"{answers_header}\n\n"
        f"{documents_questions.strip()}"
    )
    return prompt

def parse_answers(response: str, num_questions: int) -> List[str]:
    pattern = r"Answer\s*(\d+)\s*:\s*(.*?)(?=\s*Answer\s*\d+\s*:|$)"
    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
    answers = [""] * num_questions
    for match in matches:
        idx, ans = int(match[0]) - 1, match[1].strip()
        if 0 <= idx < num_questions:
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

# -------------------- PDF.co Integration --------------------
async def convert_to_pdfco_pdf(file_url: str) -> str:
    endpoint = "https://api.pdf.co/v1/file/convert/to/pdf"
    headers = {"x-api-key": PDFCO_API_KEY, "Content-Type": "application/json"}
    payload = {"url": file_url, "inline": True}
    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint, headers=headers, json=payload) as resp:
            result = await resp.json()
            if result.get("error", False):
                raise HTTPException(status_code=400, detail=f"PDF.co error: {result.get('message')}")
            return result.get("url")

async def extract_text_from_pdfco(file_url: str) -> str:
    print(f"[DEBUG] extract_text_from_pdfco called with url: {file_url}")
    ext = file_url.split("?")[0].split(".")[-1].lower()
    if ext in {"docx", "eml"}:
        file_url = await convert_to_pdfco_pdf(file_url)

    pdfco_endpoint = "https://api.pdf.co/v1/pdf/convert/to/text"
    headers = {"x-api-key": PDFCO_API_KEY, "Content-Type": "application/json"}
    payload = {"url": file_url, "inline": True}

    print(f"[DEBUG] Sending payload to PDF.co: {payload}")
    async with aiohttp.ClientSession() as session:
        async with session.post(pdfco_endpoint, headers=headers, json=payload) as resp:
            result = await resp.json()
            if result.get("error"):
                raise HTTPException(status_code=400, detail=f"PDF.co Error: {result.get('message')}")
            text = result.get("body", "").strip()
            if not text:
                raise HTTPException(status_code=400, detail="PDF.co returned no text.")
            return text

# -------------------- Route --------------------
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
        questions = body.questions or []

        print("[DEBUG] Starting text extraction")
        text = await extract_text_from_pdfco(url)

        if not text.strip():
            raise HTTPException(status_code=400, detail="Empty text extracted.")

        print("[DEBUG] Starting chunking")
        chunks = chunk_text(text)

        print("[DEBUG] Uploading chunks to Qdrant")
        upload_chunks_with_qdrant(QDRANT_COLLECTION, chunks)

        answers = []
        batch_size_questions = choose_batch_size(len(questions))
        for i in range(0, len(questions), batch_size_questions):
            batch_questions = questions[i:i + batch_size_questions]
            contexts = []
            for question in batch_questions:
                top_chunks = search_qdrant(QDRANT_COLLECTION, question, top_k=7)
                context = join_all_lines("\n".join(top_chunks))
                contexts.append(context)

            prompt = build_prompt(batch_questions, contexts)
            response = await query_gemini(prompt)
            answers_batch = parse_answers(response, len(batch_questions))
            answers.extend(answers_batch)

        print(f"[DEBUG] Total time: {time.perf_counter() - start:.2f} seconds")
        return JSONResponse(content={"answers": answers})
    except Exception as e:
        print("[ERROR]", str(e))
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    
@app.get("/")
async def health_check():
    return {"status": "ok"}


# -------------------- Register Router --------------------
app.include_router(api_router)

# -------------------- Run --------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
