from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Optional, List
import os
import tempfile
import fitz  # PyMuPDF for PDF
import mammoth
from mailparser import parse_from_bytes
import uvicorn
import requests
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import hashlib
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
import aiohttp

from pydantic import BaseModel

load_dotenv()

app = FastAPI()

# 🔐 API keys and config
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize clients
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=600)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

class DocumentRequest(BaseModel):
    documents: Optional[str] = None
    questions: Optional[List[str]] = None

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    return [text[i:i + size] for i in range(0, len(text), size - overlap)]

async def extract_text(file: UploadFile) -> str:
    ext = file.filename.split('.')[-1].lower()

    if ext == "pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp.close()
            doc = fitz.open(tmp.name)
            text = "\n".join([page.get_text() for page in doc])
        return text

    elif ext == "docx":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(await file.read())
            tmp.close()
            result = mammoth.extract_raw_text({'path': tmp.name})
            return result.value

    elif ext == "eml":
        parsed = parse_from_bytes(await file.read())
        return parsed.body or parsed.text_plain[0] if parsed.text_plain else ""

    else:
        raise HTTPException(status_code=400, detail="Unsupported file format.")

async def get_embeddings(text_list):
    return embedding_model.encode(text_list, convert_to_numpy=True).tolist()

def hash_texts(texts):
    """Generate a hash from the list of texts for content comparison."""
    return hashlib.md5("".join(texts).encode("utf-8")).hexdigest()

async def upload_to_qdrant(embeddings, chunks):
    new_hash = hash_texts(chunks)
    
    if client.collection_exists(new_hash):
        print(f"[✓] Collection '{new_hash}' already exists. Skipping upload.")
        return new_hash

    vector_size = len(embeddings[0])
    
    # Create new collection
    client.recreate_collection(
        collection_name=new_hash,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

    points = [
        PointStruct(
            id=i,
            vector=embedding,
            payload={"chunk": chunks[i], "hash": new_hash}
        )
        for i, embedding in enumerate(embeddings)
    ]

    client.upsert(collection_name=new_hash, points=points)
    print(f"[↑] Uploaded {len(points)} points to Qdrant collection '{new_hash}'.")
    return new_hash


async def search_qdrant(collection: str, vector, top_k=5):
    search_result = client.search(
        collection_name=collection,
        query_vector=vector,
        limit=top_k
    )
    return [{"chunk": r.payload["chunk"], "score": r.score} for r in search_result]

async def query_gemini(prompt):
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    response = requests.post(endpoint, json=payload)
    if response.ok:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"Gemini API failed: {response.status_code} - {response.text}"

@app.post("/hackrx/run")
async def process_doc(
    file: Optional[UploadFile] = File(None),
    query: Optional[str] = Form(None),
    body: Optional[DocumentRequest] = Body(None)
):
    try:
        text = None

        # === 1. Uploaded File ===
        if file:
            text = await extract_text(file)

        # === 2. Document from URL ===
        elif body and body.documents:
            url = body.documents
            filename = url.split("?")[0].split("/")[-1]
            ext = filename.split(".")[-1].lower()

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        raise HTTPException(status_code=400, detail="Failed to fetch document from URL.")
                    contents = await resp.read()

            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                tmp.write(contents)
                tmp_path = tmp.name

            if ext == "pdf":
                doc = fitz.open(tmp_path)
                text = "\n".join([page.get_text() for page in doc])
            elif ext == "docx":
                result = mammoth.extract_raw_text({'path': tmp_path})
                text = result.value
            else:
                raise HTTPException(status_code=400, detail="Only .pdf or .docx files are supported via 'documents'.")

        # === 3. No input at all ===
        if not text:
            raise HTTPException(status_code=400, detail="No file or documents URL provided.")

        # === CHUNK + EMBED + UPLOAD ===
        chunks = chunk_text(text)
        embeddings = await get_embeddings(chunks)

        new_hash = await upload_to_qdrant(embeddings, chunks)

        # === Answer questions ===
        questions = [query] if query else []
        if body and body.questions:
            questions = body.questions

        answers = []
        if questions:
            for q in questions:
                query_embedding = await get_embeddings([q])
                top_chunks = await search_qdrant(new_hash, query_embedding[0], top_k=5)
                context = "\n\n".join([f"Chunk {i+1}: {c['chunk']}" for i, c in enumerate(top_chunks)])
                prompt = f"Answer the following based on the document context:\n\nContext:\n{context}\n\nQuestion: {q}"
                answer = await query_gemini(prompt)
                answers.append(answer)

        return {"answers": answers}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def root():
    return {"message": "***** FastAPI server running *****"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
