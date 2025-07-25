from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
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



load_dotenv()


app = FastAPI()

# 🔐 API keys and config
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize clients
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY , timeout=600)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


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

async def upload_to_qdrant(collection: str, embeddings, texts):
    new_hash = hash_texts(texts)
    vector_size = len(embeddings[0])

    # Check if collection exists
    if client.collection_exists(collection):
        try:
            # Scroll to fetch one stored point with payload
            existing_points, _ = client.scroll(collection_name=collection, limit=1, with_payload=True)
            if existing_points:
                stored_hash = existing_points[0].payload.get("hash")
                if stored_hash == new_hash:
                    print(f"[✓] Skipping upload: Collection '{collection}' already has identical content.")
                    return
                else:
                    print(f"[!] Content changed. Deleting collection '{collection}'...")
                    client.delete_collection(collection)
        except Exception as e:
            print(f"[!] Failed to check existing collection: {e}")
            client.delete_collection(collection)

    # Recreate and upload
    client.recreate_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

    points = [
        PointStruct(
            id=i,
            vector=vec,
            payload={"chunk": texts[i], "session_id": collection, "hash": new_hash}
        )
        for i, vec in enumerate(embeddings)
    ]

    client.upsert(collection_name=collection, points=points)
    print(f"[↑] Uploaded {len(points)} points to Qdrant collection '{collection}'.")


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


@app.post("/process")
async def process(
    file: Optional[UploadFile] = File(None),
    query: Optional[str] = Form(None),
    session_id: str = Form(default="doc_session")
):
    try:
        if file:
            # Extract text, chunk, embed, upload
            text = await extract_text(file)
            chunks = chunk_text(text)
            embeddings = await get_embeddings(chunks)
            await upload_to_qdrant(session_id, embeddings, chunks)

            if query:
                # If query is provided together, search & answer
                query_embedding = await get_embeddings([query])
                top_chunks = await search_qdrant(session_id, query_embedding[0], top_k=5)

                context = "\n\n".join([f"Chunk {i+1}: {c['chunk']}" for i, c in enumerate(top_chunks)])
                prompt = f"Answer the following based on the document context:\n\nContext:\n{context}\n\nQuestion: {query}"

                gemini_answer = await query_gemini(prompt)
                return {
                    "message": "Document processed and query answered",
                    "chunks": len(chunks),
                    "session_id": session_id,
                    "answer": gemini_answer,
                    "chunks_used": top_chunks
                }

            # Otherwise just return document upload info
            return {"message": "Document processed", "chunks": len(chunks), "session_id": session_id}

        elif query:
            # If only query is provided, search existing collection
            query_embedding = await get_embeddings([query])
            top_chunks = await search_qdrant(session_id, query_embedding[0], top_k=5)

            context = "\n\n".join([f"Chunk {i+1}: {c['chunk']}" for i, c in enumerate(top_chunks)])
            prompt = f"Answer the following based on the document context:\n\nContext:\n{context}\n\nQuestion: {query}"

            gemini_answer = await query_gemini(prompt)
            return {"answer": gemini_answer, "chunks_used": top_chunks}

        else:
            raise HTTPException(status_code=400, detail="Either file or query is required.")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/")
def root():
    return {"message": "***** FastAPI server running *****"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
