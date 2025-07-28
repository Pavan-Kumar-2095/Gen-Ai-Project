from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body , Header
from fastapi.responses import JSONResponse
from typing import Optional, List
import os
import tempfile
import time
# from mailparser import parse_from_bytes
import uvicorn
import requests
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import hashlib
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
import aiohttp
from pydantic import BaseModel
import uuid
import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re



load_dotenv()

app = FastAPI()

# 🔐 API keys and config
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BEARER = os.getenv("BEARER_KEY")


# Initialize clients
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=600)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


class DocumentRequest(BaseModel):
    documents: Optional[str] = None
    questions: Optional[List[str]] = None

# chunk_size=1500, chunk_overlap=400
def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','],chunk_size=1500, chunk_overlap=400)    
    return splitter.split_text(text)



async def get_embeddings(text_list):
    return embedding_model.encode(text_list, convert_to_numpy=True).tolist()


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

collection_name = "documents"
batch_size = 500
async def upload_to_qdrant(embeddings, chunks):
    # new_hash = hash_texts(chunks)  # This identifies the document

    

    # Create the shared collection only once
    if not client.collection_exists(collection_name):
        print(f"[+] Creating collection '{collection_name}'")
        vector_size = len(embeddings[0])
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

    # Create points with shared collection and per-document payload
    points = [
        PointStruct(
            id=str(uuid.uuid4()),   # Ensure unique IDs across all docs
            vector=embedding,
            payload={"chunk": chunks[i]}
        )
        for i, embedding in enumerate(embeddings)
    ]

    print(f"[↑] Uploading {len(points)} points for documents")
    start_upload = time.perf_counter()
    for batch in chunked(points, batch_size):
        client.upsert(collection_name=collection_name, points=batch)
    end_upload = time.perf_counter()
    print(f"[✓] Done uploading to '{collection_name}'  ,  {end_upload - start_upload}")

    # return new_hash





async def search_qdrant(collection: str, vector, top_k=5):
    search_result = client.search(
        collection_name=collection,
        query_vector=vector,
        limit=top_k
    )
    return [{"chunk": r.payload["chunk"], "score": r.score} for r in search_result]




async def query_gemini(prompt):
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    response = requests.post(endpoint, json=payload)
    if response.ok:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"Gemini API failed: {response.status_code} - {response.text}"
    



app = FastAPI()
# /api/v1
@app.post("/hackrx/run")
async def process_doc(
    body: DocumentRequest = Body(...),
    authorization: Optional[str] = Header(None),
):
    start= time.perf_counter()
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

        # Save to temp file
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
        else:
            raise HTTPException(status_code=400, detail="Only .pdf or .docx files are supported.")

        if not text or text.strip() == "":
            raise HTTPException(status_code=400, detail="Document contains no extractable text.")

        # Chunk & embed
        chunks = chunk_text(text)
        questions = body.questions or []

        embeddings, query_embeddings = await asyncio.gather(
            get_embeddings(chunks),
            get_embeddings(questions)
        )

        await upload_to_qdrant(embeddings, chunks)

        # Process in pairs
        answers = []
        timings = []

        for i in range(0, len(questions), 2):
            print(f"\nProcessing Question 1: {questions[i]}")
            print(f"Processing Question 2: {questions[i + 1]}")

            # Qdrant search
            start_qdrant_1 = time.perf_counter()
            context1 = await search_qdrant(collection_name, query_embeddings[i], top_k=6)
            end_qdrant_1 = time.perf_counter()

            start_qdrant_2 = time.perf_counter()
            context2 = await search_qdrant(collection_name, query_embeddings[i + 1], top_k=6)
            end_qdrant_2 = time.perf_counter()

            # Prompt
            prompt = f"""Answer the following questions using only their respective Document. Keep each answer to one concise sentence.Respond in **this exact format**:
                        Answer 1: <your answer to Q1>
                        Answer 2: <your answer to Q2>
                        Document  1:
                        {context1}

                        Q1: {questions[i]}

                        ---

                        Document 2:
                        {context2}

                        Q2: {questions[i + 1]}
                        """.strip()


            # Query Gemini
            print("Querying Gemini...")
            start_gemini = time.perf_counter()
            response = await query_gemini(prompt)
            print(response)
            end_gemini = time.perf_counter()
            print(f"✅ Gemini responded in {end_gemini - start_gemini:.4f}s")

            pattern = r"Answer 1:\s*(.*?)\s*Answer 2:\s*(.*)"
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)

            if match:
                ans1 = match.group(1).strip() or "The document does not provide enough information to answer this."
                ans2 = match.group(2).strip() or "The document does not provide enough information to answer this."
            else:
                ans1 = "The document does not provide enough information to answer this."
                ans2 = "The document does not provide enough information to answer this."


            answers.append(ans1)
            answers.append(ans2)

            timings.append({
                "questions": [questions[i], questions[i + 1]],
                "qdrant_time_q1": end_qdrant_1 - start_qdrant_1,
                "qdrant_time_q2": end_qdrant_2 - start_qdrant_2,
                "gemini_time": end_gemini - start_gemini
            })

        # Cleanup
        # collection_name.delete()
        end= time.perf_counter()
        print(end - start)

        return {
            "answers": answers,
            "timings": timings
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})



@app.get("/")
def root():
    return {"message": "***** FastAPI server running *****"}

if __name__ == "__main__":
    uvicorn.run("Server:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

##################################### FOR File uploads directly #####################################

# async def extract_text(file: UploadFile) -> str:
#     ext = file.filename.split('.')[-1].lower()

#     if ext == "pdf":
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#             tmp.write(await file.read())
#             tmp.close()
#             doc = fitz.open(tmp.name)
#             text = "\n".join([page.get_text() for page in doc])
#         return text

#     elif ext == "docx":
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
#             tmp.write(await file.read())
#             tmp.close()
#             result = mammoth.extract_raw_text({'path': tmp.name})
#             return result.value

#     elif ext == "eml":
#         parsed = parse_from_bytes(await file.read())
#         return parsed.body or parsed.text_plain[0] if parsed.text_plain else ""

#     else:
#         raise HTTPException(status_code=400, detail="Unsupported file format.")




# async def inter_query_gemini(question, context):
#     prompt = (
#                 "Given the following context and question, extract only the parts of the context "
#                 "that are directly relevant to answering the question.\n\n"
#                 f"Question: {question}\n\n"
#                 f"Context:\n{context}\n\n"
#                 "Relevant text:"
#     )

#     payload = {"contents": [{"parts": [{"text": prompt}]}]}
#     url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    
#     res = requests.post(url, json=payload)
#     if res.ok:
#         return res.json()['candidates'][0]['content']['parts'][0]['text']
#     return f"Gemini API failed: {res.status_code} - {res.text}"

#####################################################################################################
# curl -X POST http://localhost:8000/hackrx/run \
#   -H "Content-Type: application/json" \
#   -H "Accept: application/json" \
#   -H "Authorization: Bearer fc3fb7ea7e00291256bfa945f2d45d1fed6ceb80a5d8763563d12816dcbc9e8d" \
#   -d '{
#     "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
#     "questions": [
#       "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
#       "What is the waiting period for pre-existing diseases (PED) to be covered?",
#       "Does this policy cover maternity expenses, and what are the conditions?",
#       "What is the waiting period for cataract surgery?",
#       "Are the medical expenses for an organ donor covered under this policy?",
#       "What is the No Claim Discount (NCD) offered in this policy?",
#       "Is there a benefit for preventive health check-ups?",
#       "How does the policy define a 'Hospital'?",
#       "What is the extent of coverage for AYUSH treatments?",
#       "Are there any sub-limits on room rent and ICU charges for Plan A?"
#     ]
#   }' \
#   -w "\nTotal time: %{time_total} seconds\n"