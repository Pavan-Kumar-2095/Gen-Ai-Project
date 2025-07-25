from fastapi import FASTAPI , Request
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

app = FASTAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_method=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer("all-MiniLM-L6-v2")

@app.post("/embed")
async def embed_text(request: Request):
    data = await request.json()
    text = data.get("text" , "").strip()
    if not text:
        return{"error" : "No text provided"}
    
    embedding = model.encode(text).tolist()
    return{"embedding" : embedding}