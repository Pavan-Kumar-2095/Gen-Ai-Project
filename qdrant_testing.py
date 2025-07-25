from qdrant_client import QdrantClient
from qdrant_client.models import Distance , VectorParams
from sentence_transformers import SentenceTransformer

QDRANT_URL = "https://7df02a24-c7fb-4745-b1b8-c9502f67d43e.us-west-1-0.aws.cloud.qdrant.io:6333"

API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.JnZRLS0_qgAKgHp732Uy64yMf3Vwucsd0_aA7uskUd0"

client  = QdrantClient(
    url = QDRANT_URL,
    api_key = API_KEY
)

model = SentenceTransformer("all-MiniLM-L6-v2")

COLLECTION_NAME = "documents"

client.recreate_collection(
    collection_name = COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

#sample text
text = "knee surgery is covered under the 3-month policy in pune"
embedding = model.encode(text).tolist()

client.upsert(
    collection_name = COLLECTION_NAME,
    points=[{
        "id" : 1,
        "vector" : embedding,
        "payload": {"text": text}

    }]
)

query = "46-year-old male, knee surgery , pune , 3-month policy"
query_embedding = model.encode(query).tolist()

result = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_embedding,
    limit=3  
)


for res in result:
    print(f"score : {res.score:.3f} | match : {res.payload['text']}")

















































