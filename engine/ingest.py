# ingest.py: python ingest.py book.pdf class-9-12 physics
import os
import sys
import uuid
import httpx
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

path, level, subject = sys.argv[1:4]
EMBED_URL = os.getenv('EMBED_URL', 'http://infer:8000/v1/embeddings')
QDRANT_URL = os.getenv('QDRANT_URL', 'http://qdrant:6333')

qdrant = QdrantClient(url=QDRANT_URL)
if not qdrant.collection_exists('textbooks'):
    qdrant.create_collection('textbooks', vectors_config=VectorParams(size=1024, distance=Distance.COSINE))


def embed(texts):
    r = httpx.post(EMBED_URL, json={'model': 'embedding', 'input': texts}, timeout=600)
    r.raise_for_status()
    return [d['embedding'] for d in r.json()['data']]


def chunks(text, size=900, overlap=150):
    for start in range(0, len(text), size - overlap):
        yield text[start:start + size]


pieces = []
for page_no, page in enumerate(PdfReader(path).pages, start=1):
    for piece in chunks(page.extract_text() or ''):
        if len(piece.strip()) >= 80:
            pieces.append((piece, page_no))

for i in range(0, len(pieces), 32):
    batch = pieces[i:i + 32]
    vectors = embed([text for text, _ in batch])
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={
                'text': text,
                'page': page_no,
                'level': level,
                'subject': subject.strip().casefold(),
                'source': path.split('/')[-1],
            },
        )
        for vec, (text, page_no) in zip(vectors, batch)
    ]
    qdrant.upsert('textbooks', points)

print('stored', len(pieces), 'chunks')

