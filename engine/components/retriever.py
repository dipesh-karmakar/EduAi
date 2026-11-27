import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue


class Retriever:
    def __init__(self, config):
        c = config['components']['retriever']
        llm = config['components']['llm']
        self.qdrant = AsyncQdrantClient(url=c['url'])
        self.collection, self.top_k, self.min_score = c['collection'], c['top_k'], c['min_score']
        self.embed_url = llm['base_url'] + '/embeddings'
        self.embed_model = llm['embedding_model']
        self.http = httpx.AsyncClient(timeout=300)

    async def search(self, question, level):
        if not await self.qdrant.collection_exists(self.collection):
            return []
        r = await self.http.post(self.embed_url, json={'model': self.embed_model, 'input': [question]})
        r.raise_for_status()
        vector = r.json()['data'][0]['embedding']
        only_level = Filter(must=[FieldCondition(key='level', match=MatchValue(value=level))])
        result = await self.qdrant.query_points(self.collection, query=vector, query_filter=only_level, limit=self.top_k)
        return [
            {'source': h.payload['source'] + ' p.' + str(h.payload['page']), 'text': h.payload['text'], 'score': h.score}
            for h in result.points
            if h.score >= self.min_score
        ]


def build(config, parts):
    return Retriever(config)
