import httpx
from collections import OrderedDict
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue


class Retriever:
    def __init__(self, config, parts):
        c = config['components']['retriever']
        llm = config['components']['llm']
        self.qdrant = AsyncQdrantClient(url=c['url'])
        self.collection, self.top_k, self.min_score = c['collection'], c['top_k'], c['min_score']
        self.embed_url = llm['base_url'] + '/embeddings'
        self.embed_model = llm['embedding_model']
        self.infer_lock = parts['llm'].infer_lock
        self.http = httpx.AsyncClient(timeout=300)
        self._collection_exists = None
        self._embedding_cache = OrderedDict()
        self._embedding_cache_limit = 256

    async def search(self, question, level):
        if self._collection_exists is None:
            self._collection_exists = await self.qdrant.collection_exists(self.collection)
        if not self._collection_exists:
            return []
        cache_key = ' '.join(question.lower().split())
        vector = self._embedding_cache.get(cache_key)
        if vector is None:
            async with self.infer_lock:
                r = await self.http.post(self.embed_url, json={'model': self.embed_model, 'input': [question]})
                r.raise_for_status()
                vector = r.json()['data'][0]['embedding']
            self._embedding_cache[cache_key] = vector
            self._embedding_cache.move_to_end(cache_key)
            while len(self._embedding_cache) > self._embedding_cache_limit:
                self._embedding_cache.popitem(last=False)
        else:
            self._embedding_cache.move_to_end(cache_key)
        only_level = Filter(must=[FieldCondition(key='level', match=MatchValue(value=level))])
        result = await self.qdrant.query_points(self.collection, query=vector, query_filter=only_level, limit=self.top_k)
        return [
            {'source': h.payload['source'] + ' p.' + str(h.payload['page']), 'text': h.payload['text'], 'score': h.score}
            for h in result.points
            if h.score >= self.min_score
        ]


def build(config, parts):
    return Retriever(config, parts)
