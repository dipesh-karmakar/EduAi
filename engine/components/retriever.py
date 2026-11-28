# Description: Retrieves level- and subject-relevant textbook passages from Qdrant.
# Author: Dipesh Karmakar <dipesh.karmakar@gmail.com>
# Modifications:
# - 433859a: Added subject filtering with a grade-only fallback.
# - 7de329d: Serialized embedding requests with chat inference.
# - 374a71c: Added an embedding cache.
# - f9dc837: Added the initial textbook retriever.
# - Uncommitted: Added subject filtering with a grade-only fallback.

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

    async def search(self, question, level, subject=None):
        if self._collection_exists is None:
            self._collection_exists = await self.qdrant.collection_exists(self.collection)
        if not self._collection_exists:
            return []
        cache_key = ' '.join(question.lower().split())
        vector = self._embedding_cache.get(cache_key)
        if vector is None:
            # Embedding and chat requests share a model server, so serialize alias changes.
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

        level_condition = FieldCondition(key='level', match=MatchValue(value=level))
        subject = subject.strip().casefold() if isinstance(subject, str) else ''
        if subject:
            # Prefer subject-specific evidence; use the wider level pool only if needed.
            subject_filter = Filter(must=[
                level_condition,
                FieldCondition(key='subject', match=MatchValue(value=subject)),
            ])
            result = await self.qdrant.query_points(
                self.collection, query=vector, query_filter=subject_filter, limit=self.top_k
            )
            evidence = self._format_points(result.points)
            if evidence:
                return evidence

        level_filter = Filter(must=[level_condition])
        result = await self.qdrant.query_points(
            self.collection, query=vector, query_filter=level_filter, limit=self.top_k
        )
        return self._format_points(result.points)

    def _format_points(self, points):
        return [
            {'source': h.payload['source'] + ' p.' + str(h.payload['page']), 'text': h.payload['text'], 'score': h.score}
            for h in points
            if h.score >= self.min_score
        ]


def build(config, parts):
    return Retriever(config, parts)
