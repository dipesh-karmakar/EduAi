import sys
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine'))

from components.retriever import Retriever


def point(score=0.9):
    return SimpleNamespace(
        payload={'source': 'physics.pdf', 'page': 3, 'text': 'Gravity evidence'},
        score=score,
    )


class RetrieverSearchTests(unittest.IsolatedAsyncioTestCase):
    def make_retriever(self, responses):
        retriever = Retriever.__new__(Retriever)
        retriever._collection_exists = True
        retriever._embedding_cache = OrderedDict([('what is gravity?', [0.1, 0.2])])
        retriever._embedding_cache_limit = 256
        retriever.collection = 'textbooks'
        retriever.top_k = 5
        retriever.min_score = 0.45
        retriever.qdrant = SimpleNamespace(
            query_points=AsyncMock(
                side_effect=[SimpleNamespace(points=points) for points in responses]
            )
        )
        return retriever

    async def test_filters_by_level_and_normalized_subject(self):
        retriever = self.make_retriever([[point()]])

        results = await retriever.search('What is gravity?', 'class-9-12', ' Physics ')

        self.assertEqual(len(results), 1)
        query_filter = retriever.qdrant.query_points.call_args.kwargs['query_filter']
        conditions = {condition.key: condition.match.value for condition in query_filter.must}
        self.assertEqual(conditions, {'level': 'class-9-12', 'subject': 'physics'})

    async def test_falls_back_to_level_filter_when_subject_has_no_good_matches(self):
        retriever = self.make_retriever([[point(score=0.2)], [point()]])

        results = await retriever.search('What is gravity?', 'class-9-12', 'physics')

        self.assertEqual(len(results), 1)
        self.assertEqual(retriever.qdrant.query_points.await_count, 2)
        subject_filter, fallback_filter = [
            call.kwargs['query_filter']
            for call in retriever.qdrant.query_points.await_args_list
        ]
        self.assertEqual(
            {condition.key for condition in subject_filter.must},
            {'level', 'subject'},
        )
        self.assertEqual(
            [condition.key for condition in fallback_filter.must],
            ['level'],
        )


if __name__ == '__main__':
    unittest.main()
