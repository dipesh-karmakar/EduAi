# Description: Tests that the planner's subject is routed to textbook retrieval.
# Author: Dipesh Karmakar <dipesh.karmakar@gmail.com>
# Modifications:
# - 433859a: Added unit tests for the retriever and pipeline.
# - Uncommitted: Added subject-routing coverage and moved tests into ./test.

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine'))

from pipeline import run


class FakeLLM:
    async def plan(self, question):
        return {
            'subject': 'physics',
            'wants_picture': False,
            'needs_current_facts': False,
        }

    async def answer(self, question, style, evidence):
        return {'answer': 'Gravity attracts masses.', 'confident': True, 'needs_picture': False}


class FakeRetriever:
    def __init__(self):
        self.args = None

    async def search(self, *args):
        self.args = args
        return []


class PipelineSubjectRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_passes_planned_subject_to_retriever(self):
        retriever = FakeRetriever()
        parts = {'llm': FakeLLM(), 'retriever': retriever}
        job = {'text': 'Test subject routing for physics', 'images': [], 'level': 'class-9-12'}
        config = {'levels': {'class-9-12': 'Use suitable detail.'}}

        async for _ in run(job, parts, config):
            pass

        self.assertEqual(
            retriever.args,
            ('Test subject routing for physics', 'class-9-12', 'physics'),
        )


if __name__ == '__main__':
    unittest.main()
