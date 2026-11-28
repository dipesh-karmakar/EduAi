# Description: Tests that the planner's subject is routed to textbook retrieval.
# Author: Dipesh Karmakar <dipesh.karmakar@gmail.com>
# Modifications:
# - 433859a: Added unit tests for the retriever and pipeline.
# - Uncommitted: Added subject-routing and unwanted-image regression coverage.

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine'))

from pipeline import _wants_picture, run


# These fakes isolate subject routing from model and database services.
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


class FakeImageQuestionLLM(FakeLLM):
    async def read_picture(self, images, text):
        return 'Explain this picture.'

    async def plan(self, question):
        return {
            'subject': 'science',
            'wants_picture': True,
            'needs_current_facts': False,
        }

    async def answer(self, question, style, evidence):
        return {'answer': 'This picture shows a plant.', 'confident': True, 'needs_picture': True}

    async def with_gpu_released(self, operation):
        return await operation


class FakeImageGen:
    def __init__(self):
        self.prompts = []

    async def generate(self, prompt):
        self.prompts.append(prompt)
        return 'generated-image'


class PipelineSubjectRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_passes_planned_subject_to_retriever(self):
        retriever = FakeRetriever()
        parts = {'llm': FakeLLM(), 'retriever': retriever}
        job = {'text': 'Test subject routing for physics', 'images': [], 'level': 'class-9-12'}
        config = {'levels': {'class-9-12': 'Use suitable detail.'}}

        # Run the async pipeline and verify the planner's subject reaches retrieval.
        async for _ in run(job, parts, config):
            pass

        self.assertEqual(
            retriever.args,
            ('Test subject routing for physics', 'class-9-12', 'physics'),
        )


class PipelineUploadedImageTests(unittest.IsolatedAsyncioTestCase):
    async def test_explaining_uploaded_image_does_not_generate_another_image(self):
        imagegen = FakeImageGen()
        job = {'text': 'Explain this picture', 'images': ['uploaded-image'], 'level': 'class-9-12'}
        parts = {'llm': FakeImageQuestionLLM(), 'imagegen': imagegen}
        config = {'levels': {'class-9-12': 'Use suitable detail.'}}

        events = [event async for event in run(job, parts, config)]

        self.assertEqual(imagegen.prompts, [])
        self.assertFalse(any(value == 'Drawing a picture...' for kind, value in events if kind == 'status'))


class PictureGenerationDecisionTests(unittest.TestCase):
    def test_uploaded_image_does_not_trigger_a_supplementary_image(self):
        job = {'text': 'Explain this picture', 'images': ['uploaded-image']}

        self.assertFalse(_wants_picture(job))

    def test_explicit_no_image_instruction_overrides_model_flags(self):
        prompts = (
            'Please do not generate any image.',
            "I don't want a picture.",
            'No images, please.',
            'I asked not to generate any image.',
            'Without generating an image, explain the answer.',
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                job = {'text': prompt, 'images': []}
                self.assertFalse(_wants_picture(job))

    def test_uploaded_image_can_still_request_a_new_diagram(self):
        job = {'text': 'Draw a diagram to explain this problem.', 'images': ['uploaded-image']}

        self.assertTrue(_wants_picture(job))

    def test_text_only_model_flags_do_not_trigger_unsolicited_picture(self):
        job = {'text': 'Explain photosynthesis', 'images': []}

        self.assertFalse(_wants_picture(job))

    def test_explicit_text_only_picture_request_triggers_generation(self):
        job = {'text': 'Please draw a diagram of photosynthesis.', 'images': []}

        self.assertTrue(_wants_picture(job))


if __name__ == '__main__':
    unittest.main()
