# Description: Provides serialized chat, planning, and GPU-release model requests.
# Author: Dipesh Karmakar <dipesh.karmakar@gmail.com>
# Modifications:
# - 7de329d: Serialized inference and added image-generation prompt guidance.
# - f9dc837: Added the initial language-model component.

import asyncio
import json
import re
import httpx


def _parse_json(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.S).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find('{'), text.rfind('}')
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


class LLM:
    def __init__(self, cfg):
        self.base = cfg['base_url']
        self.chat_model = cfg['chat_model']
        self.free_model = cfg['free_model']
        self.infer_lock = asyncio.Lock()
        self.http = httpx.AsyncClient(timeout=600)

    async def complete(self, messages, model=None, schema=None, temperature=0.3, max_tokens=900):
        # The inference server swaps a single active model, so serialize all calls.
        async with self.infer_lock:
            return await self._complete(messages, model, schema, temperature, max_tokens)

    async def _complete(self, messages, model=None, schema=None, temperature=0.3, max_tokens=900):
        body = {
            'model': model or self.chat_model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
        }
        if schema:
            body['response_format'] = {'type': 'json_object', 'schema': schema}
        r = await self.http.post(self.base + '/chat/completions', json=body)
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content']

    async def ask_json(self, system, user, schema, **kw):
        sys_prompt = (
            system
            + '\nRespond ONLY with a valid JSON object matching this schema (no markdown, no extra keys): '
            + json.dumps(schema)
        )
        text = await self.complete(
            [{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': user}],
            schema=schema,
            **kw,
        )
        return _parse_json(text)

    async def read_picture(self, images, text):
        parts = [
            {
                'type': 'text',
                'text': 'Write out the exact question in this picture as plain text, with all numbers, options and diagram labels. Say which part is unclear. ' + text,
            }
        ]
        parts += [{'type': 'image_url', 'image_url': {'url': url}} for url in images]
        return await self.complete([{'role': 'user', 'content': parts}], temperature=0, max_tokens=600)

    async def plan(self, question):
        schema = {
            'type': 'object',
            'properties': {
                'subject': {'type': 'string'},
                'wants_picture': {'type': 'boolean'},
                'needs_current_facts': {'type': 'boolean'},
            },
            'required': ['subject', 'wants_picture', 'needs_current_facts'],
        }
        system = 'Classify the student question. wants_picture is true only if the student asks for a picture, drawing or diagram. needs_current_facts is true if the answer depends on recent events, news, prices or dates.'
        out = await self.ask_json(system, question, schema, temperature=0, max_tokens=150)
        out.setdefault('subject', '')
        out.setdefault('wants_picture', False)
        out.setdefault('needs_current_facts', False)
        return out

    async def answer(self, question, style, evidence, notes='', temperature=0.3):
        schema = {
            'type': 'object',
            'properties': {
                'answer': {'type': 'string'},
                'confident': {'type': 'boolean'},
                'needs_picture': {'type': 'boolean'},
                'picture_prompt': {'type': 'string'},
                'final_expression': {'type': 'string'},
            },
            'required': ['answer', 'confident'],
        }
        sources = '\n\n'.join('[' + str(i + 1) + '] ' + e['source'] + ': ' + e['text'] for i, e in enumerate(evidence))
        system = (
            'You are a patient, accurate tutor. Be correct first and simple second. '
            + style
            + ' Use the numbered sources when they help and cite them like [1]. '
            + 'If the sources do not contain the answer and you are not sure, set confident to false. '
            + 'Set needs_picture to true only if the student explicitly requested a new picture or diagram, and then write picture_prompt. '
            + 'For picture prompts, describe a clear, uncluttered composition with accurate visual relationships. Do not request text, labels, lettering, or watermarks in the image; put necessary labels in the written answer instead. '
            + 'When image generation is available, do not say you cannot draw or generate images. '
            + 'If the answer is a calculation, put the final calculation in final_expression as a plain expression with numbers only.'
        )
        user = 'Sources:\n' + sources + '\n\nQuestion: ' + question + ('\n\nFix these problems: ' + notes if notes else '')
        out = await self.ask_json(system, user, schema, temperature=temperature, max_tokens=1200)
        if 'answer' not in out:
            for alt in ('response', 'explanation', 'content', 'text', 'reply'):
                if isinstance(out.get(alt), str):
                    out['answer'] = out[alt]
                    break
            else:
                out['answer'] = json.dumps(out)
        out.setdefault('confident', False)
        return out

    async def short_final(self, answer):
        schema = {'type': 'object', 'properties': {'final': {'type': 'string'}}, 'required': ['final']}
        out = await self.ask_json(
            'Extract only the final result of this answer in a few words or one number.',
            answer,
            schema,
            temperature=0,
            max_tokens=60,
        )
        return str(out.get('final', next(iter(out.values()), '')))

    async def with_gpu_released(self, operation):
        # Keep inference locked while image generation uses the GPU after unloading the chat model.
        async with self.infer_lock:
            await self._complete(
                [{'role': 'user', 'content': 'hi'}],
                model=self.free_model,
                max_tokens=1,
            )
            return await operation


def build(config, parts):
    return LLM(config['components']['llm'])
