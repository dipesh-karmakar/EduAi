import json
import time
import uuid
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from components import load
from pipeline import run

config = yaml.safe_load(open('config.yaml'))
parts = load(config)
app = FastAPI()
PREFIX = 'tutor-'


@app.get('/v1/models')
def models():
    return {
        'object': 'list',
        'data': [{'id': PREFIX + name, 'object': 'model', 'owned_by': 'local'} for name in config['levels']],
    }


def split_message(messages):
    content = messages[-1]['content']
    if isinstance(content, str):
        return content, []
    text = ' '.join(p['text'] for p in content if p['type'] == 'text')
    images = [p['image_url']['url'] for p in content if p['type'] == 'image_url']
    return text, images


def chunk(model, delta, finish=None):
    data = {
        'id': 'chatcmpl-' + uuid.uuid4().hex,
        'object': 'chat.completion.chunk',
        'created': int(time.time()),
        'model': model,
        'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}],
    }
    return 'data: ' + json.dumps(data) + '\n\n'


@app.post('/v1/chat/completions')
async def chat(request: Request):
    body = await request.json()
    model = body['model']
    level = model[len(PREFIX):] if model.startswith(PREFIX) else 'class-9-12'
    if level not in config['levels']:
        level = 'class-9-12'
    text, images = split_message(body['messages'])
    job = {'text': text, 'images': images, 'level': level}

    async def stream():
        yield chunk(model, {'role': 'assistant'})
        async for kind, value in run(job, parts, config):
            yield chunk(model, {'content': ('_' + value + '_\n\n') if kind == 'status' else value})
        yield chunk(model, {}, 'stop')
        yield 'data: [DONE]\n\n'

    if body.get('stream'):
        return StreamingResponse(stream(), media_type='text/event-stream')
    final = ''
    async for kind, value in run(job, parts, config):
        if kind == 'reply':
            final = value
    return JSONResponse({
        'id': 'chatcmpl-' + uuid.uuid4().hex,
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': model,
        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': final}, 'finish_reason': 'stop'}],
    })
