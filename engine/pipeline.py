# Description: Orchestrates tutoring, retrieval, web search, verification, and images.
# Author: Dipesh Karmakar <dipesh.karmakar@gmail.com>
# Modifications:
# - 433859a: Passed the planned subject to textbook retrieval.
# - 7de329d: Serialized model requests and handled image-service failures.
# - 374a71c: Added caches, image timeouts, and zero-sample verifier skipping.
# - 3898fb9: Optimized pipeline concurrency and added request timeouts.
# - f9dc837: Added the initial tutor pipeline.
# - Uncommitted: Passes the planned subject to retrieval and prevents unwanted images.

import logging
from collections import OrderedDict
import time
import hashlib
import re

log = logging.getLogger('engine')

_answer_cache = OrderedDict()
_answer_cache_limit = 256
_answer_cache_ttl_seconds = 600


# Cache keys normalize whitespace and include the learner's level.
def _cache_key(question, level):
    normalized = ' '.join(question.lower().split())
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    return level, digest


def _get_cached_answer(question, level, allow_cache):
    if not allow_cache:
        return None
    key = _cache_key(question, level)
    item = _answer_cache.get(key)
    if item is None:
        return None
    answer, expires_at = item
    if expires_at <= time.monotonic():
        _answer_cache.pop(key, None)
        return None
    _answer_cache.move_to_end(key)
    return answer


def _set_cached_answer(question, level, answer):
    key = _cache_key(question, level)
    _answer_cache[key] = (answer, time.monotonic() + _answer_cache_ttl_seconds)
    _answer_cache.move_to_end(key)
    while len(_answer_cache) > _answer_cache_limit:
        _answer_cache.popitem(last=False)


async def safe(coro, default):
    try:
        return await coro
    except Exception as err:
        # Log component failures while letting the rest of the answer pipeline continue.
        log.warning('component failed: %s', err)
        return default


_PICTURE_TERMS = r'(?:images?|pictures?|illustrations?|diagrams?|maps?|charts?)'
_NO_PICTURE_PATTERNS = (
    rf'\b(?:do not|don\'t|dont|never|no need to|not to)\s+'
    rf'(?:generate|create|draw|make|produce|illustrate|show|include)\b[^.!?\n]{{0,80}}\b{_PICTURE_TERMS}\b',
    rf'\bwithout\s+(?:generating|creating|drawing|making|illustrating|showing|including)\b'
    rf'[^.!?\n]{{0,80}}\b{_PICTURE_TERMS}\b',
    rf'\bno\s+{_PICTURE_TERMS}\b',
    rf'\b(?:i\s+)?(?:do not|don\'t|dont)\s+want\b[^.!?\n]{{0,80}}\b{_PICTURE_TERMS}\b',
)
_PICTURE_REQUEST_PATTERN = (
    rf'\b(?:draw|generate|create|make|produce|illustrate|show|include)\b'
    rf'[^.!?\n]{{0,80}}\b{_PICTURE_TERMS}\b'
)


def _wants_picture(job, plan, draft):
    user_text = ' '.join((job.get('text') or '').split())
    # Explicit user opt-outs take precedence over both model-generated flags.
    if any(re.search(pattern, user_text, flags=re.IGNORECASE) for pattern in _NO_PICTURE_PATTERNS):
        return False
    if job.get('images'):
        # An uploaded image is context, not permission to generate a second image.
        return bool(re.search(_PICTURE_REQUEST_PATTERN, user_text, flags=re.IGNORECASE))
    return bool(plan.get('wants_picture') or draft.get('needs_picture'))


def format_reply(draft, evidence, picture, report):
    text = draft['answer'].strip()
    if picture:
        text = re.sub(r"^(?:I (?:cannot|can't|am unable to) (?:draw|generate)[^.]*\.\s*)+", '', text, flags=re.IGNORECASE)
        text = re.sub(r"^(?:However,\s*)?I can describe[^.]*\.\s*", '', text, flags=re.IGNORECASE)
    if evidence:
        text += '\n\nSources:\n' + '\n'.join('- [' + str(i + 1) + '] ' + e['source'] for i, e in enumerate(evidence))
    if picture:
        text += '\n\n![picture](' + picture + ')\n\nGenerated pictures can contain spelling mistakes in labels.'
    if report['passed']:
        if report['checks']:
            text += '\n\nChecked: ' + ', '.join(report['checks']) + '.'
    else:
        text += '\n\nI could not fully confirm this answer (' + '; '.join(report['notes']) + '). Please check it before relying on it.'
    return text


async def run(job, parts, config):
    llm, level = parts['llm'], job['level']
    style = config['levels'][level]
    # 1 and 2: receive and understand
    question = job['text']
    cached = _get_cached_answer(question, level, not job['images'])
    if cached is not None:
        yield 'reply', cached
        return
    if job['images']:
        yield 'status', 'Reading your picture...'
        question = await llm.read_picture(job['images'], job['text'])
    if 'retriever' in parts:
        yield 'status', 'Looking in the textbooks...'
    plan = await safe(llm.plan(question), {'subject': '', 'wants_picture': False, 'needs_current_facts': False})
    evidence = await safe(parts['retriever'].search(question, level, plan.get('subject')), []) if 'retriever' in parts else []
    # 3: answer from what it knows
    draft = await llm.answer(question, style, evidence)
    # 4: search the internet, only if needed
    weak = (not draft['confident']) or plan['needs_current_facts']
    if weak and 'websearch' in parts:
        yield 'status', 'Searching the web...'
        found = await safe(parts['websearch'].search(question), [])
        if found:
            evidence = evidence + found
            draft = await llm.answer(question, style, evidence)
    # 5: verify, and revise before switching models for image generation
    report = {'passed': True, 'notes': [], 'checks': []}
    if 'verifier' in parts:
        rounds = config['components']['verifier']['max_rounds']
        for attempt in range(rounds + 1):
            yield 'status', 'Checking the answer...'
            report = await parts['verifier'].check(question, level, draft, evidence)
            if report['passed'] or attempt == rounds:
                break
            draft = await llm.answer(question, style, evidence, '; '.join(report['notes']))
    # 6: draw a picture only after all LLM work is complete
    picture = None
    wants = _wants_picture(job, plan, draft)
    if wants and 'imagegen' in parts:
        yield 'status', 'Drawing a picture...'
        picture = await llm.with_gpu_released(parts['imagegen'].generate(draft.get('picture_prompt') or question))
    elif plan['wants_picture'] and wants:
        draft['answer'] += '\n\n(Picture generation is switched off.)'
    # 7: reply
    reply = format_reply(draft, evidence, picture, report)
    if not job['images'] and not plan['needs_current_facts'] and not wants and report['passed']:
        _set_cached_answer(question, level, reply)
    yield 'reply', reply
