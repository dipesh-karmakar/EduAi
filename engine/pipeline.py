# Description: Orchestrates tutoring, retrieval, web search, verification, and images.
# Author: Dipesh Karmakar <dipesh.karmakar@gmail.com>
# Modifications:
# - 433859a: Passed the planned subject to textbook retrieval.
# - 7de329d: Serialized model requests and handled image-service failures.
# - 374a71c: Added caches, image timeouts, and zero-sample verifier skipping.
# - 3898fb9: Optimized pipeline concurrency and added request timeouts.
# - f9dc837: Added the initial tutor pipeline.
# - Uncommitted: Adds subject-aware retrieval, image-generation guards, and stage timings.

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


async def _timed(job, timings, stage, operation):
    started = time.perf_counter()
    try:
        return await operation
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000
        timings[stage] = timings.get(stage, 0) + elapsed_ms
        log.info(
            'request_stage request_id=%s stage=%s duration_ms=%.1f',
            job.get('request_id', '-'),
            stage,
            elapsed_ms,
        )


def _log_request_timing(job, level, timings, started, cache_hit):
    stages = ' '.join(f'{stage}_ms={elapsed:.1f}' for stage, elapsed in timings.items())
    log.info(
        'request_timing request_id=%s level=%s cache_hit=%s total_ms=%.1f stages=%s',
        job.get('request_id', '-'),
        level,
        cache_hit,
        (time.perf_counter() - started) * 1000,
        stages or 'none',
    )


_PICTURE_TERMS = r'(?:images?|pictures?|illustrations?|diagrams?|maps?|charts?)'
_NO_PICTURE_PATTERNS = (
    rf'\b(?:do not|don\'t|dont|never|no need to|not to|not)\s+'
    rf'(?:generate|create|draw|make|produce|illustrate|show|include)\b[^.!?\n]{{0,80}}\b{_PICTURE_TERMS}\b',
    rf'\bwithout\s+(?:generating|creating|drawing|making|illustrating|showing|including)\b'
    rf'[^.!?\n]{{0,80}}\b{_PICTURE_TERMS}\b',
    rf'\bno\s+need\s+for\b[^.!?\n]{{0,80}}\b{_PICTURE_TERMS}\b',
    rf'\bno\s+{_PICTURE_TERMS}\b',
    rf'\b(?:i\s+)?(?:do not|don\'t|dont)\s+want\b[^.!?\n]{{0,80}}\b{_PICTURE_TERMS}\b',
)
_PICTURE_REQUEST_PATTERN = (
    rf'\b(?:draw|generate|create|make|produce|illustrate|show|include|provide)\b'
    rf'[^.!?\n]{{0,80}}\b{_PICTURE_TERMS}\b'
    rf'|\b(?:i want|i need|i would like)\b[^.!?\n]{{0,80}}\b{_PICTURE_TERMS}\b'
)


def _wants_picture(job):
    user_text = ' '.join((job.get('text') or '').split())
    # Generate only on explicit user requests; model flags cannot initiate image work.
    if any(re.search(pattern, user_text, flags=re.IGNORECASE) for pattern in _NO_PICTURE_PATTERNS):
        return False
    return bool(re.search(_PICTURE_REQUEST_PATTERN, user_text, flags=re.IGNORECASE))


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
    request_started = time.perf_counter()
    timings = {}
    llm, level = parts['llm'], job['level']
    style = config['levels'][level]
    # 1 and 2: receive and understand
    question = job['text']
    cached = _get_cached_answer(question, level, not job['images'])
    if cached is not None:
        _log_request_timing(job, level, timings, request_started, cache_hit=True)
        yield 'reply', cached
        return
    if job['images']:
        yield 'status', 'Reading your picture...'
        question = await _timed(
            job, timings, 'image_read', llm.read_picture(job['images'], job['text'])
        )
    if 'retriever' in parts:
        yield 'status', 'Looking in the textbooks...'
    plan = await _timed(
        job,
        timings,
        'planning',
        safe(llm.plan(question), {'subject': '', 'wants_picture': False, 'needs_current_facts': False}),
    )
    evidence = await _timed(
        job,
        timings,
        'retrieval',
        safe(parts['retriever'].search(question, level, plan.get('subject')), []),
    ) if 'retriever' in parts else []
    # 3: answer from what it knows
    draft = await _timed(job, timings, 'answer', llm.answer(question, style, evidence))
    # 4: search the internet, only if needed
    weak = (not draft['confident']) or plan['needs_current_facts']
    if weak and 'websearch' in parts:
        yield 'status', 'Searching the web...'
        found = await _timed(job, timings, 'web_search', safe(parts['websearch'].search(question), []))
        if found:
            evidence = evidence + found
            draft = await _timed(job, timings, 'answer', llm.answer(question, style, evidence))
    # 5: verify, and revise before switching models for image generation
    report = {'passed': True, 'notes': [], 'checks': []}
    if 'verifier' in parts:
        rounds = config['components']['verifier']['max_rounds']
        for attempt in range(rounds + 1):
            yield 'status', 'Checking the answer...'
            report = await _timed(
                job,
                timings,
                'verification',
                parts['verifier'].check(question, level, draft, evidence),
            )
            if report['passed'] or attempt == rounds:
                break
            draft = await _timed(
                job,
                timings,
                'answer',
                llm.answer(question, style, evidence, '; '.join(report['notes'])),
            )
    # 6: draw a picture only after all LLM work is complete
    picture = None
    wants = _wants_picture(job)
    if wants and 'imagegen' in parts:
        yield 'status', 'Drawing a picture...'
        picture = await _timed(
            job,
            timings,
            'image_generation',
            llm.with_gpu_released(parts['imagegen'].generate(draft.get('picture_prompt') or question)),
        )
    elif plan['wants_picture'] and wants:
        draft['answer'] += '\n\n(Picture generation is switched off.)'
    # 7: reply
    reply = format_reply(draft, evidence, picture, report)
    if not job['images'] and not plan['needs_current_facts'] and not wants and report['passed']:
        _set_cached_answer(question, level, reply)
    _log_request_timing(job, level, timings, request_started, cache_hit=False)
    yield 'reply', reply
