import logging

log = logging.getLogger('engine')


async def safe(coro, default):
    try:
        return await coro
    except Exception as err:
        log.warning('component failed: %s', err)
        return default


def format_reply(draft, evidence, picture, report):
    text = draft['answer'].strip()
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
    if job['images']:
        yield 'status', 'Reading your picture...'
        question = await llm.read_picture(job['images'], job['text'])
    plan = await safe(llm.plan(question), {'subject': '', 'wants_picture': False, 'needs_current_facts': False})
    # 3: answer from what it knows
    evidence = []
    if 'retriever' in parts:
        yield 'status', 'Looking in the textbooks...'
        evidence = await safe(parts['retriever'].search(question, level), [])
    draft = await llm.answer(question, style, evidence)
    # 4: search the internet, only if needed
    weak = (not evidence) or (not draft['confident']) or plan['needs_current_facts']
    if weak and 'websearch' in parts:
        yield 'status', 'Searching the web...'
        found = await safe(parts['websearch'].search(question), [])
        if found:
            evidence = evidence + found
            draft = await llm.answer(question, style, evidence)
    # 5: draw a picture, only if needed
    picture = None
    wants = plan['wants_picture'] or draft.get('needs_picture')
    if wants and 'imagegen' in parts:
        yield 'status', 'Drawing a picture...'
        await safe(llm.free_gpu(), None)
        picture = await safe(parts['imagegen'].generate(draft.get('picture_prompt') or question), None)
    elif plan['wants_picture']:
        draft['answer'] += '\n\n(Picture generation is switched off.)'
    # 6: verify, and revise if a check fails
    report = {'passed': True, 'notes': [], 'checks': []}
    if 'verifier' in parts:
        rounds = config['components']['verifier']['max_rounds']
        for attempt in range(rounds + 1):
            yield 'status', 'Checking the answer...'
            report = await parts['verifier'].check(question, level, draft, evidence)
            if report['passed'] or attempt == rounds:
                break
            draft = await llm.answer(question, style, evidence, '; '.join(report['notes']))
    # 7: reply
    yield 'reply', format_reply(draft, evidence, picture, report)
