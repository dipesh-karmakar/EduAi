# Description: Checks answer calculations, source support, and reading level.
# Author: Dipesh Karmakar <dipesh.karmakar@gmail.com>
# Modifications:
# - 3898fb9: Skipped repeated-answer checks when the sample count is zero.
# - f9dc837: Added the initial answer verifier.

import math
import re
import textstat
from simpleeval import simple_eval

FUNCS = {'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan, 'log': math.log, 'exp': math.exp}
NAMES = {'pi': math.pi, 'e': math.e}


class Verifier:
    def __init__(self, config, llm):
        c = config['components']['verifier']
        self.llm, self.samples = llm, c['samples']
        self.grades, self.styles = config['grades'], config['levels']

    async def check(self, question, level, draft, evidence):
        notes, passed = [], []
        answer = draft['answer']
        expr = draft.get('final_expression', '').replace('^', '**')
        if expr:
            try:
                value = float(simple_eval(expr, functions=FUNCS, names=NAMES))
                numbers = [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', answer.replace(',', ''))]
                if any(abs(value - n) <= 0.01 * max(1.0, abs(value)) for n in numbers):
                    passed.append('maths recomputed')
                else:
                    notes.append('the calculation gives ' + str(round(value, 4)) + ', which the answer does not state')
            except Exception:
                notes.append('the final calculation could not be evaluated')
        if evidence:
            schema = {'type': 'object', 'properties': {'unsupported': {'type': 'array', 'items': {'type': 'string'}}}, 'required': ['unsupported']}
            sources = '\n\n'.join(e['text'] for e in evidence)
            system = 'List the claims in the answer that the sources do not support. Ignore common knowledge and simple arithmetic. Return an empty list if every claim is supported.'
            out = await self.llm.ask_json(system, 'Sources:\n' + sources + '\n\nAnswer:\n' + answer, schema, temperature=0, max_tokens=300)
            if out['unsupported']:
                notes.append('not supported by the sources: ' + '; '.join(out['unsupported'][:3]))
            else:
                passed.append('claims matched to sources')
        if expr and self.samples > 0:
            finals = [await self.llm.short_final(answer)]
            for _ in range(self.samples):
                again = await self.llm.answer(question, self.styles[level], evidence, temperature=0.7)
                finals.append(await self.llm.short_final(again['answer']))
            same = [f for f in finals if f.strip().lower() == finals[0].strip().lower()]
            if len(same) * 2 > len(finals):
                passed.append('repeated answers agree')
            else:
                notes.append('repeated answers disagree')
        if textstat.flesch_kincaid_grade(answer) <= self.grades[level] + 2:
            passed.append('level fits')
        else:
            notes.append('the wording is too hard for this level')
        return {'passed': not notes, 'notes': notes, 'checks': passed}


def build(config, parts):
    return Verifier(config, parts['llm'])
