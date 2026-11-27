import asyncio
import base64
import json
import random
import httpx


class ImageGen:
    def __init__(self, config):
        c = config['components']['imagegen']
        self.url = c['url']
        self.workflow = json.load(open(c['workflow']))
        self.prompt_node, self.seed_node, self.seed_field = c['prompt_node'], c['seed_node'], c['seed_field']
        self.http = httpx.AsyncClient(timeout=600)

    async def generate(self, prompt):
        wf = json.loads(json.dumps(self.workflow))
        prompt_inputs = wf[self.prompt_node]['inputs']
        if 'text' in prompt_inputs:
            prompt_inputs['text'] = prompt
        else:
            if 'clip_l' in prompt_inputs:
                prompt_inputs['clip_l'] = prompt
            if 't5xxl' in prompt_inputs:
                prompt_inputs['t5xxl'] = prompt
        wf[self.seed_node]['inputs'][self.seed_field] = random.randint(0, 2**31)
        r = await self.http.post(self.url + '/prompt', json={'prompt': wf})
        r.raise_for_status()
        pid = r.json()['prompt_id']
        while True:
            await asyncio.sleep(2)
            history = (await self.http.get(self.url + '/history/' + pid)).json()
            if pid in history:
                break
        image = next(i for node in history[pid]['outputs'].values() for i in node.get('images', []))
        raw = await self.http.get(self.url + '/view', params=image)
        await self.http.post(self.url + '/free', json={'unload_models': True, 'free_memory': True})
        return 'data:image/png;base64,' + base64.b64encode(raw.content).decode()


def build(config, parts):
    return ImageGen(config)
