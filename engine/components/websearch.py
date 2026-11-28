import httpx
import trafilatura
import asyncio


class WebSearch:
    def __init__(self, config):
        c = config['components']['websearch']
        self.url, self.max_pages = c['url'], c['max_pages']
        self.http = httpx.AsyncClient(timeout=30, follow_redirects=True)

    async def search(self, question):
        r = await self.http.get(self.url, params={'q': question, 'format': 'json'})
        r.raise_for_status()

        async def fetch(hit):
            try:
                page = await self.http.get(hit['url'], timeout=10)
                text = trafilatura.extract(page.text) or hit.get('content', '')
            except Exception:
                text = hit.get('content', '')
            return {'source': hit['url'], 'text': text[:1500], 'score': 1.0}

        hits = r.json().get('results', [])[:self.max_pages]
        found = await asyncio.gather(*(fetch(hit) for hit in hits))
        return list(found)


def build(config, parts):
    return WebSearch(config)
