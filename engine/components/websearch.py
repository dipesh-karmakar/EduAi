import httpx
import trafilatura


class WebSearch:
    def __init__(self, config):
        c = config['components']['websearch']
        self.url, self.max_pages = c['url'], c['max_pages']
        self.http = httpx.AsyncClient(timeout=30, follow_redirects=True)

    async def search(self, question):
        r = await self.http.get(self.url, params={'q': question, 'format': 'json'})
        r.raise_for_status()
        found = []
        for hit in r.json().get('results', [])[:self.max_pages]:
            try:
                page = await self.http.get(hit['url'])
                text = trafilatura.extract(page.text) or hit.get('content', '')
            except Exception:
                text = hit.get('content', '')
            found.append({'source': hit['url'], 'text': text[:1500], 'score': 1.0})
        return found


def build(config, parts):
    return WebSearch(config)
