# Description: Searches the web and extracts text from relevant result pages.
# Author: Dipesh Karmakar <dipesh.karmakar@gmail.com>
# Modifications:
# - 374a71c: Added a search cache and bounded web-page retrieval.
# - 3898fb9: Parallelized page retrieval and added timeouts.
# - f9dc837: Added the initial web-search component.

import httpx
import trafilatura
import asyncio
from collections import OrderedDict
import time


class WebSearch:
    def __init__(self, config):
        c = config['components']['websearch']
        self.url, self.max_pages = c['url'], c['max_pages']
        self.http = httpx.AsyncClient(timeout=30, follow_redirects=True)
        self._search_cache = OrderedDict()
        self._search_cache_limit = 128
        self._search_cache_ttl_seconds = 300

    async def search(self, question):
        # Cache by normalized query text to avoid repeating identical network searches.
        key = ' '.join(question.lower().split())
        cached = self._search_cache.get(key)
        if cached is not None:
            found, expires_at = cached
            if expires_at > time.monotonic():
                self._search_cache.move_to_end(key)
                return found
            self._search_cache.pop(key, None)

        r = await self.http.get(self.url, params={'q': question, 'format': 'json'})
        r.raise_for_status()

        # Fetch result pages concurrently and keep snippets if a page cannot be extracted.
        async def fetch(hit):
            try:
                page = await self.http.get(hit['url'], timeout=10)
                text = trafilatura.extract(page.text) or hit.get('content', '')
            except Exception:
                text = hit.get('content', '')
            return {'source': hit['url'], 'text': text[:1500], 'score': 1.0}

        hits = r.json().get('results', [])[:self.max_pages]
        found = await asyncio.gather(*(fetch(hit) for hit in hits))
        found = list(found)
        self._search_cache[key] = (found, time.monotonic() + self._search_cache_ttl_seconds)
        self._search_cache.move_to_end(key)
        while len(self._search_cache) > self._search_cache_limit:
            self._search_cache.popitem(last=False)
        return found


def build(config, parts):
    return WebSearch(config)
