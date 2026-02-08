from .base_tool import BaseTool
import warnings
import requests
from typing import Any, Dict, List

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    # Newer package name
    from ddgs import DDGS
except ImportError:
    try:
        # Backward-compatible import for duckduckgo-search
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

class SearchTool(BaseTool):
    def __init__(self, config: dict):
        self.config = config
    
    @property
    def name(self) -> str:
        return "search_web"
    
    @property
    def description(self) -> str:
        return "Search the web using DuckDuckGo for current information"
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find information on the web"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of search results to return",
                    "default": 5
                }
            },
            "required": ["query"]
        }

    def _normalize_max_results(self, max_results: int) -> int:
        """Clamp max results to a safe, useful range."""
        default_max = int(self.config.get('search', {}).get('max_results', 5))
        try:
            requested = int(max_results)
        except (TypeError, ValueError):
            requested = default_max

        if requested < 1:
            requested = default_max

        return max(1, min(requested, 10))

    def _perform_text_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Execute DDGS text search while suppressing legacy package rename warnings."""
        ddgs_module = str(getattr(DDGS, "__module__", ""))

        # Legacy duckduckgo_search emits a forced RuntimeWarning on DDGS() construction.
        if ddgs_module.startswith("duckduckgo_search"):
            original_warn = warnings.warn

            def _filtered_warn(message, *args, **kwargs):
                if "has been renamed to `ddgs`" in str(message):
                    return None
                return original_warn(message, *args, **kwargs)

            warnings.warn = _filtered_warn
            try:
                with DDGS() as ddgs:
                    return list(ddgs.text(query, max_results=max_results))
            finally:
                warnings.warn = original_warn

        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    
    def execute(self, query: str, max_results: int = 5) -> list:
        """Search the web using DuckDuckGo and fetch page content"""
        try:
            normalized_query = (query or "").strip()
            if not normalized_query:
                return [{"error": "Search query cannot be empty."}]

            if DDGS is None:
                return [{
                    "error": (
                        "Search dependency missing: install 'duckduckgo-search' or 'ddgs' "
                        "to enable web search."
                    )
                }]

            requested_max_results = self._normalize_max_results(max_results)
            results = self._perform_text_search(normalized_query, requested_max_results)
            
            simplified_results: List[Dict[str, Any]] = []
            seen_urls = set()
            
            for result in results:
                try:
                    url = result.get('href') or result.get('url')
                    title = result.get('title') or "Untitled"
                    snippet = result.get('body') or result.get('snippet') or ""

                    if url:
                        if url in seen_urls:
                            continue
                        seen_urls.add(url)

                    if not url:
                        simplified_results.append({
                            "title": title,
                            "url": "",
                            "snippet": snippet,
                            "content": "No URL returned by search provider"
                        })
                        continue

                    # Fetch content with requests
                    response = requests.get(
                        url,
                        headers={'User-Agent': self.config.get('search', {}).get('user_agent', 'Mozilla/5.0')},
                        timeout=10
                    )
                    response.raise_for_status()

                    content_type = str(response.headers.get("Content-Type", "")).lower()
                    is_text_like = (
                        "text/" in content_type or
                        "application/json" in content_type or
                        "application/xml" in content_type
                    )
                    
                    if not is_text_like:
                        text = f"Skipped non-text content type: {content_type or 'unknown'}"
                    elif BeautifulSoup is not None:
                        # Parse HTML with BeautifulSoup when available.
                        soup = BeautifulSoup(response.text, 'html.parser')

                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.decompose()

                        # Get text content
                        text = soup.get_text()
                    else:
                        # Minimal fallback without BeautifulSoup dependency.
                        text = response.text

                    # Clean up whitespace
                    text = ' '.join(text.split())
                    
                    # Limit content length
                    content_snippet = text[:1000] + "..." if len(text) > 1000 else text
                    
                    simplified_results.append({
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                        "content": content_snippet
                    })
                
                except Exception as e:
                    # If we can't fetch the page, still include the search result
                    simplified_results.append({
                        "title": result.get('title', 'Untitled'),
                        "url": result.get('href') or result.get('url', ''),
                        "snippet": result.get('body') or result.get('snippet', ''),
                        "content": f"Could not fetch content: {str(e)}"
                    })
            
            return simplified_results
        
        except Exception as e:
            return [{"error": f"Search failed: {str(e)}"}]
