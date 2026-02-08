from .base_tool import BaseTool
import requests

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
    
    def execute(self, query: str, max_results: int = 5) -> list:
        """Search the web using DuckDuckGo and fetch page content"""
        try:
            if DDGS is None:
                return [{
                    "error": (
                        "Search dependency missing: install 'duckduckgo-search' or 'ddgs' "
                        "to enable web search."
                    )
                }]

            # Use DDGS library
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            
            simplified_results = []
            
            for result in results:
                try:
                    url = result.get('href') or result.get('url')
                    title = result.get('title') or "Untitled"
                    snippet = result.get('body') or result.get('snippet') or ""

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
                    
                    if BeautifulSoup is not None:
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
