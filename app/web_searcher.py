import requests
from bs4 import BeautifulSoup
from typing import List, Dict
from duckduckgo_search import DDGS

class WebSearcher:
    """Handles web searches and content extraction"""
    
    def __init__(self, max_results: int = 5):
        self.max_results = max_results
    
    def search(self, query: str) -> List[Dict[str, str]]:
        """
        Search the web for information
        Returns list of results with title, url, and snippet
        """
        print(f"🔍 Searching web for: {query}")
        
        try:
            # Using DuckDuckGo (free, no API key needed)
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))
            
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'title': result.get('title', ''),
                    'url': result.get('href', ''),
                    'snippet': result.get('body', ''),
                })
            
            print(f"✅ Found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            print(f"❌ Search error: {str(e)}")
            return []
    
    def fetch_content(self, url: str) -> str:
        """
        Fetch and extract main content from a URL
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(['script', 'style']):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text[:5000]  # Limit to first 5000 characters
            
        except Exception as e:
            print(f"❌ Error fetching {url}: {str(e)}")
            return ""
