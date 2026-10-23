import wikipedia
from duckduckgo_search import DDGS

def evolved_memory_tool(query: str) -> str:
    """
    Dynamically searches for information to answer the user's query.
    Args:
        query (str): The user's query.
    """
    # Try Wikipedia first
    try:
        search_results = wikipedia.search(query)
        if search_results:
            # Truncate content to prevent smolagents context limit overflow
            page_content = wikipedia.page(search_results[0], auto_suggest=False).content[:1500] 
            return f"WIKIPEDIA CONTENT: {page_content}..."
    except Exception:
        pass # Fallback to DuckDuckGo

    # Try DuckDuckGo
    try:
        results = DDGS().text(query, max_results=1)
        if results:
            # Extract actual snippet instead of just URL
            return f"DUCKDUCKGO SNIPPET: {results[0]['body']}"
        return 'No information found'
    except Exception as e:
        return f'An error occurred: {str(e)}'