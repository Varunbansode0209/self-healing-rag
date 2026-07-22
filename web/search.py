import os
from pathlib import Path

from dotenv import load_dotenv
from tavily import TavilyClient

# -----------------------------------------------------
# Load Environment Variables
# -----------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found in .env")

# -----------------------------------------------------
# Initialize Tavily Client
# -----------------------------------------------------

client = TavilyClient(api_key=TAVILY_API_KEY)


# -----------------------------------------------------
# Search Function
# -----------------------------------------------------

def search_web(
    query: str,
    max_results: int = 3,
    search_depth: str = "advanced",
) -> dict:
    """
    Search the web using Tavily.

    Returns:
        {
            "success": bool,
            "results": list[dict],   # empty list if success=False or no hits
            "error": str | None
        }
    """
    try:
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
        )

        results = [
            {
                "title":   item.get("title", ""),
                "url":     item.get("url", ""),
                "snippet": item.get("content", ""),
            }
            for item in response.get("results", [])
        ]

        return {"success": True, "results": results, "error": None}

    except Exception as e:
        return {"success": False, "results": [], "error": str(e)}


# -----------------------------------------------------
# Testing
# -----------------------------------------------------

if __name__ == "__main__":

    query = input("Enter your question: ")

    response = search_web(query)

    if not response["success"]:
        print(f"Error: {response['error']}")
    elif not response["results"]:
        print("No results found.")
    else:
        print("\nSearch Results\n")

        for index, result in enumerate(response["results"], start=1):
            print("=" * 80)
            print(f"Result {index}")
            print(f"Title   : {result['title']}")
            print(f"URL     : {result['url']}")
            print(f"Snippet : {result['snippet']}")
            print("=" * 80)