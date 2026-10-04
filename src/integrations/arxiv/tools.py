from typing import Any
import json
import subprocess
import sys

ARXIV_TOOL_SYSTEM_HINT = (
    "Use arxiv_search to find research papers by topic, author, or keyword. "
    "Use arxiv_get_paper for detailed information about a specific paper by ID."
)

ARXIV_TOOL_NAMES = {"arxiv_search", "arxiv_get_paper"}


def _call_arxiv_server(method: str, **params) -> dict[str, Any]:
    """Call the arxiv MCP server."""
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                ".mcp/arxiv_server.py",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="/home/philipp/software/python/Signal Processing/NLP/Personal/sophon",
        )
        request = json.dumps({"method": method, "params": params})
        stdout, stderr = process.communicate(input=request, timeout=30)

        if stderr:
            return {"error": f"Server error: {stderr}"}

        response = json.loads(stdout)
        return response.get("result", response)
    except json.JSONDecodeError as e:
        return {"error": f"Invalid server response: {e}"}
    except subprocess.TimeoutExpired:
        return {"error": "arxiv server timeout"}
    except Exception as e:
        return {"error": f"arxiv server error: {e}"}


def arxiv_search(query: str, max_results: int = 10) -> dict[str, Any]:
    """Search arXiv for papers matching the query.

    Args:
        query: Search query (keywords, author, title)
        max_results: Maximum number of results (default 10)

    Returns:
        List of papers with title, authors, abstract, PDF URL, etc.
    """
    if not query or not query.strip():
        return {"error": "Query cannot be empty"}

    result = _call_arxiv_server("search", query=query, max_results=min(max_results, 100))

    if isinstance(result, list):
        return {
            "count": len(result),
            "papers": result
        }
    return result


def arxiv_get_paper(paper_id: str) -> dict[str, Any]:
    """Get detailed information about a specific paper.

    Args:
        paper_id: arXiv paper ID (e.g., "2306.04338v1" or "http://arxiv.org/abs/2306.04338v1")

    Returns:
        Paper details including title, authors, abstract, PDF URL, publication date.
    """
    if not paper_id or not paper_id.strip():
        return {"error": "Paper ID cannot be empty"}

    # Extract just the ID if a full URL was provided
    if "arxiv.org" in paper_id:
        paper_id = paper_id.split("/abs/")[-1]

    return _call_arxiv_server("get", paper_id=paper_id)


def arxiv_chat_tools() -> list[dict[str, Any]]:
    """Return arxiv tools for chat."""
    return [
        {
            "name": "arxiv_search",
            "description": "Search arXiv for research papers by topic, author, or keyword",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'transformer models', 'author:Bengio', 'title:neural networks')",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (1-100, default 10)",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "arxiv_get_paper",
            "description": "Get detailed information about a specific arXiv paper by ID",
            "input_schema": {
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "arXiv paper ID (e.g., '2306.04338v1' or full URL)",
                    },
                },
                "required": ["paper_id"],
            },
        },
    ]


def execute_arxiv_tool(name: str, args: dict[str, Any]) -> str:
    """Execute an arxiv tool by name."""
    try:
        if name == "arxiv_search":
            query = str(args.get("query", ""))
            max_results = int(args.get("max_results", 10))
            result = arxiv_search(query, max_results)
        elif name == "arxiv_get_paper":
            paper_id = str(args.get("paper_id", ""))
            result = arxiv_get_paper(paper_id)
        else:
            return f"error: unknown arxiv tool '{name}'"

        if isinstance(result, dict) and "error" in result:
            return f"error: {result['error']}"

        return json.dumps(result, indent=2)
    except Exception as e:
        return f"error: {str(e)}"
