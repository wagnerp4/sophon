#!/usr/bin/env python3
import json
import sys
import arxiv
from typing import Any

def search_papers(query: str, max_results: int = 10) -> list[dict]:
    """Search arXiv for papers matching the query."""
    try:
        client = arxiv.Client()
        results = []
        for paper in client.results(arxiv.Search(query=query, max_results=max_results)):
            results.append({
                "id": paper.entry_id,
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "summary": paper.summary,
                "published": str(paper.published),
                "pdf_url": paper.pdf_url,
                "categories": paper.categories,
            })
        return results
    except Exception as e:
        return [{"error": str(e)}]

def get_paper(paper_id: str) -> dict:
    """Get detailed information about a specific paper."""
    try:
        client = arxiv.Client()
        paper = next(client.results(arxiv.Search(id_list=[paper_id])))
        return {
            "id": paper.entry_id,
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "summary": paper.summary,
            "published": str(paper.published),
            "pdf_url": paper.pdf_url,
            "categories": paper.categories,
        }
    except Exception as e:
        return {"error": str(e)}

def main():
    """Simple stdio-based MCP server for arXiv."""
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break

            request = json.loads(line)
            method = request.get("method")
            params = request.get("params", {})

            if method == "search":
                result = search_papers(params.get("query", ""), params.get("max_results", 10))
                response = {"result": result}
            elif method == "get":
                result = get_paper(params.get("paper_id", ""))
                response = {"result": result}
            else:
                response = {"error": f"Unknown method: {method}"}

            print(json.dumps(response))
            sys.stdout.flush()
        except json.JSONDecodeError:
            print(json.dumps({"error": "Invalid JSON"}))
            sys.stdout.flush()
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            sys.stdout.flush()

if __name__ == "__main__":
    main()
