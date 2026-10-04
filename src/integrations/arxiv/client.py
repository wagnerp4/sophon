from pathlib import Path


def arxiv_tools_enabled() -> bool:
    """Check if arxiv tools are available."""
    arxiv_server = Path(__file__).parent.parent.parent.parent / ".mcp" / "arxiv_server.py"
    return arxiv_server.exists()
