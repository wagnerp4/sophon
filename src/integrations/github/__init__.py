from __future__ import annotations

from typing import Any

from integrations.github.client import github_request, github_token, github_tools_enabled

GITHUB_STATUS = "github_status"
GITHUB_REPOS = "github_repos"
GITHUB_REPO = "github_repo"
GITHUB_CREATE = "github_create"
GITHUB_TOOL_NAMES = frozenset({GITHUB_STATUS, GITHUB_REPOS, GITHUB_REPO, GITHUB_CREATE})

GITHUB_STATUS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": GITHUB_STATUS,
        "description": "Show whether a GitHub token is set and which user it authenticates.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
}

GITHUB_REPOS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": GITHUB_REPOS,
        "description": "List the authenticated user's GitHub repositories, newest update first.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "How many repos to return. Default 20. Max 50."},
            },
            "additionalProperties": False,
        },
    },
}

GITHUB_REPO_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": GITHUB_REPO,
        "description": "Read one repository the token can see. owner/name.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "owner/repo or repo name under the authenticated user."},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    },
}

GITHUB_CREATE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": GITHUB_CREATE,
        "description": (
            "Create a repository on GitHub for the authenticated user. "
            "Does not write a local checkout and does not touch other drives. "
            "Needs harness approval. Local files stay in the sophon workspace via editor_propose_edit."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Repository name."},
                "description": {"type": "string"},
                "private": {"type": "boolean", "description": "Default true."},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    },
}

GITHUB_TOOL_SYSTEM_HINT = (
    "Use github_status, github_repos, and github_repo for the authenticated GitHub account. "
    "Use github_create to create a remote repository. It does not git-init a folder on another drive. "
    "Do not use shell_exec or web_search to create or push repositories. "
    "web_search source=github only searches public repos."
)


def github_chat_tools() -> list[dict[str, Any]]:
    if not github_tools_enabled():
        return []
    return [GITHUB_STATUS_TOOL, GITHUB_REPOS_TOOL, GITHUB_REPO_TOOL, GITHUB_CREATE_TOOL]


def execute_github_tool(name: str, arguments: dict[str, Any]) -> str:
    if name == GITHUB_STATUS:
        return _status()
    if name == GITHUB_REPOS:
        limit = arguments.get("limit")
        try:
            count = int(limit) if limit is not None else 20
        except (TypeError, ValueError):
            count = 20
        return _repos(max(1, min(count, 50)))
    if name == GITHUB_REPO:
        return _repo(str(arguments.get("name") or ""))
    if name == GITHUB_CREATE:
        private = arguments.get("private")
        is_private = True if private is None else bool(private)
        return _create(
            str(arguments.get("name") or ""),
            str(arguments.get("description") or ""),
            is_private,
        )
    return f"error: unknown github tool {name!r}"


def _status() -> str:
    if not github_token():
        return "github: off (set SOPHON_GITHUB_TOKEN or GH_TOKEN in .env)"
    code, payload, err = github_request("GET", "/user")
    if err:
        return err
    if not isinstance(payload, dict):
        return f"error: github {code}"
    login = str(payload.get("login") or "")
    return f"github: on user={login} public_repos={payload.get('public_repos', '-')}"


def _repos(limit: int) -> str:
    code, payload, err = github_request("GET", f"/user/repos?per_page={limit}&sort=updated")
    if err:
        return err
    if not isinstance(payload, list):
        return f"error: github {code}"
    lines = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = str(item.get("full_name") or "")
        url = str(item.get("html_url") or "")
        private = "private" if item.get("private") else "public"
        lines.append(f"{name} {private} {url}")
    return "\n".join(lines) if lines else "(no repositories)"


def _repo(name: str) -> str:
    slug = name.strip().strip("/")
    if not slug:
        return "error: name is required"
    if "/" not in slug:
        code, user, err = github_request("GET", "/user")
        if err or not isinstance(user, dict):
            return err or "error: github user"
        slug = str(user.get("login") or "") + "/" + slug
    code, payload, err = github_request("GET", "/repos/" + slug)
    if err:
        return err
    if not isinstance(payload, dict):
        return f"error: github {code}"
    return "\n".join(
        [
            str(payload.get("full_name") or ""),
            str(payload.get("html_url") or ""),
            str(payload.get("clone_url") or ""),
            "private" if payload.get("private") else "public",
            str(payload.get("description") or ""),
            f"default_branch={payload.get('default_branch') or '-'}",
        ]
    )


def _create(name: str, description: str, private: bool) -> str:
    repo = name.strip().strip("/")
    if not repo or "/" in repo or "\\" in repo or ".." in repo:
        return "error: name must be a single repository name"
    code, payload, err = github_request(
        "POST",
        "/user/repos",
        {"name": repo, "description": description, "private": private, "auto_init": False},
    )
    if err:
        return err
    if not isinstance(payload, dict):
        return f"error: github {code}"
    return "\n".join(
        [
            "created " + str(payload.get("full_name") or repo),
            str(payload.get("html_url") or ""),
            str(payload.get("clone_url") or ""),
            "Local files stay in the sophon workspace. This did not write another drive.",
        ]
    )
