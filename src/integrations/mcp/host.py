from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RetraceTool:
    name: str
    connector: str
    schema: dict[str, Any]


def mcp_enabled() -> bool:
    raw = os.environ.get("SOPHON_MCP", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def disabled_ids() -> set[str]:
    raw = os.environ.get("SOPHON_MCP_DISABLED", "")
    return {bit.strip().lower() for bit in raw.split(",") if bit.strip()}


def _root(project_root: object) -> Path:
    if isinstance(project_root, Path):
        return project_root
    return Path.cwd()


def configured_servers(project_root: object) -> list[dict[str, Any]]:
    path = _root(project_root) / ".sophon" / "mcp.yaml"
    if path.is_file():
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            data = {}
        servers = data.get("servers") if isinstance(data, dict) else None
        if isinstance(servers, list):
            return [item for item in servers if isinstance(item, dict)]
        return []
    return [
        {"id": "obsidian", "transport": "inprocess"},
        {"id": "zotero", "transport": "inprocess"},
    ]


def _server_live(server: dict[str, Any]) -> bool:
    if server.get("disabled"):
        return False
    server_id = str(server.get("id") or "").strip().lower()
    if not server_id or server_id in disabled_ids():
        return False
    transport = str(server.get("transport") or "inprocess").strip().lower()
    if transport != "inprocess":
        # TODO: stdio MCP client for servers that are not wrapped as NexusTools
        return False
    return True


def retrace_tools(project_root: object) -> list[RetraceTool]:
    if not mcp_enabled():
        return []
    rows: list[RetraceTool] = []
    for server in configured_servers(project_root):
        if not _server_live(server):
            continue
        server_id = str(server.get("id") or "").strip().lower()
        if server_id == "obsidian":
            rows.extend(_obsidian_retrace())
        elif server_id == "zotero":
            rows.extend(_zotero_retrace())
    return rows


def _obsidian_retrace() -> list[RetraceTool]:
    from integrations.obsidian.client import obsidian_tools_enabled
    from integrations.obsidian.tools import obsidian_chat_tools
    from harness.tools.registry import schema_name

    if not obsidian_tools_enabled():
        return []
    out: list[RetraceTool] = []
    for tool in obsidian_chat_tools():
        name = schema_name(tool)
        if name:
            out.append(RetraceTool(name=name, connector="obsidian", schema=tool))
    return out


def _zotero_retrace() -> list[RetraceTool]:
    from integrations.zotero.client import zotero_tools_enabled
    from integrations.zotero.tools import zotero_chat_tools
    from harness.tools.registry import schema_name

    if not zotero_tools_enabled():
        return []
    out: list[RetraceTool] = []
    for tool in zotero_chat_tools():
        name = schema_name(tool)
        if name:
            out.append(RetraceTool(name=name, connector="zotero", schema=tool))
    return out


def injected_names(project_root: object, seen: list[str]) -> list[str]:
    names: list[str] = []
    raw = os.environ.get("SOPHON_INJECTED_TOOLS", "")
    for bit in raw.split(","):
        token = bit.strip()
        if token and token not in names:
            names.append(token)
    path = _root(project_root) / ".sophon" / "injected.yaml"
    if path.is_file():
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            data = {}
        listed = data.get("tools") if isinstance(data, dict) else None
        if isinstance(listed, list):
            for item in listed:
                token = str(item).strip()
                if token and token not in names:
                    names.append(token)
    for token in seen:
        if token and token not in names:
            names.append(token)
    # TODO: pull live LM Studio mcp.json schemas instead of name stubs
    return names


def mcp_status_lines(project_root: object) -> list[str]:
    lines = [f"mcp: {'on' if mcp_enabled() else 'off'} (SOPHON_MCP)"]
    if not mcp_enabled():
        return lines
    for server in configured_servers(project_root):
        server_id = str(server.get("id") or "")
        transport = str(server.get("transport") or "inprocess")
        live = _server_live(server)
        lines.append(f"  {server_id} transport={transport} {'on' if live else 'off'}")
    tools = retrace_tools(project_root)
    lines.append(f"  retrace tools: {len(tools)} (shadowed in the model schema when a NexusTool has the same name)")
    return lines
