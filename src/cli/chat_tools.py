from __future__ import annotations

from typing import Any

from integrations.obsidian.tools import OBSIDIAN_TOOL_SYSTEM_HINT, obsidian_chat_tools
from integrations.shell.tools import SHELL_TOOL_SYSTEM_HINT, shell_chat_tools
from processing.audio.speech.tools import SPEAK_TOOL, SST_TOOL_SYSTEM_HINT, TRANSCRIBE_TOOL, TTS_TOOL_SYSTEM_HINT


def default_chat_tools(
    *,
    tts_tool: bool = False,
    sst_tool: bool = False,
    obsidian_tool: bool = False,
    shell_tool: bool = False,
) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    if tts_tool:
        tools.append(SPEAK_TOOL)
    if sst_tool:
        tools.append(TRANSCRIBE_TOOL)
    if obsidian_tool:
        tools.extend(obsidian_chat_tools())
    if shell_tool:
        tools.extend(shell_chat_tools())
    return tools


def chat_tools_system_hint(
    *,
    tts_tool: bool = False,
    sst_tool: bool = False,
    obsidian_tool: bool = False,
    shell_tool: bool = False,
) -> str | None:
    parts: list[str] = []
    if tts_tool:
        parts.append(TTS_TOOL_SYSTEM_HINT)
    if sst_tool:
        parts.append(SST_TOOL_SYSTEM_HINT)
    if obsidian_tool:
        parts.append(OBSIDIAN_TOOL_SYSTEM_HINT)
    if shell_tool:
        parts.append(SHELL_TOOL_SYSTEM_HINT)
    if obsidian_tool and shell_tool:
        parts.append(
            "Vault tools read Obsidian notes. Shell tools navigate the local disk and run commands. "
            "Use vault_* for notes, shell_* for the workspace/filesystem. Both are available."
        )
    if not parts:
        return None
    return " ".join(parts)
