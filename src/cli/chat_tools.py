from __future__ import annotations

from typing import Any

from cli.assist_tools import EDITOR_TOOL_SYSTEM_HINT, editor_chat_tools
from integrations.arxiv.tools import ARXIV_TOOL_SYSTEM_HINT, arxiv_chat_tools
from integrations.github import GITHUB_TOOL_SYSTEM_HINT, github_chat_tools
from integrations.google.tools import GOOGLE_TOOL_SYSTEM_HINT, google_chat_tools
from integrations.obsidian.tools import OBSIDIAN_TOOL_SYSTEM_HINT, obsidian_chat_tools
from integrations.overleaf.tools import OVERLEAF_TOOL_SYSTEM_HINT, overleaf_chat_tools
from integrations.shell.tools import SHELL_TOOL_SYSTEM_HINT, shell_chat_tools
from integrations.zotero.tools import ZOTERO_TOOL_SYSTEM_HINT, zotero_chat_tools
from processing.audio.speech.tools import SPEAK_TOOL, SST_TOOL_SYSTEM_HINT, TRANSCRIBE_TOOL, TTS_TOOL_SYSTEM_HINT
from processing.text.memory.tools import MEMORY_TOOL_SYSTEM_HINT, memory_chat_tools
from processing.text.skills import SKILL_TOOL_SYSTEM_HINT, skill_chat_tools
from harness.subagent.tools import SUBAGENT_TOOL_SYSTEM_HINT, subagent_chat_tools


def default_chat_tools(
    *,
    tts_tool: bool = False,
    sst_tool: bool = False,
    arxiv_tool: bool = False,
    obsidian_tool: bool = False,
    zotero_tool: bool = False,
    google_tool: bool = False,
    overleaf_tool: bool = False,
    shell_tool: bool = False,
    github_tool: bool = False,
    editor_tool: bool = False,
    memory_tool: bool = False,
    skill_tool: bool = False,
    subagent_tool: bool = False,
) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    if tts_tool:
        tools.append(SPEAK_TOOL)
    if sst_tool:
        tools.append(TRANSCRIBE_TOOL)
    if arxiv_tool:
        tools.extend(arxiv_chat_tools())
    if obsidian_tool:
        tools.extend(obsidian_chat_tools())
    if zotero_tool:
        tools.extend(zotero_chat_tools())
    if google_tool:
        tools.extend(google_chat_tools())
    if overleaf_tool:
        tools.extend(overleaf_chat_tools())
    if shell_tool:
        tools.extend(shell_chat_tools())
    if github_tool:
        tools.extend(github_chat_tools())
    if editor_tool:
        tools.extend(editor_chat_tools())
    if memory_tool:
        tools.extend(memory_chat_tools())
    if skill_tool:
        tools.extend(skill_chat_tools())
    if subagent_tool:
        tools.extend(subagent_chat_tools())
    return tools


def chat_tools_system_hint(
    *,
    tts_tool: bool = False,
    sst_tool: bool = False,
    arxiv_tool: bool = False,
    obsidian_tool: bool = False,
    zotero_tool: bool = False,
    google_tool: bool = False,
    overleaf_tool: bool = False,
    shell_tool: bool = False,
    github_tool: bool = False,
    editor_tool: bool = False,
    memory_tool: bool = False,
    skill_tool: bool = False,
    subagent_tool: bool = False,
) -> str | None:
    parts: list[str] = []
    if tts_tool:
        parts.append(TTS_TOOL_SYSTEM_HINT)
    if sst_tool:
        parts.append(SST_TOOL_SYSTEM_HINT)
    if arxiv_tool:
        parts.append(ARXIV_TOOL_SYSTEM_HINT)
    if obsidian_tool:
        parts.append(OBSIDIAN_TOOL_SYSTEM_HINT)
    if zotero_tool:
        parts.append(ZOTERO_TOOL_SYSTEM_HINT)
    if google_tool:
        parts.append(GOOGLE_TOOL_SYSTEM_HINT)
    if overleaf_tool:
        parts.append(OVERLEAF_TOOL_SYSTEM_HINT)
    if shell_tool:
        parts.append(SHELL_TOOL_SYSTEM_HINT)
    if github_tool:
        parts.append(GITHUB_TOOL_SYSTEM_HINT)
    if editor_tool:
        parts.append(EDITOR_TOOL_SYSTEM_HINT)
    if memory_tool:
        parts.append(MEMORY_TOOL_SYSTEM_HINT)
    if skill_tool:
        parts.append(SKILL_TOOL_SYSTEM_HINT)
    if subagent_tool:
        parts.append(SUBAGENT_TOOL_SYSTEM_HINT)
    if obsidian_tool and shell_tool:
        parts.append(
            "Vault tools read Obsidian notes. Shell tools navigate the workspace and run approved commands. "
            "Use vault_* for notes, shell_* for the workspace. Both are available."
        )
    if zotero_tool and shell_tool:
        parts.append(
            "Use zotero_* for the reference library. Use shell_* for the git/project workspace. "
            "Zotero is not a folder under the current directory."
        )
    if zotero_tool and obsidian_tool:
        parts.append(
            "Use zotero_* for papers and citations. Use vault_* for Obsidian notes."
        )
    if overleaf_tool and shell_tool:
        parts.append(
            "Use overleaf_* for Overleaf Git projects. Use shell_* for the local workspace. "
            "Overleaf clones under data/overleaf are not the workspace root."
        )
    if overleaf_tool and obsidian_tool:
        parts.append(
            "Use overleaf_* for Overleaf LaTeX projects. Use vault_* for Obsidian notes."
        )
    if google_tool and shell_tool:
        parts.append(
            "Use gmail_* / drive_* / bookmarks_tree / web_search for Google data. "
            "Use shell_* for the local workspace."
        )
    if editor_tool and shell_tool:
        parts.append(
            "Use editor_propose_edit for source changes. Use shell_exec only for running commands, not for writing files."
        )
    if not parts:
        return None
    return " ".join(parts)
