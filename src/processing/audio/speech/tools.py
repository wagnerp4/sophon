from __future__ import annotations

from typing import Any

SPEAK_TOOL_NAME = "speak"

SPEAK_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SPEAK_TOOL_NAME,
        "description": (
            "Speak aloud via orodruin TTS (Pipecat Kokoro by default). "
            "Call this when the user asks to hear something, read something aloud, "
            "run TTS, or play audio for chat text or a workspace document. "
            "Prefer path for files; turn for a prior chat message; text for short phrases."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Literal text to speak.",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Workspace-relative or absolute path to a text/markdown document to speak."
                    ),
                },
                "turn": {
                    "type": "string",
                    "description": (
                        "Chat turn to speak: last|assistant|user|<index> "
                        "(index into user/assistant messages)."
                    ),
                },
            },
            "additionalProperties": False,
        },
    },
}

TTS_TOOL_SYSTEM_HINT = (
    "You can call the speak tool to play TTS audio in the user's terminal. "
    "Use it for natural-language requests like 'read that aloud', 'tts this', "
    "or 'speak the file notes.md'. Pass path for documents, turn for prior chat "
    "content, or text for short strings. Do not invent file paths."
)

TRANSCRIBE_TOOL_NAME = "transcribe"

TRANSCRIBE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TRANSCRIBE_TOOL_NAME,
        "description": (
            "Transcribe an audio file via orodruin SST (Qwen3-ASR by default). "
            "Call this when the user asks to transcribe, dictate from a recording, "
            "or convert speech in a workspace audio file to text."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative or absolute path to an audio file.",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}

SST_TOOL_SYSTEM_HINT = (
    "You can call the transcribe tool to run Qwen ASR on a local audio file. "
    "Pass path to a wav/mp3/flac/ogg/m4a file. Do not invent file paths."
)


def default_chat_tools(*, tts_tool: bool = True, sst_tool: bool = False) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    if tts_tool:
        tools.append(SPEAK_TOOL)
    if sst_tool:
        tools.append(TRANSCRIBE_TOOL)
    return tools
