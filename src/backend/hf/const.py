from __future__ import annotations

BUILTIN_SPECIAL_TOKEN_PATTERNS: tuple[str, ...] = (
    r"</s>",
    r"<s>",
    r"<\|endoftext\|>",
    r"<\|end_of_text\|>",
    r"<\|begin_of_text\|>",
    r"<\|eot_id\|>",
    r"<\|start_header_id\|>",
    r"<\|end_header_id\|>",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"<\|user\|>",
    r"<\|assistant\|>",
    r"<\|system\|>",
    r"<\|turn>[a-zA-Z_]+\n",
    r"<\|channel>[a-zA-Z_]+\n",
    r"<\|think\|>",
    r"<\|tool>",
    r"<\|tool_call>[^\s]*",
    r"<\|tool_response>",
    r"<\|image\|>",
    r"<\|audio\|>",
    r"<\|video\|>",
    r"<bos>",
    r"<eos>",
    r"<pad>",
)

DEFAULT_LOCAL_DIR = "models/meta-llama-Llama-2-7b-chat-hf"
