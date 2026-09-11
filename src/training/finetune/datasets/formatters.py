from __future__ import annotations

ALPACA_INSTRUCTION_RESPONSE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""


def format_alpaca_instruction_response(
    instruction: str,
    response: str,
    *,
    eos_token: str,
) -> str:
    return ALPACA_INSTRUCTION_RESPONSE.format(instruction, response) + eos_token


def format_inference_prompt(instruction: str) -> str:
    return ALPACA_INSTRUCTION_RESPONSE.format(instruction, "")
