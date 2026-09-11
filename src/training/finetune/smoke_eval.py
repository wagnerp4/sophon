from __future__ import annotations

import json
from typing import Any

from training.finetune.datasets.formatters import format_inference_prompt

SMOKE_QUESTIONS: tuple[str, ...] = (
    "I have 10 apples, my brother took half of them from me, I lost 1, and my friend gave me 3. How many do I have now?",
    "I earn five euros per hour. I worked two hours yesterday and five hours today. How much did I earn in total?",
    "In year 2000 I was 20 years old. My sister is 5 years younger than me. How old is she in 2020?",
)


def run_smoke_eval(
    model: object,
    tokenizer: object,
    *,
    max_new_tokens: int = 70,
) -> dict[str, Any]:
    import torch

    if hasattr(model, "eval"):
        model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results: list[dict[str, str]] = []
    for idx, question in enumerate(SMOKE_QUESTIONS, start=1):
        prompt = format_inference_prompt(question)
        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
        results.append({"question": question, "response": text})
    return {"questions": results}


def write_smoke_eval(path: object, payload: dict[str, Any]) -> None:
    from pathlib import Path

    p = Path(path)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
