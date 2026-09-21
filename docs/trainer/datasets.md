# Datasets

Training is instruction SFT packed as Alpaca `### Instruction` / `### Response`. `/eval model` is multiple-choice scoring on the same sources. Those two paths do not share a loader.

`/finetune-datasets` prints id, source, and `max_examples`. Default cap is 500 rows. `max_examples=0` uses the full split.

## Train loaders

- `gsm8k_instructions` — Hub `qwedsacf/grade-school-math-instructions`, split `train`, fields `INSTRUCTION` / `RESPONSE`.
- `hellaswag` — local jsonl. Looks for `data/hellaswag/data/hellaswag_{split}.jsonl`, then `data/benchmarks/hellaswag/hellaswag_{split}.jsonl`. Instruction is activity + context. Response is the gold ending (`label`).
- `mmlu` — local CSV. Looks for a combined `mmlu_{split}.csv` or a subject directory such as `data/test/data/{split}` (eval dump). Instruction is question plus A-D. Response is `letter. answer text`. Split aliases: `train` also tries `auxiliary_train` and `dev`.

Eval manifests stay under [`data/benchmarks/hellaswag.yaml`](../../data/benchmarks/hellaswag.yaml) and [`data/benchmarks/mmlu.yaml`](../../data/benchmarks/mmlu.yaml). `/eval model hellaswag` and `/eval model mmlu` do not train.

Sources:

- [GSM8K](https://arxiv.org/abs/2110.14168)
- [HellaSwag](https://arxiv.org/abs/1905.07830)
- [MMLU](https://arxiv.org/abs/2009.03300)
